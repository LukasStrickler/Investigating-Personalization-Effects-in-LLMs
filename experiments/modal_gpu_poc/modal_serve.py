#!/usr/bin/env python3
"""Deploy a subject model on Modal as an OpenAI-compatible vLLM server.

WHY THIS EXISTS
    The personalization experiments self-host subject models through the repo's
    ``vllm`` provider (OpenAI-compatible HTTP on Slurm). When Slurm is unavailable,
    this deploys the same kind of endpoint on Modal so ``run_behavioral_audit_modal.py``
    runs unchanged against the ``modal`` inference provider.

CHANGES SINCE THE INITIAL MODAL PR (#28)
    The first version used ``@modal.web_server`` and worked for the Gemma 4 E2B run in
    some workspaces, but during the Ministral 3 8B deploy we hit two production bugs:

    1. **Request routing** — with ``@modal.web_server``, the container started and vLLM
       loaded, yet client requests hung indefinitely (nothing reached vLLM). Modal's ASGI
       path (``@modal.asgi_app``) routes reliably, so we run ``vllm serve`` on localhost
       and forward through a thin Starlette reverse proxy. This is *not* a revert; it
       fixes broken ingress while keeping the same OpenAI API surface.

    2. **Model/env mismatch in-container** — Modal re-imports this module inside the
       container without the deploy shell's ``MODAL_SERVE_*`` vars. Values needed at
       runtime are baked into the image ``.env()`` and read via ``_read_runtime_env()``.

    Other intentional changes (not reversions):
    - HF cache volume ``gemma4-hf-cache`` → ``pers-hf-cache`` (shared across models).
    - ``ffmpeg`` + ``vllm>=0.20.0`` for current vLLM on debian-slim.
    - Ministral 3 needs ``mistral_common`` and ``--*-format mistral`` vLLM flags.
    - Gemma keeps ``TRITON_ATTN``; Ministral leaves the attention backend unset.

DEPLOY (default — Gemma 4 E2B on L4)
    set -a; source .env; set +a
    modal deploy experiments/modal_gpu_poc/modal_serve.py
    export MODAL_BASE_URL="https://<workspace>--pers-subject-serve-serve.modal.run/v1"
    # Copy the URL printed by `modal deploy` — Modal appends the function name `serve`
    # to the hostname (…-serve-serve.modal.run for app pers-subject-serve).

DEPLOY (gated model, e.g. Ministral 3 8B on L4)
    python experiments/modal_gpu_poc/setup_modal_hf.py
    MODAL_SERVE_APP_NAME=pers-ministral3-8b-serve \\
    MODAL_SERVE_MODEL_ID=mistralai/Ministral-3-8B-Instruct-2512 \\
    MODAL_SERVE_NAME=ministral-3-8b \\
    MODAL_SERVE_GPU=L4 \\
    MODAL_SERVE_MAX_MODEL_LEN=8192 \\
    MODAL_SERVE_HF_TOKEN=1 \\
    modal deploy experiments/modal_gpu_poc/modal_serve.py

Deploy knobs (all optional except as noted): ``MODAL_SERVE_APP_NAME``,
``MODAL_SERVE_MODEL_ID``, ``MODAL_SERVE_NAME``, ``MODAL_SERVE_GPU``,
``MODAL_SERVE_HF_TOKEN=1`` (gated repos — run setup_modal_hf.py first),
``MODAL_SERVE_STARTUP_TIMEOUT``, ``MODAL_SERVE_GPU_LOG=1``.

TEAR DOWN
    modal app stop <MODAL_SERVE_APP_NAME>

The app scales to zero after ``MODAL_SERVE_SCALEDOWN`` idle seconds (default 300).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import modal

# --------------------------------------------------------------------------- #
# Config (deploy-time env knobs)
# --------------------------------------------------------------------------- #
APP_NAME = os.getenv("MODAL_SERVE_APP_NAME", "pers-subject-serve")
MODEL_ID = os.getenv("MODAL_SERVE_MODEL_ID", "google/gemma-4-E2B-it")
SERVED_NAME = os.getenv("MODAL_SERVE_NAME", "gemma-4-e2b")
GPU = os.getenv("MODAL_SERVE_GPU", "L4")
MAX_MODEL_LEN = int(os.getenv("MODAL_SERVE_MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.getenv("MODAL_SERVE_GPU_MEM_UTIL", "0.90"))
TENSOR_PARALLEL = int(os.getenv("MODAL_SERVE_TENSOR_PARALLEL", "1"))
ATTENTION_BACKEND = os.getenv("MODAL_SERVE_ATTENTION_BACKEND", "").strip()
MAX_NUM_SEQS = int(os.getenv("MODAL_SERVE_MAX_NUM_SEQS", "128"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("MODAL_SERVE_MAX_NUM_BATCHED_TOKENS", "16384"))
ENFORCE_EAGER = os.getenv("MODAL_SERVE_ENFORCE_EAGER", "1") == "1"
GPU_LOG = os.getenv("MODAL_SERVE_GPU_LOG", "") == "1"
VLLM_SPEC = os.getenv("MODAL_SERVE_VLLM_SPEC", "vllm>=0.20.0")
SCALEDOWN = int(os.getenv("MODAL_SERVE_SCALEDOWN", "300"))
MIN_CONTAINERS = int(os.getenv("MODAL_SERVE_MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.getenv("MODAL_SERVE_MAX_CONTAINERS", "1"))
MAX_INPUTS = int(os.getenv("MODAL_SERVE_MAX_INPUTS", "100"))
STARTUP_TIMEOUT = int(os.getenv("MODAL_SERVE_STARTUP_TIMEOUT", "2400"))
SERVE_PORT = 8000
HF_CACHE = "/cache/hf"
# Renamed from gemma4-hf-cache (PR #28) — shared across subject models; old volume is orphaned.
HF_VOLUME = os.getenv("MODAL_SERVE_HF_VOLUME", "pers-hf-cache")

TOKEN_SECRET = os.getenv("MODAL_SERVE_TOKEN_SECRET", "modal-serve-token")
AUTH_ENABLED = os.getenv("MODAL_SERVE_AUTH", "") == "1"
HF_TOKEN_SECRET = os.getenv("MODAL_SERVE_HF_TOKEN_SECRET", "huggingface-token")
HF_TOKEN_ENABLED = os.getenv("MODAL_SERVE_HF_TOKEN", "") == "1"

# Image env must match deploy-time secret flags (Modal re-imports this module in-container).
_IS_MINISTRAL3 = "Ministral-3" in MODEL_ID or "ministral-3" in SERVED_NAME.lower()
_default_attn = "" if ATTENTION_BACKEND else ("TRITON_ATTN" if not _IS_MINISTRAL3 else "")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")  # vLLM 0.20+ needs torchcodec/ffmpeg on debian-slim
    .pip_install(
        VLLM_SPEC,
        *(["mistral_common>=1.11.0"] if _IS_MINISTRAL3 else []),
        "nvidia-ml-py>=12.535",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "MODAL_SERVE_MODEL_ID": MODEL_ID,
            "MODAL_SERVE_NAME": SERVED_NAME,
            "MODAL_SERVE_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "MODAL_SERVE_GPU_MEM_UTIL": str(GPU_MEM_UTIL),
            "MODAL_SERVE_TENSOR_PARALLEL": str(TENSOR_PARALLEL),
            "MODAL_SERVE_ATTENTION_BACKEND": ATTENTION_BACKEND,
            "MODAL_SERVE_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
            "MODAL_SERVE_MAX_NUM_BATCHED_TOKENS": str(MAX_NUM_BATCHED_TOKENS),
            "MODAL_SERVE_ENFORCE_EAGER": "1" if ENFORCE_EAGER else "0",
            "MODAL_SERVE_GPU_LOG": "1" if GPU_LOG else "0",
            "MODAL_SERVE_HF_TOKEN": "1" if HF_TOKEN_ENABLED else "0",
            "MODAL_SERVE_AUTH": "1" if AUTH_ENABLED else "0",
            **({"VLLM_ATTENTION_BACKEND": _default_attn} if _default_attn else {}),
        }
    )
)

app = modal.App(APP_NAME, image=image)
hf_cache_vol = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)


def _serve_secrets() -> list[modal.Secret]:
    secrets: list[modal.Secret] = []
    if AUTH_ENABLED:
        secrets.append(modal.Secret.from_name(TOKEN_SECRET))
    if HF_TOKEN_ENABLED:
        secrets.append(modal.Secret.from_name(HF_TOKEN_SECRET))
    return secrets


def _read_runtime_env(name: str, default: str = "") -> str:
    """Image env is authoritative inside the container (deploy shell vars do not propagate)."""
    return os.environ.get(name, default)


def _needs_mistral_vllm_format(model_id: str) -> bool:
    return "Ministral-3" in model_id or "Mistral-Small-3" in model_id


def _gpu_monitor_loop() -> None:
    while True:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            print(f"[gpu-monitor] {out}", flush=True)  # noqa: T201
        except Exception as exc:  # noqa: BLE001
            print(f"[gpu-monitor] nvidia-smi failed: {exc}", flush=True)  # noqa: T201
        time.sleep(10)


def _vllm_cmd(model_id: str, served_name: str, **opts: object) -> list[str]:
    cmd = [
        "vllm", "serve", model_id,
        "--served-model-name", served_name,
        "--host", "0.0.0.0",
        "--port", str(SERVE_PORT),
        "--max-model-len", str(opts["max_model_len"]),
        "--gpu-memory-utilization", str(opts["gpu_mem_util"]),
        "--tensor-parallel-size", str(opts["tensor_parallel"]),
        "--max-num-seqs", str(opts["max_num_seqs"]),
        "--max-num-batched-tokens", str(opts["max_batched"]),
        "--enable-prefix-caching",
    ]
    if opts.get("attention_backend"):
        cmd += ["--attention-backend", str(opts["attention_backend"])]
    if opts.get("enforce_eager"):
        cmd.append("--enforce-eager")
    if _needs_mistral_vllm_format(model_id):
        cmd += [
            "--tokenizer-mode", "mistral",
            "--config-format", "mistral",
            "--load-format", "mistral",
            "--tool-call-parser", "mistral",
            "--enable-auto-tool-choice",
        ]
    return cmd


@app.function(
    gpu=GPU,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=_serve_secrets(),
    timeout=24 * 60 * 60,
    scaledown_window=SCALEDOWN,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve():
    """Run ``vllm serve`` on localhost and expose it through a Starlette ASGI
    reverse proxy served by Modal's ASGI runtime.

    Why a proxy instead of ``@modal.web_server``: Modal's raw TCP port-proxy
    (``web_server``) is not routed in every workspace — requests hang with no
    container ever receiving them — whereas the ASGI path (``asgi_app`` /
    ``fastapi_endpoint``) is. Wrapping vLLM in a thin ASGI proxy makes the
    endpoint portable across workspaces with no change for OpenAI clients: the
    served model name, routes (``/v1/chat/completions`` …) and auth header all
    pass straight through to vLLM on ``127.0.0.1:{SERVE_PORT}``.
    """
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.background import BackgroundTask
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    model_id = _read_runtime_env("MODAL_SERVE_MODEL_ID", MODEL_ID)
    served_name = _read_runtime_env("MODAL_SERVE_NAME", SERVED_NAME)
    cmd = _vllm_cmd(
        model_id,
        served_name,
        max_model_len=_read_runtime_env("MODAL_SERVE_MAX_MODEL_LEN", str(MAX_MODEL_LEN)),
        gpu_mem_util=_read_runtime_env("MODAL_SERVE_GPU_MEM_UTIL", str(GPU_MEM_UTIL)),
        tensor_parallel=int(_read_runtime_env("MODAL_SERVE_TENSOR_PARALLEL", str(TENSOR_PARALLEL))),
        attention_backend=_read_runtime_env("MODAL_SERVE_ATTENTION_BACKEND", ATTENTION_BACKEND),
        max_num_seqs=_read_runtime_env("MODAL_SERVE_MAX_NUM_SEQS", str(MAX_NUM_SEQS)),
        max_batched=_read_runtime_env("MODAL_SERVE_MAX_NUM_BATCHED_TOKENS", str(MAX_NUM_BATCHED_TOKENS)),
        enforce_eager=_read_runtime_env("MODAL_SERVE_ENFORCE_EAGER", "1" if ENFORCE_EAGER else "0") == "1",
    )
    gpu_log = _read_runtime_env("MODAL_SERVE_GPU_LOG", "1" if GPU_LOG else "0") == "1"

    token = os.environ.get("VLLM_SERVE_TOKEN", "").strip()
    if token:
        cmd += ["--api-key", token]

    print(f"[modal-serve] model={model_id} gpu={GPU}", flush=True)  # noqa: T201
    print(f"[modal-serve] {' '.join(cmd)}", flush=True)  # noqa: T201

    if gpu_log:
        threading.Thread(target=_gpu_monitor_loop, daemon=True).start()

    # vLLM runs as a background process on localhost; the proxy forwards to it.
    subprocess.Popen(cmd)  # noqa: S603

    upstream = f"http://127.0.0.1:{SERVE_PORT}"
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(None))
    ready = asyncio.Event()
    # Hop-by-hop headers must not be forwarded (RFC 7230 §6.1); Modal/uvicorn/
    # httpx set their own content-length + transfer-encoding per hop.
    hop = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}

    async def _wait_ready() -> None:
        if ready.is_set():
            return
        waited = 0.0
        while not ready.is_set() and waited < STARTUP_TIMEOUT:
            try:
                if (await client.get("/health")).status_code == 200:
                    ready.set()
                    print("[modal-serve] vLLM ready", flush=True)  # noqa: T201
                    return
            except Exception:  # noqa: BLE001 — vLLM not up yet
                pass
            await asyncio.sleep(2.0)
            waited += 2.0

    # Starlette (not FastAPI) endpoint: the request is passed positionally, so
    # there is no annotation introspection — which matters because this module
    # uses ``from __future__ import annotations`` (PEP 563) and FastAPI would try
    # to resolve the stringized ``Request`` annotation against module globals and
    # fail (it is imported locally here), mis-classifying it as a query param.
    async def _proxy(request):
        await _wait_ready()
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        req_headers = [
            (k.decode(), v.decode())
            for k, v in request.headers.raw
            if k.decode().lower() not in hop
        ]
        up_req = client.build_request(
            request.method, url, headers=req_headers, content=await request.body()
        )
        up_resp = await client.send(up_req, stream=True)
        resp_headers = {
            k.decode(): v.decode()
            for k, v in up_resp.headers.raw
            if k.decode().lower() not in hop
        }
        return StreamingResponse(
            up_resp.aiter_raw(),
            status_code=up_resp.status_code,
            headers=resp_headers,
            background=BackgroundTask(up_resp.aclose),
        )

    return Starlette(
        routes=[
            Route(
                "/{path:path}",
                _proxy,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            )
        ]
    )
