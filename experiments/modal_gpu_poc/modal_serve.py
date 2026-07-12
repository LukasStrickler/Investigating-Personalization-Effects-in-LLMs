#!/usr/bin/env python3
"""Deploy a subject model on Modal as an OpenAI-compatible vLLM server.

WHY THIS EXISTS
    The personalization experiments self-host their subject models through the
    repo's `vllm` provider (an OpenAI-compatible HTTP endpoint that the Slurm
    launcher starts on the university cluster). When that cluster is unavailable
    this deploys the SAME kind of endpoint on Modal's serverless GPUs instead, so
    the existing Stage-1 pipeline — `ExperimentRunner` → matrix CSV — runs
    unchanged against the `modal` provider.

    vLLM's OpenAI server gives continuous batching + prefix caching for free.

DEPLOY
    set -a; source .env; set +a                       # Modal CLI auth (MODAL_TOKEN_*)
    MODAL=/path/to/.venv-modal/bin/modal
    # default: Gemma 4 E2B on an L4
    $MODAL deploy experiments/modal_gpu_poc/modal_serve.py
    # a bigger subject on a bigger GPU:
    MODAL_SERVE_MODEL_ID=google/gemma-4-31b-it MODAL_SERVE_NAME=gemma-4-31b \
      MODAL_SERVE_GPU=A100-80GB \
      $MODAL deploy experiments/modal_gpu_poc/modal_serve.py

    `modal deploy` prints the endpoint URL. Point the inference client at it:
        export MODAL_BASE_URL="https://<workspace>--pers-subject-serve-serve.modal.run/v1"
        export MODAL_API_KEY="EMPTY"      # or the token in the modal-serve-token secret

TEAR DOWN
    $MODAL app stop pers-subject-serve     # or let it scale to zero when idle

The app scales to zero after `MODAL_SERVE_SCALEDOWN` idle seconds (default 300),
so a deployed-but-unused server costs nothing beyond storage.
"""

from __future__ import annotations

import os
import subprocess

import modal

# --------------------------------------------------------------------------- #
# Config (deploy-time env knobs)
# --------------------------------------------------------------------------- #
APP_NAME = os.getenv("MODAL_SERVE_APP_NAME", "pers-subject-serve")
MODEL_ID = os.getenv("MODAL_SERVE_MODEL_ID", "google/gemma-4-E2B-it")  # HF repo
# --served-model-name: MUST equal the `model:` field of the matching alias in
# config/inference.modal.example.yaml (e.g. gemma-4-e2b).
SERVED_NAME = os.getenv("MODAL_SERVE_NAME", "gemma-4-e2b")
GPU = os.getenv("MODAL_SERVE_GPU", "L4")            # L4 fits ~5B bf16; bigger models need A100/H100
MAX_MODEL_LEN = int(os.getenv("MODAL_SERVE_MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.getenv("MODAL_SERVE_GPU_MEM_UTIL", "0.90"))
SCALEDOWN = int(os.getenv("MODAL_SERVE_SCALEDOWN", "300"))     # idle seconds before scale-to-zero
MIN_CONTAINERS = int(os.getenv("MODAL_SERVE_MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.getenv("MODAL_SERVE_MAX_CONTAINERS", "1"))
# How many HTTP requests Modal proxies to ONE container at once. This is the
# crux of throughput: a @modal.web_server defaults to max_inputs=1, so Modal
# hands vLLM a single request at a time and vLLM's continuous batching never
# engages (the server shows "Running: 1 reqs" no matter how many the client
# sends). Set this >= the client's provider max_concurrency so vLLM actually
# batches; vLLM queues anything beyond what fits in the KV cache.
MAX_INPUTS = int(os.getenv("MODAL_SERVE_MAX_INPUTS", "100"))
SERVE_PORT = 8000
HF_CACHE = "/cache/hf"

# Optional bearer-token auth. OFF by default → the endpoint is open (fine for a
# short private run; the URL is unguessable). To enforce a token, create the secret
# and opt in at deploy time:
#   modal secret create modal-serve-token VLLM_SERVE_TOKEN=$(openssl rand -hex 24)
#   MODAL_SERVE_AUTH=1 modal deploy experiments/modal_gpu_poc/modal_serve.py
# then set the client's MODAL_API_KEY to the same token (else MODAL_API_KEY=EMPTY).
TOKEN_SECRET = os.getenv("MODAL_SERVE_TOKEN_SECRET", "modal-serve-token")
AUTH_ENABLED = os.getenv("MODAL_SERVE_AUTH", "") == "1"

# Optional Hugging Face token for GATED subject models (e.g. Llama, gated Qwen/
# Mistral variants). Gemma 4 E2B and Mistral-7B-Instruct-v0.3 are ungated, so the
# default needs no token. To serve a gated repo, store the token in a secret and
# opt in at deploy time (vLLM/huggingface_hub read HF_TOKEN from the env):
#   modal secret create huggingface-token HF_TOKEN=hf_...
#   MODAL_SERVE_HF_TOKEN=1 MODAL_SERVE_MODEL_ID=meta-llama/... modal deploy …
HF_TOKEN_SECRET = os.getenv("MODAL_SERVE_HF_TOKEN_SECRET", "huggingface-token")
HF_TOKEN_ENABLED = os.getenv("MODAL_SERVE_HF_TOKEN", "") == "1"

# --------------------------------------------------------------------------- #
# Image — enforce-eager + Triton backend avoids flashinfer's nvcc JIT on the slim
# image (no CUDA devel toolchain in debian_slim).
#
# IMPORTANT: the model-selection knobs (MODEL_ID / SERVED_NAME / …) are BAKED into
# the image env here. Modal re-imports this module *inside the container* with only
# the image env present — the deploy shell's MODAL_SERVE_* vars do NOT propagate —
# so without baking them, the container's module-level os.getenv() would silently
# revert every deploy to the Gemma defaults regardless of what you asked for.
# (GPU/scaledown/max_containers are decorator args evaluated locally, so they don't
# need this; only values read *inside* serve() at container runtime do.)
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm",
        "nvidia-ml-py>=12.535",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HOME": HF_CACHE,
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            # Deploy-time model selection → reaches the container via the image env.
            "MODAL_SERVE_MODEL_ID": MODEL_ID,
            "MODAL_SERVE_NAME": SERVED_NAME,
            "MODAL_SERVE_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "MODAL_SERVE_GPU_MEM_UTIL": str(GPU_MEM_UTIL),
        }
    )
)

app = modal.App(APP_NAME, image=image)
hf_cache_vol = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)


def _serve_secrets() -> list[modal.Secret]:
    """Only the secrets explicitly opted into. `Secret.from_name` is lazy, so naming
    a missing secret fails at container start — hence both are off by default:

    - bearer-token auth   (MODAL_SERVE_AUTH=1     → secret ``modal-serve-token``)
    - Hugging Face token  (MODAL_SERVE_HF_TOKEN=1 → secret ``huggingface-token``, gated repos)
    """
    secrets: list[modal.Secret] = []
    if AUTH_ENABLED:
        secrets.append(modal.Secret.from_name(TOKEN_SECRET))
    if HF_TOKEN_ENABLED:
        secrets.append(modal.Secret.from_name(HF_TOKEN_SECRET))
    return secrets


@app.function(
    gpu=GPU,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=_serve_secrets(),
    timeout=24 * 60 * 60,
    scaledown_window=SCALEDOWN,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=MAX_INPUTS)  # let one container serve many requests → vLLM batches
@modal.web_server(port=SERVE_PORT, startup_timeout=15 * 60)
def serve() -> None:
    """Launch `vllm serve` as an OpenAI-compatible endpoint.

    @modal.web_server expects this to RETURN while the subprocess keeps listening,
    so we Popen (not wait). Modal proxies HTTPS on the printed *.modal.run URL to
    SERVE_PORT and health-probes it until vLLM finishes loading (~2-3 min cold).
    """
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(SERVE_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--enforce-eager",          # skip CUDA-graph capture (no nvcc on slim image)
        "--enable-prefix-caching",  # reuse shared persona-history prefills across Q1/Q2
    ]
    token = os.environ.get("VLLM_SERVE_TOKEN", "").strip()
    if token:
        cmd += ["--api-key", token]
    print(f"[modal-serve] {' '.join(cmd)}", flush=True)  # noqa: T201
    subprocess.Popen(cmd)
