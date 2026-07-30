# Modal setup and deploy

Setup/deploy scripts for hosting a subject model on [Modal](https://modal.com).
You run these yourself to sync the HF secret and deploy a vLLM endpoint. Experiment
runners under `experiments/` do not import this folder; they call the deployed URL
through `src/inference`.

| Role | Command |
| --- | --- |
| Setup (HF secret) | `uv run python scripts/modal/setup_modal_hf.py` (needs `uv sync --extra modal`) |
| Deploy (GPU serve) | `uv run modal deploy scripts/modal/modal_serve.py` |
| Usage | `experiments/.../run_*_modal.py` + `MODAL_BASE_URL` |

Use this when Slurm is unavailable. Modal serves an OpenAI-compatible vLLM endpoint
for the repo's `modal` inference provider.

Stage 1 writes matrix CSVs to the gitignored `logs/` working area. Use
`export_results.py` to copy finished Stage-1 artifacts into a committable
`results_<tag>/` directory (same layout as `results_full001/`, ...).

Stage 2 (LLM judge) is **not** part of the personalized path above - run it
separately with `run_behavioral_audit.py` (`STAGE2_ONLY=True`) once Stage-1 CSVs
exist. For persona-free **baseline** runs, Stage 2 runs in the same process; see
§ Baseline runs.

## Workflow

```bash
cd <repo-root>
uv sync --extra modal
set -a; source .env; set +a    # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET for deploy

# 1. Deploy the subject-model server (prints a *.modal.run URL)
uv run modal deploy scripts/modal/modal_serve.py
export MODAL_BASE_URL="https://<workspace>--pers-subject-serve-serve.modal.run/v1"
export MODAL_API_KEY="EMPTY"
# Always copy the URL from `modal deploy` output - see § Modal URL pattern below.

# 2. Point the inference client at Modal
cp config/inference.modal.example.yaml config/inference.yaml
# Default example uses max_concurrency: 100 (single L4). Tune if you scale out - § Throughput.

# 3. Stage 1 - generate subject responses → logs/...-stage1/*.csv
python experiments/behavioral_audit/run_behavioral_audit_modal.py \
    --run-tag full001-e2b --subject-alias gemma-4-e2b_modal
# Smoke: add --limit 4

# 4. Export Stage-1 CSVs from logs/ into a committable results dir
python experiments/behavioral_audit/export_results.py \
    --run-tag full001-e2b --subject-alias gemma-4-e2b_modal

# 5. Tear down (or let it scale to zero after MODAL_SERVE_SCALEDOWN idle seconds)
modal app stop pers-subject-serve
```

## Gated models (e.g. Ministral 3 8B)

Gated HuggingFace repos need a token synced to Modal before deploy:

```bash
# Accept license + add HF_TOKEN to .env, then:
.venv/bin/python scripts/modal/setup_modal_hf.py

MODAL_SERVE_APP_NAME=pers-ministral3-8b-serve \
MODAL_SERVE_MODEL_ID=mistralai/Ministral-3-8B-Instruct-2512 \
MODAL_SERVE_NAME=ministral-3-8b \
MODAL_SERVE_GPU=L4 \
MODAL_SERVE_MAX_MODEL_LEN=8192 \
MODAL_SERVE_MAX_CONTAINERS=6 \
MODAL_SERVE_MAX_INPUTS=30 \
MODAL_SERVE_HF_TOKEN=1 \
modal deploy scripts/modal/modal_serve.py

export MODAL_BASE_URL="https://<workspace>--pers-ministral3-8b-serve-serve.modal.run/v1"

# Multi-L4 deploy - raise client concurrency to match (30 × 6 = 180):
# edit config/inference.yaml → providers.modal.max_concurrency: 180

python experiments/behavioral_audit/run_behavioral_audit_modal.py \
    --run-tag full001-ministral3-8b --subject-alias ministral-3-8b_modal

python experiments/behavioral_audit/export_results.py \
    --run-tag full001-ministral3-8b --subject-alias ministral-3-8b_modal
```

OpenRouter smoke test (no GPU, **not** weight-identical to the Modal subject):
`--subject-alias ministral-3-8b_openrouter`.

## Baseline runs (persona-free control)

`run_behavioral_audit_modal.py` (above) replays each persona's history - the
**personalized** condition. The **baseline** is the persona-free control: the same
two probes asked with *no* history (a short framing turn in its place), repeated
`N_ITERATIONS=50` times each → 100 Stage-1 responses per model. It measures the
model's default recommendation distribution, to compare against the personalized runs.

Two differences from the full path above:

- Use `run_behavioral_audit_baseline_modal.py` (not `run_behavioral_audit_modal.py`).
- It runs **Stage 1 and Stage 2 in one process** - the OpenRouter judge alias
  (`gpt-4o-mini_paid`) is already in `config/inference.modal.example.yaml`, so no
  separate `STAGE2_ONLY` step is needed.

Deploy the subject exactly as above, then run the baseline. Pass exactly **one**
`--subject-alias` per run (each Modal model is a separate deploy behind its own
`MODAL_BASE_URL`):

```bash
set -a; source .env; set +a
cp config/inference.modal.example.yaml config/inference.yaml
export MODAL_API_KEY="EMPTY"

# ── Gemma 4 E2B - default deploy (pers-subject-serve, 1× L4) ──
modal deploy scripts/modal/modal_serve.py
export MODAL_BASE_URL="https://<workspace>--pers-subject-serve-serve.modal.run/v1"
python experiments/behavioral_audit/run_behavioral_audit_baseline_modal.py \
    --run-tag baseline-e2b --subject-alias gemma-4-e2b_modal
python experiments/behavioral_audit/export_results.py \
    --run-tag baseline-e2b --subject-alias gemma-4-e2b_modal
modal app stop pers-subject-serve

# ── Ministral 3 8B - gated (see § Gated models for setup_modal_hf.py + deploy env) ──
export MODAL_BASE_URL="https://<workspace>--pers-ministral3-8b-serve-serve.modal.run/v1"
python experiments/behavioral_audit/run_behavioral_audit_baseline_modal.py \
    --run-tag baseline-ministral3-8b --subject-alias ministral-3-8b_modal
python experiments/behavioral_audit/export_results.py \
    --run-tag baseline-ministral3-8b --subject-alias ministral-3-8b_modal
modal app stop pers-ministral3-8b-serve
```

Each run writes `results_baseline-<tag>/` with the **same layout** as
`results_full001-*/` - per question a Stage-1 matrix CSV (`...-q{1,2}-stage1/*.csv` +
`.meta.json`) and a Stage-2 `...-q{1,2}-stage2.judgments.csv`, plus one
`EXPORT_MANIFEST.json`. That per-question split is the pipeline's standard export
format (produced by `export_results.py`, read by `eval_behavioral_audit_baseline.ipynb`);
do not merge the files. Baseline dirs are small (~100 rows/model total).

Both baselines were produced on a single L4 (`N_ITERATIONS=50`, judge
`gpt-4o-mini_paid`) → `results_baseline-e2b/` and `results_baseline-ministral3-8b/`.

## Files

| file | role |
|---|---|
| `modal_serve.py` | vLLM OpenAI server on Modal (model selected via deploy env vars) |
| `setup_modal_hf.py` | verify HF token + sync Modal `huggingface-token` secret |
| `modal_utils.py` | shared `.env` loader for setup script |
| `config/inference.modal.example.yaml` | provider + model aliases |
| `../behavioral_audit/run_behavioral_audit_modal.py` | Stage 1 matrix runner (personalized runs) |
| `../behavioral_audit/run_behavioral_audit_baseline_modal.py` | persona-free baseline runner for Modal subjects (Stage 1 + Stage 2) |
| `../behavioral_audit/run_behavioral_audit_baseline.py` | persona-free baseline runner for paid-API subjects (edit constants in-file) |
| `../run_direct_probing_modal.py` | direct-probing two-stage runner for Modal subjects (Stage 1 + Stage 2 + export) |
| `../behavioral_audit/export_results.py` | copy Stage-1 CSVs and Stage-2 judgments from `logs/` → `results_<tag>/` |

## Bug fixes since the initial Modal PR (#28)

The first Modal integration (PR #28, Gemma 4 E2B) shipped with `@modal.web_server`.
That run completed successfully, but the Ministral 3 8B deploy exposed bugs the Gemma
path did not trigger. The changes below are **fixes**, not arbitrary reversions.

| Change | Symptom / reason | What we do now |
|--------|----------------|----------------|
| `@modal.web_server` → ASGI proxy | Container healthy, vLLM loaded, but HTTP requests hung forever - traffic never reached vLLM on port 8000 | Run `vllm serve` on `127.0.0.1` and expose a Starlette reverse proxy via `@modal.asgi_app()` |
| Doc URL `...-serve.modal.run` → `...-serve-serve.modal.run` | Copy-paste from docs failed to connect | Modal hostnames include **both** the app name and the function name (`serve`) |
| `max_concurrency: 180` in example YAML | Would over-subscribe a single default L4 (tuned for 6× L4 Ministral fleet) | Example stays **100** (1 container × 100 inputs); Ministral README shows **180** when scaling out |
| HF volume `gemma4-hf-cache` → `pers-hf-cache` | Volume name was Gemma-specific | Generic shared cache; old volume is orphaned but harmless |
| `vllm` unpinned → `vllm>=0.20.0` + `ffmpeg` | vLLM 0.20+ needs torchcodec/ffmpeg on debian-slim | Pin minimum version, install ffmpeg in image |
| `TRITON_ATTN` for all models | Ministral 3 needs Mistral tokenizer/config/load format | Gemma keeps `TRITON_ATTN`; Ministral gets `--*-format mistral` flags and no forced Triton backend |
| `setup_modal_hf.py` | Gated repos 401 without in-container `HF_TOKEN` | Verify HF access locally, sync `huggingface-token` Modal secret |

Gemma 4 E2B on the **new** `modal_serve.py` (ASGI path) should still work with the
default deploy - no Ministral-specific env vars required.

## Serving: ASGI proxy, not `web_server`

`modal_serve.py` runs `vllm serve` on `127.0.0.1:8000` and puts a tiny **Starlette ASGI
reverse proxy** (`@modal.asgi_app()`) in front of it - it deliberately does **not** use
`@modal.web_server`. Modal's raw TCP port-proxy (`web_server`) is not routed in every
workspace (the container comes up healthy but requests hang with nothing reaching vLLM),
whereas the ASGI path is routed everywhere. The proxy passes routes, the served model name
and the auth header straight through, and holds the first requests until vLLM's `/health`
is ready - so scale-to-zero cold starts just look like a slow first response (no
`min_containers` needed).

## Modal URL pattern

For app `pers-subject-serve` and function `serve`, Modal prints:

```text
https://<workspace>--pers-subject-serve-serve.modal.run
```

The trailing `-serve` is the **function** name, not a typo. Always use the URL from
`modal deploy` output; set `MODAL_BASE_URL` to that host with `/v1` appended.

## Throughput / GPU choice

`@modal.concurrent(max_inputs=N)` gives each container vLLM continuous batching. Keep the
client's provider `max_concurrency` in `config/inference.yaml` ≈
`MODAL_SERVE_MAX_INPUTS × MODAL_SERVE_MAX_CONTAINERS`.

| Deploy | Server knobs | Client `max_concurrency` |
|--------|--------------|--------------------------|
| Default (Gemma 4 E2B, 1× L4) | `MAX_INPUTS=100`, `MAX_CONTAINERS=1` | **100** (shipped in example YAML) |
| Ministral 3 8B, scaled out | `MAX_INPUTS=30`, `MAX_CONTAINERS=6` | **180** |

An 8B model on an **L4** is KV-cache-bound (16 GB weights leave little KV room), so one L4
runs only ~30 concurrent sequences at ~350 tok/s and the audit's long (~1,800-token) responses
make a single L4 slow. Scale **out** instead of up on free-tier workspaces.

The full 7,738-response Ministral 3 8B run used 6× L4 with `max_concurrency: 180` and
finished in ~1.5 h with 0 failures. A single **A100-40GB** would be faster and cheaper total,
but Modal gates A100/H100 behind a workspace payment method - free-tier workspaces are
limited to L4/A10/T4.

Check spend any time with `modal billing report --for "this month"`.
