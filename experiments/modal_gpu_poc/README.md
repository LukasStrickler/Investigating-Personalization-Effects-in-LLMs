# Modal GPU integration — self-host subject models on rented GPUs

Run the behavioral audit's **Stage 1 subject model on [Modal](https://modal.com)**
(serverless GPU) when the university Slurm cluster is unavailable. Modal serves an
OpenAI-compatible vLLM endpoint that plugs into the repo's `modal` inference provider.

Stage 1 writes matrix CSVs to the gitignored `logs/` working area. Use
`export_results.py` to copy finished Stage-1 artifacts into a committable
`results_<tag>/` directory (same layout as `results_full001/`, …).

Stage 2 (LLM judge) is **not** part of this path — run it separately with
`run_behavioral_audit.py` (`STAGE2_ONLY=True`) once Stage-1 CSVs exist in
`logs/behavioral-audit-<run-tag>-q{1,2}-stage1/` (export does not move them;
it copies a snapshot into `results_<tag>/` for git). Set `RUN_TAG` and
`EXPERIMENT_MODELS = ["gemma-4-e2b_modal"]` in that script before running.

## Workflow

```bash
cd <repo-root>
set -a; source .env; set +a    # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET for deploy

# 1. Deploy the subject-model server (prints a *.modal.run URL)
modal deploy experiments/modal_gpu_poc/modal_serve.py
export MODAL_BASE_URL="https://<workspace>--pers-subject-serve-serve.modal.run/v1"
export MODAL_API_KEY="EMPTY"   # or the server's bearer token if MODAL_SERVE_AUTH=1

# 2. Point the inference client at Modal
cp config/inference.modal.example.yaml config/inference.yaml

# 3. Stage 1 — generate subject responses → logs/…-stage1/*.csv
#    Add --limit 4 for a smoke test.
python experiments/behavioral_audit/run_behavioral_audit_modal.py \
    --run-tag full001-e2b --subject-alias gemma-4-e2b_modal

# 4. Export Stage-1 CSVs from logs/ into a committable results dir
python experiments/behavioral_audit/export_results.py \
    --run-tag full001-e2b --subject-alias gemma-4-e2b_modal
git add experiments/behavioral_audit/results_full001-e2b && git commit -m "Add full001-e2b stage1"

# 5. Tear down (or let it scale to zero after MODAL_SERVE_SCALEDOWN idle seconds)
modal app stop pers-subject-serve
```

| file | role |
|---|---|
| `modal_serve.py` | deploy a vLLM OpenAI server for a subject model on Modal |
| `config/inference.modal.example.yaml` | `modal` subject provider config |
| `../behavioral_audit/run_behavioral_audit_modal.py` | Stage 1 — subject responses → matrix CSVs in `logs/` |
| `../behavioral_audit/export_results.py` | copy finished Stage-1 from `logs/` → `results_<tag>/` |

### Bigger subject model

Deploy with a larger GPU and matching alias:

```bash
MODAL_SERVE_MODEL_ID=google/gemma-4-31b-it MODAL_SERVE_NAME=gemma-4-31b \
  MODAL_SERVE_GPU=A100-80GB modal deploy experiments/modal_gpu_poc/modal_serve.py
python experiments/behavioral_audit/run_behavioral_audit_modal.py \
    --run-tag my-run --subject-alias gemma-4-31b_modal
```

### Throughput

The server uses `@modal.concurrent(max_inputs=100)` so one GPU batches many
in-flight requests (vLLM continuous batching). Keep the client's provider
`max_concurrency` in the YAML config ≈ `MODAL_SERVE_MAX_INPUTS` (default 100).
Without this, a `@modal.web_server` serves one request at a time and the run is
much slower.

### Persona set

With default `--sample-per-group 10000` and `--seed 42`, the runner uses all
**3869** personas from `personas.jsonl` — the same `prompt_id` set as
`results_full001/` and `results_full002/`.
