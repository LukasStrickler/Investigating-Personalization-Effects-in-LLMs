# Running experiments on vLLM (bwUniCluster & Helix)

Point the inference client at a local vLLM server instead of OpenRouter. Same
`ExperimentRunner`, same CSV schema. Stage 1 on the GPU node uses loopback HTTP;
results resume from CSV after walltime kills.

## Notebook vs cluster

| | Notebook / local | Cluster (Slurm) |
| --- | --- | --- |
| Stage 1 | [`experiments/direct_probing/run_direct_probing.py`](../experiments/direct_probing/run_direct_probing.py) or notebook | [`run_cluster_direct_probing_stage1.py`](../scripts/run_cluster_direct_probing_stage1.py) via `launch_vllm.sh` |
| Stage 2 | `run_judges` in same run (one vLLM config file) | [`run_cluster_direct_probing_stage2.py`](../scripts/run_cluster_direct_probing_stage2.py) on login/CPU (OpenRouter) |
| Config Stage 1 | `cp config/inference.vllm.example.yaml config/inference.yaml` | Same |
| Config Stage 2 | Same vLLM config (judge on local server) | `cp config/inference.example.yaml config/inference.yaml` |
| Start vLLM | `vllm serve ...` yourself | `launch_vllm.sh` |
| Scripts | Not needed | [`scripts/`](../scripts/README.md) pair |

Details: [`scripts/README.md`](../scripts/README.md), [`scripts/slurm/`](../scripts/slurm/).

## Checklist (cluster)

| Step | Command |
| --- | --- |
| Venv | Prefer `uv sync` when uv is available; else `module load devel/miniforge && python -m venv .venv && pip install -e .` |
| Logs dir | `mkdir -p logs/slurm` (before first `sbatch`) |
| Workspace | `ws_allocate hf_cache 60` (Helix: 30) |
| Weights | `bash scripts/slurm/prefetch_model.sh <MODEL>` (ungated Qwen needs no `HF_TOKEN`) |
| Config | `cp config/inference.vllm.example.yaml config/inference.yaml` |
| Helix | `module load system/singularity && bash scripts/slurm/pull_vllm_image.sh` |
| Smoke | see [Cluster quickstart](#cluster-quickstart) |
| Prod Stage 1 | `bash scripts/slurm/submit_vllm.sh prod bwunicluster` |
| Stage 2 | OpenRouter config + Stage 2 script or `sbatch scripts/slurm/stage2.sbatch` |

Slurm does not load `.env`. Export `HF_TOKEN` before prefetching gated models only.
Container tag: `scripts/slurm/vllm_constants.sh` (`VLLM_IMAGE_TAG=v0.8.5`).

### Smoke pass criteria

After `bash scripts/slurm/submit_vllm.sh smoke ...`:

1. `logs/slurm/<jobid>/launcher.log` — `server READY` and `client exited rc=0`
2. `logs/vllm-<alias>/matrix.csv` — one data row (demo + `LIMIT=1`)
3. Optional: `logs/vllm-<alias>/matrix.csv.<alias>.complete` exists

Stage 2 is a separate step after prod Stage 1 completes.

## Three names

| Role | Example | Set via |
| --- | --- | --- |
| Weights | `google/gemma-4-31b-it` | `vllm serve` first arg / `MODEL=` |
| Served name | `gemma-4-31b` | `--served-model-name` / `SERVED=` — must match config `model:` |
| Alias key | `gemma-4-31b` | YAML key under `model_aliases:` / `EXPERIMENT_MODEL` |

Stage 1 Slurm env: `RUN_CELLS_ALIAS` (= alias key). Stage 2 Slurm env: `MODEL_ALIAS` (same value).

```bash
vllm serve google/gemma-4-31b-it --served-model-name gemma-4-31b --port 8000
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer EMPTY" | grep gemma-4-31b
```

`VLLM_API_KEY=EMPTY` is a dummy bearer token. `VLLM_BASE_URL` is set by the launcher from `$PORT`.

## Local / notebook

```bash
# terminal 1
vllm serve google/gemma-4-31b-it --served-model-name gemma-4-31b --port 8000

# terminal 2
cp config/inference.vllm.example.yaml config/inference.yaml
export VLLM_API_KEY=EMPTY
python experiments/direct_probing/run_direct_probing.py
```

Stage 1 CSV: `logs/<EXPERIMENT_NAME>-stage1/<timestamp>.csv`  
Stage 2 judgments: `logs/judges/direct-probing/<EXPERIMENT_NAME>-stage2.judgments.csv`

No GPU: `python scripts/run_cluster_direct_probing_stage1.py --config config/inference.vllm.example.yaml --model-alias mock-test --prompts-source demo --limit 1`

## Cluster quickstart

From repo root on a login node:

```bash
ws_allocate hf_cache 60
# Prefer uv when available: uv sync && source .venv/bin/activate
module load devel/miniforge && python -m venv .venv && . .venv/bin/activate && pip install -e .
bash scripts/slurm/prefetch_model.sh Qwen/Qwen2.5-7B-Instruct
cp config/inference.vllm.example.yaml config/inference.yaml
mkdir -p logs/slurm

export MODEL=Qwen/Qwen2.5-7B-Instruct SERVED=qwen2.5-7b-instruct RUN_CELLS_ALIAS=qwen2.5-7b-instruct VENV=$PWD/.venv
bash scripts/slurm/submit_vllm.sh smoke bwunicluster
tail -f logs/slurm/<jobid>/launcher.log
```

Helix smoke:

```bash
module load system/singularity devel/miniforge
bash scripts/slurm/pull_vllm_image.sh
# same venv/prefetch/config/mkdir as above
bash scripts/slurm/submit_vllm.sh smoke helix SIF=$PWD/vllm-openai.sif
```

On the GPU node: `launch_vllm.sh` starts `vllm serve` in a container, then runs the Stage 1
script on the host venv. Results: `logs/vllm-<alias>/matrix.csv`.

Manual `sbatch` without `PROMPTS_SOURCE=demo,LIMIT=1` runs the full persona set (~3.7k prompts).
`submit_vllm.sh smoke` sets those automatically.

Prod: `bash scripts/slurm/submit_vllm.sh prod bwunicluster`

Gemma direct probing: prefetch `google/gemma-4-31b-it`, set `MODEL`/`SERVED`/`RUN_CELLS_ALIAS` to
`gemma-4-31b`. Match `SAMPLE_PER_GROUP` (default 10000) between Stage 1 and Stage 2.

## Stage 2 (judge)

After `logs/vllm-<alias>/matrix.csv.<alias>.complete` exists (or column fully SUCCESS). Needs internet.

```bash
# Stage 2 needs a config that defines the judge alias (paper runs used gpt-4o-mini_paid).
# inference.example.yaml and inference.modal.example.yaml both include it.
cp config/inference.modal.example.yaml config/inference.yaml
export OPENROUTER_API_KEY=sk-or-...

python scripts/run_cluster_direct_probing_stage2.py \
  --csv-path logs/vllm-gemma-4-31b/matrix.csv \
  --model-alias gemma-4-31b \
  --judge-alias gpt-4o-mini_paid \
  --sample-per-group 10000
```

Output: `logs/judges/direct-probing/matrix-stage2.judgments.csv` (default name from CSV stem).
Override with `--experiment-name my-run-stage2` → `my-run-stage2.judgments.csv`.

Slurm: `sbatch scripts/slurm/stage2.sbatch` with `CSV_PATH`, `MODEL_ALIAS` (= `RUN_CELLS_ALIAS`),
`JUDGE_ALIAS`, `VENV`, optional `SAMPLE_PER_GROUP`.

## Resume and sweep

Finished cells are written immediately. Resubmit the same Stage 1 job to continue.

Multi-model: `scripts/slurm/sweep.sh` + `scripts/slurm/models.example.txt` (`ALIAS HF_REPO SERVED [TP]`).

## Cluster reference

| | bwUniCluster 3.0 | Helix |
| --- | --- | --- |
| container | Enroot/Pyxis | Singularity (`module load system/singularity`, `SIF=`) |
| test queue | `dev_gpu_h100`, `dev_gpu_a100_il` | `devel` |
| prod queue | `gpu_h100` (72h), `gpu_a100_il` (48h) | `gpu-single` (120h) |
| GPU request | `--gres=gpu:N` | `--gres=gpu:A100:N` |
| HF cache | `ws_allocate hf_cache 60` | `ws_allocate hf_cache 30` |

Match `--gres` to `$TP`. Pass `MAXLEN=` if a model OOMs on load.

## Troubleshooting

| symptom | fix |
| --- | --- |
| `HF_HOME=... does not exist` | `ws_allocate hf_cache 60`, re-prefetch |
| will PULL the model | run `scripts/slurm/prefetch_model.sh` on login node |
| `GatedRepoError` | accept HF license, `hf auth login`, re-prefetch |
| readiness timeout | raise `READINESS_TIMEOUT=`, lower `MAXLEN=`, or more GPU |
| `port 8000 already in use` | `PORT=<other>` |
| server up but wrong model id | `SERVED` ≠ config `model:` |
| `ModuleNotFoundError: experiments` | submit from repo root, `VENV=$PWD/.venv` |
| exit 6 on resume | CSV prompt set changed — new `--csv-path` or match `SAMPLE_PER_GROUP` |
| Stage 2 no internet | run on login/CPU node, not GPU compute node |
| wrong judge classes | Stage 2 `--sample-per-group` must match Stage 1 |

`RUNTIME=venv` skips the container if you `pip install vllm` matching cluster CUDA.
