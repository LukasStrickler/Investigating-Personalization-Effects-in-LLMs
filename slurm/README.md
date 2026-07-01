# Slurm launchers

GPU Stage 1 (vLLM) and CPU Stage 2 (OpenRouter judge). Run from repo root on login nodes.

Guide: [docs/running-vllm-on-clusters.md](../docs/running-vllm-on-clusters.md)

## Files

| File | Role |
| --- | --- |
| `submit_vllm.sh` | Submit helper: `smoke` or `prod` |
| `bwunicluster.sbatch` / `helix.sbatch` | Job templates → `launch_vllm.sh` |
| `launch_vllm.sh` | Start vLLM, run Stage 1 script |
| `stage2.sbatch` | CPU job → Stage 2 script |
| `prefetch_model.sh` | Download weights (login node) |
| `pull_vllm_image.sh` | Pull vLLM container (Helix: needs `module load system/singularity`) |
| `vllm_constants.sh` | Image tag (`VLLM_IMAGE_TAG=v0.8.5`) |
| `sweep.sh` | Multi-model serial sweep |
| `models.example.txt` | Sweep input: `ALIAS HF_REPO SERVED [TP]` |

## Stage 1 `--export` vars

| Var | Example | Meaning |
| --- | --- | --- |
| `MODEL` | `google/gemma-3-27b-it` | HF weights |
| `SERVED` | `gemma-4-31b` | `--served-model-name` (= config `model:`) |
| `RUN_CELLS_ALIAS` | `gemma-4-31b` | YAML alias key / CSV column |
| `VENV` | `$PWD/.venv` | Host venv with `pip install -e .` |
| `SAMPLE_PER_GROUP` | `10000` | Personas per gender×race (must match Stage 2) |

Helix Stage 1: `bash slurm/submit_vllm.sh smoke helix SIF=$PWD/vllm-openai.sif`

## Stage 2 `--export` vars

| Var | Example | Meaning |
| --- | --- | --- |
| `CSV_PATH` | `logs/vllm-gemma-4-31b/matrix.csv` | Stage 1 matrix |
| `MODEL_ALIAS` | `gemma-4-31b` | Same as Stage 1 `RUN_CELLS_ALIAS` |
| `JUDGE_ALIAS` | `gemma-3-4b` | OpenRouter alias from `inference.example.yaml` |
| `SAMPLE_PER_GROUP` | `10000` | Must match Stage 1 |

## Logs

| Path | Contents |
| --- | --- |
| `logs/slurm/vllm-<jobid>.out` | Slurm stdout (GPU job) |
| `logs/slurm/<jobid>/launcher.log` | Preflight, readiness |
| `logs/slurm/<jobid>/vllm-server.log` | vLLM stdout |
| `logs/vllm-<alias>/matrix.csv` | Stage 1 results |
| `logs/vllm-<alias>/matrix.csv.<alias>.complete` | Column done marker |

Resubmit the same Stage 1 command after walltime — finished cells are skipped.
