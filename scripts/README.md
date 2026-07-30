# Scripts

Cluster launch, Modal setup/deploy, and cost calibration.
Experiment runners: [`experiments/`](../experiments/). Client library: [`src/inference/`](../src/inference/).

| Area | Contents | When to use |
| --- | --- | --- |
| Cluster Python | [`run_cluster_direct_probing_stage1.py`](run_cluster_direct_probing_stage1.py), [`run_cluster_direct_probing_stage2.py`](run_cluster_direct_probing_stage2.py) | Called by Slurm jobs / login-node Stage 2 |
| [`slurm/`](slurm/) | sbatch templates + shell launchers | bwUniCluster / Helix GPU Stage 1 + CPU Stage 2 |
| [`modal/`](modal/) | Modal setup + GPU serve deploy | New Modal account, refresh HF secret, or deploy a subject endpoint |
| Root utility | [`estimate_cost.py`](estimate_cost.py) | OpenRouter cost calibration for behavioral audit |

Cluster guide: [`docs/running-vllm-on-clusters.md`](../docs/running-vllm-on-clusters.md).

## Cluster Python (Stage 1 / Stage 2)

| Script | Stage | Provider |
| --- | --- | --- |
| [`run_cluster_direct_probing_stage1.py`](run_cluster_direct_probing_stage1.py) | 1: subject responses | vLLM |
| [`run_cluster_direct_probing_stage2.py`](run_cluster_direct_probing_stage2.py) | 2: judge | OpenRouter |

Local (non-cluster): `vllm serve` + [`experiments/direct_probing/run_direct_probing.py`](../experiments/direct_probing/run_direct_probing.py).

### Outputs

| Stage | Cluster | Notebook |
| --- | --- | --- |
| 1 | `logs/vllm-<alias>/matrix.csv` | `logs/<name>-stage1/<timestamp>.csv` |
| 2 | `logs/judges/direct-probing/<csv-stem>-stage2.judgments.csv` (default) | `logs/judges/direct-probing/<EXPERIMENT_NAME>-stage2.judgments.csv` |

Cluster Stage 1 uses a fixed CSV path so you can resume across `sbatch` resubmits.

## Slurm (`scripts/slurm/`)

GPU Stage 1 (vLLM) and CPU Stage 2 (OpenRouter judge). Run from the **repo root** on login nodes.

| File | Role |
| --- | --- |
| `submit_vllm.sh` | Submit helper: `smoke` or `prod` |
| `bwunicluster.sbatch` / `helix.sbatch` | Job templates -> `launch_vllm.sh` |
| `launch_vllm.sh` | Start vLLM, run Stage 1 script |
| `stage2.sbatch` | CPU job -> Stage 2 script |
| `prefetch_model.sh` | Download weights (login node) |
| `pull_vllm_image.sh` | Pull vLLM container (Helix: `module load system/singularity`) |
| `vllm_constants.sh` | Image tag (`VLLM_IMAGE_TAG=v0.8.5`) |
| `sweep.sh` | Multi-model serial sweep |
| `models.example.txt` | Sweep input: `ALIAS HF_REPO SERVED [TP]` |

### Stage 1 `--export` vars

| Var | Example | Meaning |
| --- | --- | --- |
| `MODEL` | `google/gemma-4-31b-it` | HF weights |
| `SERVED` | `gemma-4-31b` | `--served-model-name` (= config `model:`) |
| `RUN_CELLS_ALIAS` | `gemma-4-31b` | YAML alias key / CSV column |
| `VENV` | `$PWD/.venv` | Host venv (`uv sync` locally, or `pip install -e .` on cluster) |
| `SAMPLE_PER_GROUP` | `10000` | Personas per gender x region (must match Stage 2) |

Helix Stage 1: `bash scripts/slurm/submit_vllm.sh smoke helix SIF=$PWD/vllm-openai.sif`

### Stage 2 `--export` vars

| Var | Example | Meaning |
| --- | --- | --- |
| `CSV_PATH` | `logs/vllm-gemma-4-31b/matrix.csv` | Stage 1 matrix |
| `MODEL_ALIAS` | `gemma-4-31b` | Same as Stage 1 `RUN_CELLS_ALIAS` |
| `JUDGE_ALIAS` | `gpt-4o-mini_paid` | OpenRouter judge alias (must exist in your `inference.yaml`) |
| `SAMPLE_PER_GROUP` | `10000` | Must match Stage 1 |

### Logs

| Path | Contents |
| --- | --- |
| `logs/slurm/vllm-<jobid>.out` | Slurm stdout (GPU job) |
| `logs/slurm/<jobid>/launcher.log` | Preflight, readiness |
| `logs/slurm/<jobid>/vllm-server.log` | vLLM stdout |
| `logs/vllm-<alias>/matrix.csv` | Stage 1 results |
| `logs/vllm-<alias>/matrix.csv.<alias>.complete` | Column done marker |

Resubmit the same Stage 1 command after walltime; finished cells are skipped.

## Modal (`scripts/modal/`)

Setup and deploy only. Experiment runners stay under `experiments/`.

| Role | Script | When |
| --- | --- | --- |
| Setup | `setup_modal_hf.py` | Fresh Modal account / when HF token changes |
| Deploy | `modal_serve.py` via `modal deploy` | When you need a live subject-model endpoint |
| Usage | `experiments/.../run_*_modal.py` | Talks to the endpoint via `src/inference` |

```bash
uv sync --extra modal
uv run python scripts/modal/setup_modal_hf.py
uv run modal deploy scripts/modal/modal_serve.py
# then: export MODAL_BASE_URL=... and run experiments/.../run_*_modal.py
```

Full guide: [`scripts/modal/README.md`](modal/README.md).

## Cost estimate

```bash
.venv/bin/python scripts/estimate_cost.py [--per-question N] [--dry-run]
```

See [`docs/cost-estimate-behavioral-audit.md`](../docs/cost-estimate-behavioral-audit.md).
