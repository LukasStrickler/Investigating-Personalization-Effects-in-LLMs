# Cluster scripts

Slurm batch entry points for direct probing. Not used for notebooks or local runs.

| Script | Stage | Provider |
| --- | --- | --- |
| [`run_cluster_direct_probing_stage1.py`](run_cluster_direct_probing_stage1.py) | 1 — subject responses | vLLM |
| [`run_cluster_direct_probing_stage2.py`](run_cluster_direct_probing_stage2.py) | 2 — judge | OpenRouter |

Local: `vllm serve` + [`experiments/run_direct_probing.py`](../experiments/run_direct_probing.py).

## Outputs

| Stage | Cluster | Notebook |
| --- | --- | --- |
| 1 | `logs/vllm-<alias>/matrix.csv` | `logs/<name>-stage1/<timestamp>.csv` |
| 2 | `logs/judges/direct-probing/<csv-stem>-stage2.judgments.csv` (default) | `logs/judges/direct-probing/<EXPERIMENT_NAME>-stage2.judgments.csv` |

Cluster Stage 1 uses a fixed CSV path for resume across `sbatch` resubmits.

Guide: [docs/running-vllm-on-clusters.md](../docs/running-vllm-on-clusters.md).  
Slurm: [slurm/README.md](../slurm/README.md).
