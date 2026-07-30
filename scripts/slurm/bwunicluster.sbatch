#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --partition=dev_gpu_h100      # TEST: 30 min. PROD: gpu_h100 (72h) / gpu_a100_il (48h)
#SBATCH --gres=gpu:1                  # MUST equal $TP (count from --gres; type from --partition)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180           # warn the batch shell 180s before walltime (clean shutdown)
#SBATCH --output=logs/slurm/vllm-%j.out
#
# bwUniCluster 3.0 — vLLM server (Enroot/Pyxis container) + host experiment client.
# Submit from the REPO ROOT on a LOGIN node, after prefetching weights and creating the venv:
#
#   ws_allocate hf_cache 60
#   module load devel/miniforge && python -m venv .venv && . .venv/bin/activate && pip install -e .
#   bash scripts/slurm/prefetch_model.sh Qwen/Qwen2.5-7B-Instruct
#   cp config/inference.vllm.example.yaml config/inference.yaml
#   mkdir -p logs/slurm                 # slurmd opens --output BEFORE this body runs
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,SERVED=qwen2.5-7b-instruct,RUN_CELLS_ALIAS=qwen2.5-7b-instruct,VENV=$PWD/.venv \
#          scripts/slurm/bwunicluster.sbatch
#
# Confirm partitions/modules live: sinfo ; scontrol show partition gpu_h100 ; module avail devel/cuda
set -euo pipefail
module load devel/cuda/12.8 || echo "WARN: 'module load devel/cuda/12.8' failed — confirm with 'module avail devel/cuda'"
export RUNTIME="${RUNTIME:-container}"   # Enroot/Pyxis vllm/vllm-openai (server only)
export CLUSTER=bwunicluster
# MODEL / SERVED / RUN_CELLS_ALIAS / VENV are REQUIRED via --export= and guarded by the launcher.
exec scripts/slurm/launch_vllm.sh
