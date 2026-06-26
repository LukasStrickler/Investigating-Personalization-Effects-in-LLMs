#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --partition=devel             # TEST: 30 min. PROD: gpu-single (shared, 120h; A100-80G/H200)
#SBATCH --gres=gpu:A100:1             # type INSIDE --gres on Helix (A40|A100|H200); count MUST equal $TP
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180           # warn the batch shell 180s before walltime (clean shutdown)
#SBATCH --output=logs/slurm/vllm-%j.out
#
# bwForCluster Helix — vLLM server (Singularity container) + host experiment client.
# Submit from the REPO ROOT on a LOGIN node, after one-time setup:
#
#   ws_allocate hf_cache 30
#   module load devel/miniforge && python -m venv .venv && . .venv/bin/activate && pip install -e .
#   bash slurm/prefetch_model.sh Qwen/Qwen2.5-7B-Instruct
#   module load system/singularity && bash slurm/pull_vllm_image.sh
#   cp config/inference.vllm.example.yaml config/inference.yaml
#   mkdir -p logs/slurm
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,SERVED=qwen2.5-7b-instruct,RUN_CELLS_ALIAS=qwen2.5-7b-instruct,VENV=$PWD/.venv,SIF=$PWD/vllm-openai.sif \
#          slurm/helix.sbatch
#
# Confirm partitions/modules live: sinfo ; scontrol show partition gpu-single ; module avail devel/cuda
set -euo pipefail
module load devel/cuda || echo "WARN: 'module load devel/cuda' failed — confirm with 'module avail devel/cuda'"
module load system/singularity || echo "WARN: 'module load system/singularity' failed — confirm with 'module avail'"
export RUNTIME="${RUNTIME:-container}"   # Singularity vllm-openai.sif (server only); needs SIF=
export CLUSTER=helix
# MODEL / SERVED / RUN_CELLS_ALIAS / VENV / SIF are REQUIRED via --export= and guarded by the launcher.
exec slurm/launch_vllm.sh
