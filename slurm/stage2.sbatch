#!/bin/bash
#SBATCH --job-name=direct-probing-stage2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage2-%j.out
#
# Direct probing Stage 2 (judge) — CPU job with internet egress.
# Uses OpenRouter from config/inference.yaml (NOT the vLLM config).
# Script: scripts/run_cluster_direct_probing_stage2.py
#
# Submit from repo root after cluster Stage 1 completes:
#
#   cp config/inference.example.yaml config/inference.yaml
#   mkdir -p logs/slurm logs/judges/direct-probing
#   sbatch --partition=<cpu-partition-with-egress> \
#     --export=ALL,CSV_PATH=logs/vllm-gemma-4-31b/matrix.csv,MODEL_ALIAS=gemma-4-31b,JUDGE_ALIAS=gemma-3-4b,VENV=$PWD/.venv \
#     slurm/stage2.sbatch
#
# Optional exports: CONFIG=... EXPERIMENT_NAME=... WORKERS=5 SAMPLE_PER_GROUP=10000
# MODEL_ALIAS is the Stage-1 column (same value as RUN_CELLS_ALIAS from launch_vllm.sh).
set -euo pipefail

: "${CSV_PATH:?set CSV_PATH=<stage-1 matrix.csv>}"
: "${MODEL_ALIAS:?set MODEL_ALIAS=<stage-1 column alias (= RUN_CELLS_ALIAS)>}"
: "${JUDGE_ALIAS:?set JUDGE_ALIAS=<judge model alias from config>}"
: "${VENV:?set VENV=$PWD/.venv}"

CONFIG="${CONFIG:-config/inference.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
WORKERS="${WORKERS:-5}"
SAMPLE_PER_GROUP="${SAMPLE_PER_GROUP:-10000}"

module load devel/miniforge 2>/dev/null || true

args=(
  --config "$CONFIG"
  --csv-path "$CSV_PATH"
  --model-alias "$MODEL_ALIAS"
  --judge-alias "$JUDGE_ALIAS"
  --workers "$WORKERS"
  --sample-per-group "$SAMPLE_PER_GROUP"
)
[[ -n "$EXPERIMENT_NAME" ]] && args+=(--experiment-name "$EXPERIMENT_NAME")

exec "$VENV/bin/python" scripts/run_cluster_direct_probing_stage2.py "${args[@]}"
