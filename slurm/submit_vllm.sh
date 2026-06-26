#!/usr/bin/env bash
# Submit a vLLM experiment job with sane smoke vs prod defaults.
#
# Usage (from repo root):
#   export MODEL=Qwen/Qwen2.5-7B-Instruct
#   export SERVED=qwen2.5-7b-instruct
#   export RUN_CELLS_ALIAS=qwen2.5-7b-instruct
#   export VENV=$PWD/.venv
#   bash slurm/submit_vllm.sh smoke bwunicluster
#   bash slurm/submit_vllm.sh prod  helix SIF=$PWD/vllm-openai.sif
#
# Optional env:
#   SLURM_ACCOUNT=<project>     passed to sbatch --account
#   SLURM_PARTITION=<name>      override partition
#   SLURM_TIME=<hh:mm:ss>       override walltime
#   TP=<n>                      tensor parallel (must match --gres GPU count)
#
set -euo pipefail

MODE="${1:?usage: submit_vllm.sh <smoke|prod> <bwunicluster|helix> [extra k=v exports...]}"
CLUSTER="${2:?usage: submit_vllm.sh <smoke|prod> <bwunicluster|helix> [extra k=v exports...]}"
shift 2
EXTRA=("$@")

: "${MODEL:?set MODEL=<hf-repo-id>}"
: "${SERVED:?set SERVED=<served-model-name>}"
: "${RUN_CELLS_ALIAS:?set RUN_CELLS_ALIAS=<alias KEY>}"
VENV="${VENV:-$PWD/.venv}"
TP="${TP:-1}"

case "$MODE" in
  smoke)
    case "$CLUSTER" in
      bwunicluster) DEFAULT_PART=dev_gpu_h100; DEFAULT_TIME=00:30:00 ;;
      helix)        DEFAULT_PART=devel;       DEFAULT_TIME=00:30:00 ;;
      *) echo "ERROR: unknown cluster '$CLUSTER'" >&2; exit 2 ;;
    esac
    RUN_EXPORTS="PROMPTS_SOURCE=demo,LIMIT=1"
    ;;
  prod)
    case "$CLUSTER" in
      bwunicluster) DEFAULT_PART=gpu_h100;  DEFAULT_TIME=72:00:00 ;;
      helix)        DEFAULT_PART=gpu-single; DEFAULT_TIME=120:00:00 ;;
      *) echo "ERROR: unknown cluster '$CLUSTER'" >&2; exit 2 ;;
    esac
    RUN_EXPORTS="PROMPTS_SOURCE=direct-probing"
    ;;
  *)
    echo "ERROR: mode must be smoke or prod (got '$MODE')" >&2
    exit 2
    ;;
esac

PART="${SLURM_PARTITION:-$DEFAULT_PART}"
TIME="${SLURM_TIME:-$DEFAULT_TIME}"

case "$CLUSTER" in
  bwunicluster) SBATCH_FILE="slurm/bwunicluster.sbatch"; GRES="gpu:${TP}" ;;
  helix)        SBATCH_FILE="slurm/helix.sbatch";        GRES="gpu:A100:${TP}" ;;
esac

EXPORT="ALL,${RUN_EXPORTS},MODEL=${MODEL},SERVED=${SERVED},RUN_CELLS_ALIAS=${RUN_CELLS_ALIAS},VENV=${VENV},TP=${TP}"
for kv in "${EXTRA[@]:-}"; do
  [[ -n "$kv" ]] && EXPORT+=",$kv"
done

SBATCH_ARGS=(--job-name="vllm-exp" --partition="$PART" --time="$TIME" --gres="$GRES" --export="$EXPORT")
[[ -n "${SLURM_ACCOUNT:-}" ]] && SBATCH_ARGS=(--account="$SLURM_ACCOUNT" "${SBATCH_ARGS[@]}")

echo "[submit_vllm] mode=$MODE cluster=$CLUSTER partition=$PART time=$TIME gres=$GRES"
exec sbatch "${SBATCH_ARGS[@]}" "$SBATCH_FILE"
