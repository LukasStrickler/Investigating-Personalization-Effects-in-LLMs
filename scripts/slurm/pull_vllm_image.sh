#!/usr/bin/env bash
# Build the Helix Singularity image for the pinned vLLM OpenAI server.
# Run on a Helix LOGIN node:
#
#   module load system/singularity
#   bash scripts/slurm/pull_vllm_image.sh              # writes vllm-openai.sif in cwd
#   bash scripts/slurm/pull_vllm_image.sh my-vllm.sif  # custom output path
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm_constants.sh
source "$here/vllm_constants.sh"

OUT="${1:-vllm-openai.sif}"
URI="docker://vllm/vllm-openai:${VLLM_IMAGE_TAG}"

module load system/singularity 2>/dev/null || true
if ! command -v singularity >/dev/null 2>&1; then
  echo "[pull_vllm_image] ERROR: singularity not on PATH (module load system/singularity?)" >&2
  exit 3
fi

echo "[pull_vllm_image] pulling $URI -> $OUT"
singularity pull --force "$OUT" "$URI"
echo "[pull_vllm_image] OK: $OUT (tag=$VLLM_IMAGE_TAG)"
