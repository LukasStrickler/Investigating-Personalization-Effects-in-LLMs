#!/usr/bin/env bash
# Prefetch model weights into the cluster HF cache. Run this on a LOGIN node (it has
# internet); compute nodes are treated as OFFLINE. Idempotent — re-running re-validates an
# existing download. The launcher (scripts/slurm/launch_vllm.sh) then loads from this same cache.
#
# Usage:
#   bash scripts/slurm/prefetch_model.sh <hf-repo-id> [more-repo-ids...]
#   bash scripts/slurm/prefetch_model.sh Qwen/Qwen2.5-7B-Instruct
#
# Gated models (Llama, Gemma): accept the license on huggingface.co FIRST, then authenticate
# out-of-band — either `hf auth login` (interactive) or export HF_TOKEN (e.g. sourced from a
# gitignored .env). NEVER hard-code a token in this file.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <hf-repo-id> [more-repo-ids...]" >&2
  exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "$here/cluster_env.sh"

HF_HOME="$(resolve_hf_home)"
export HF_HOME
mkdir -p "$HF_HOME/models"
echo "[prefetch] HF_HOME=$HF_HOME"

# Prefer the modern `hf` CLI (huggingface_hub >= 0.34; `huggingface-cli` was removed in 1.0).
if command -v hf >/dev/null 2>&1; then
  download() { hf download "$1" --local-dir "$HF_HOME/models/${1##*/}"; }
elif command -v huggingface-cli >/dev/null 2>&1; then
  download() { huggingface-cli download "$1" --local-dir "$HF_HOME/models/${1##*/}"; }
else
  echo "[prefetch] ERROR: neither 'hf' nor 'huggingface-cli' is on PATH." >&2
  echo "[prefetch]        pip install -U huggingface_hub   (or load a module that provides it)" >&2
  exit 3
fi

for repo in "$@"; do
  dest="$HF_HOME/models/${repo##*/}"
  echo "[prefetch] downloading $repo -> $dest"
  download "$repo"
  echo "[prefetch] OK: $dest"
done

echo "[prefetch] done. The launcher will resolve MODEL=<repo> to \$HF_HOME/models/<basename>."
