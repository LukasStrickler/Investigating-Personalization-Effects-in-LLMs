#!/usr/bin/env bash
# Sourced helper (do NOT execute directly): resolve the Hugging Face cache location for a
# cluster. Shared by scripts/slurm/prefetch_model.sh (login node) and scripts/slurm/launch_vllm.sh (compute
# node) so they ALWAYS agree on $HF_HOME — the launcher reads exactly what prefetch wrote.
#
# Inputs (env):
#   HF_HOME   explicit override — wins if set & non-empty (e.g. for a custom layout)
#   CLUSTER   bwunicluster | helix  (informational; resolution is workspace-based)
# Provides:
#   resolve_hf_home   prints the resolved HF_HOME to stdout (does not export)
#
# shellcheck shell=bash

resolve_hf_home() {
  # 1) Explicit override always wins.
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s\n' "$HF_HOME"
    return 0
  fi
  # 2) Preferred: a workspace named "hf_cache" (Lustre on bwUniCluster, GPFS on Helix).
  #    Allocate it once on a login node:  ws_allocate hf_cache 60   (Helix: 30)
  if command -v ws_find >/dev/null 2>&1; then
    local ws
    ws="$(ws_find hf_cache 2>/dev/null || true)"
    if [[ -n "$ws" ]]; then
      printf '%s\n' "$ws/hf"
      return 0
    fi
  fi
  # 3) Fallback: per-user $HOME cache. Works, but $HOME quota is small (200-500 GiB) and
  #    not meant for large weight sets — prefer a workspace. Warn on stderr.
  echo "[cluster_env] WARNING: no 'hf_cache' workspace found; falling back to \$HOME/hf_cache." >&2
  printf '%s\n' "${HOME}/hf_cache"
  return 0
}
