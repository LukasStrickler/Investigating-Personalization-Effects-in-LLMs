#!/usr/bin/env bash
# Multi-model sweep driver — the SINGLE chaining authority. Run on a LOGIN node, in tmux.
#
# It fills one CSV column per model, SERIALLY, into ONE shared matrix CSV. Advancement is
# decided here (CSV completeness + the client's exit code), NOT by a job self-resubmitting:
# per-job phases never requeue themselves (no #SBATCH --requeue). Guarantees:
#   * serial: one live server at a time (they would all collide on the fixed :PORT otherwise);
#   * attempt cap: a permanently-unservable model can't requeue forever (ATTEMPT_CAP);
#   * no-progress guard: if an attempt adds 0 new SUCCESS cells, stop (terminal failures);
#   * double-launch guard: refuse to start if a job named $JOB_NAME is already queued.
#
# Usage (from the repo root):
#   bash slurm/sweep.sh <cluster> <manifest> [extra k=v sbatch exports...]
#   bash slurm/sweep.sh bwunicluster slurm/models.example.txt
#   bash slurm/sweep.sh helix slurm/models.example.txt SIF=$PWD/vllm-openai.sif
#
# Manifest rows (whitespace-separated; '#' comments and blank lines ignored):
#   <ALIAS_KEY> <HF_MODEL> <SERVED> [TP]
# ALIAS_KEY must be a `vllm` alias in $CONFIG whose `model` == SERVED. The full set of
# ALIAS_KEYs becomes the matrix columns (identical every phase, so the CSV header is stable).
set -euo pipefail

CLUSTER="${1:?usage: sweep.sh <cluster> <manifest> [extra k=v exports...]}"
MANIFEST="${2:?usage: sweep.sh <cluster> <manifest> [extra k=v exports...]}"
shift 2
EXTRA_EXPORTS=("$@")

case "$CLUSTER" in
  bwunicluster) SBATCH_FILE="slurm/bwunicluster.sbatch" ;;
  helix)        SBATCH_FILE="slurm/helix.sbatch" ;;
  *) echo "ERROR: unknown cluster '$CLUSTER' (expected bwunicluster|helix)" >&2; exit 2 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 2; }

VENV="${VENV:-$PWD/.venv}"
CONFIG="${CONFIG:-config/inference.yaml}"
JOB_NAME="${JOB_NAME:-vllm-exp}"
ATTEMPT_CAP="${ATTEMPT_CAP:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm-sweep}"
CSV_PATH="${CSV_PATH:-logs/$EXPERIMENT_NAME/matrix.csv}"
POLL_SECONDS="${POLL_SECONDS:-30}"

# Parse the manifest into parallel arrays.
ALIASES=(); MODELS=(); SERVEDS=(); TPS=()
# `|| [[ -n ... ]]` so a final manifest line without a trailing newline is not dropped.
while read -r m_alias m_model m_served m_tp _rest || [[ -n "${m_alias:-}" ]]; do
  [[ -z "${m_alias:-}" || "${m_alias:0:1}" == "#" ]] && continue
  ALIASES+=("$m_alias"); MODELS+=("$m_model"); SERVEDS+=("$m_served"); TPS+=("${m_tp:-1}")
done < "$MANIFEST"
[[ ${#ALIASES[@]} -gt 0 ]] || { echo "ERROR: manifest has no model rows" >&2; exit 2; }

COLUMNS="$(IFS=,; echo "${ALIASES[*]}")"
echo "[sweep] cluster=$CLUSTER columns=$COLUMNS csv=$CSV_PATH attempt_cap=$ATTEMPT_CAP"

# Double-launch guard.
if command -v squeue >/dev/null 2>&1 && squeue -h -n "$JOB_NAME" -u "$USER" 2>/dev/null | grep -q .; then
  echo "ERROR: a job named '$JOB_NAME' is already queued/running. Refusing to double-launch." >&2
  exit 1
fi

count_success() {  # arg: alias -> prints the SUCCESS-cell count for that column
  "$VENV/bin/python" - "$CSV_PATH" "$1" <<'PY'
import sys
from pathlib import Path
csv_path, alias = sys.argv[1], sys.argv[2]
p = Path(csv_path)
if not p.exists():
    print(0)
    raise SystemExit
from inference.experiments.persistence import load_existing_matrix
_seen, completed = load_existing_matrix(p)
print(sum(1 for (_pid, a) in completed if a == alias))
PY
}

LAST_RC="?"
submit_and_wait() {  # args: alias model served tp ; sets LAST_RC to the client exit code
  local s_alias="$1" s_model="$2" s_served="$3" s_tp="$4"
  local export_str="ALL,MODEL=$s_model,SERVED=$s_served,RUN_CELLS_ALIAS=$s_alias,TP=$s_tp"
  export_str+=",VENV=$VENV,CONFIG=$CONFIG,CLUSTER=$CLUSTER,COLUMNS=$COLUMNS"
  export_str+=",EXPERIMENT_NAME=$EXPERIMENT_NAME,CSV_PATH=$CSV_PATH"
  local kv
  for kv in "${EXTRA_EXPORTS[@]:-}"; do
    [[ -n "$kv" ]] && export_str+=",$kv"
  done
  local jid
  jid="$(sbatch --parsable --job-name="$JOB_NAME" --export="$export_str" "$SBATCH_FILE")"
  echo "[sweep]   submitted job $jid (alias=$s_alias model=$s_model served=$s_served tp=$s_tp)"
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep "$POLL_SECONDS"; done
  LAST_RC="$(sacct -j "$jid" -X -n -o ExitCode 2>/dev/null | head -1 | cut -d: -f1 | tr -d ' ')"
  [[ -z "$LAST_RC" ]] && LAST_RC="?"
  echo "[sweep]   job $jid finished (client exit=$LAST_RC); log: logs/slurm/vllm-$jid.out"
}

for i in "${!ALIASES[@]}"; do
  alias_i="${ALIASES[$i]}"; model_i="${MODELS[$i]}"; served_i="${SERVEDS[$i]}"; tp_i="${TPS[$i]}"
  marker="$CSV_PATH.$alias_i.complete"
  echo "[sweep] === model $((i + 1))/${#ALIASES[@]}: alias=$alias_i ==="

  # Skip a column already finished by a previous sweep run (avoids re-booting the server).
  if [[ -f "$marker" ]]; then
    echo "[sweep]   column $alias_i already complete (marker present) — skipping."
    continue
  fi

  prev=-1
  complete=0
  for attempt in $(seq 1 "$ATTEMPT_CAP"); do
    submit_and_wait "$alias_i" "$model_i" "$served_i" "$tp_i"
    now="$(count_success "$alias_i")"
    echo "[sweep]   attempt $attempt/$ATTEMPT_CAP: rc=$LAST_RC success($alias_i)=$now"

    # AUTHORITATIVE completion signal: the client writes this marker ONLY on a clean, full run.
    # It is robust to SLURM mangling the exit code on a walltime kill (where rc may read as "0").
    if [[ -f "$marker" ]]; then
      echo "[sweep]   column $alias_i COMPLETE (marker present)."; complete=1; break
    fi
    # Fast-abort on setup errors the client reports cleanly (bad config / unreachable / mismatch).
    case "$LAST_RC" in
      2|3|4|6) echo "[sweep]   client setup error rc=$LAST_RC (non-retryable). Stopping $alias_i."; break ;;
    esac
    # Otherwise (partial=10 / outage=5 / walltime kill=signal / unknown rc): a resume retries the
    # remaining cells. Terminate only when an attempt adds NO new SUCCESS cells, or at the cap —
    # so a multi-walltime-window column keeps going as long as it makes progress.
    if [[ "$now" -le "$prev" ]]; then
      echo "[sweep]   no progress for $alias_i (prev=$prev now=$now) — remaining cells are terminal."
      break
    fi
    prev="$now"
  done
  [[ "$complete" -eq 1 ]] || echo "[sweep]   note: column $alias_i did not fully complete."
done

echo "[sweep] done. Final matrix: $CSV_PATH"
