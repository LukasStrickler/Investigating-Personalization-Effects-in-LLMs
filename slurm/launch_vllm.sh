#!/usr/bin/env bash
# Shared vLLM launcher. Invoked by slurm/<cluster>.sbatch via `exec slurm/launch_vllm.sh`.
#
# Execution model: vLLM SERVER in a container on the GPU node; experiment CLIENT on the
# HOST over loopback http://127.0.0.1:$PORT/v1. The client reads VLLM_BASE_URL (set below)
# so PORT overrides never require editing config/inference.yaml.
#
# Required env (via --export):
#   MODEL  SERVED  RUN_CELLS_ALIAS  CLUSTER  VENV
# Optional env (defaulted):
#   PORT=8000  TP=1  CONFIG=config/inference.yaml  RUNTIME=container
#   GPU_MEM_UTIL=0.85  READINESS_TIMEOUT=1200  PROMPTS_SOURCE=direct-probing  SEED=0
#   MAXLEN=<n>  COLUMNS=...  EXPERIMENT_NAME=...  CSV_PATH=...  LIMIT=<n>  SAMPLE_PER_GROUP=10000
#   SIF=<path>  (Helix)  IMAGE=...  (from slurm/vllm_constants.sh when unset)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm_constants.sh
source "$here/vllm_constants.sh"

# ---- required + defaulted env -----------------------------------------------------------
: "${MODEL:?set MODEL=<hf-repo-id or local path>}"
: "${SERVED:?set SERVED=<vllm --served-model-name; == config alias.model>}"
: "${RUN_CELLS_ALIAS:?set RUN_CELLS_ALIAS=<alias KEY / CSV column>}"
: "${CLUSTER:?set CLUSTER=bwunicluster|helix}"
: "${VENV:?set VENV=<path to the host project venv>}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
CONFIG="${CONFIG:-config/inference.yaml}"
RUNTIME="${RUNTIME:-container}"
IMAGE="${IMAGE:-$VLLM_IMAGE}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
READINESS_TIMEOUT="${READINESS_TIMEOUT:-1200}"
PROMPTS_SOURCE="${PROMPTS_SOURCE:-direct-probing}"
SEED="${SEED:-0}"
SAMPLE_PER_GROUP="${SAMPLE_PER_GROUP:-10000}"
MAXLEN="${MAXLEN:-}"
VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_DIR="logs/slurm/$JOB_ID"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/vllm-server.log"
LAUNCH_LOG="$LOG_DIR/launcher.log"

log() { echo "[launch_vllm $(date -u +%H:%M:%S)] $*" | tee -a "$LAUNCH_LOG"; }

SERVER_PID=""
CLIENT_PID=""

stop_server() {
  [[ -n "$CLIENT_PID" ]] && kill -TERM "$CLIENT_PID" 2>/dev/null || true
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  # srun/Pyxis may leave the in-container vLLM alive — sweep by port.
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
  fi
}

cleanup() { stop_server; }
on_signal() { log "received stop signal — cleaning up"; cleanup; exit 143; }
trap on_signal TERM INT USR1
trap cleanup EXIT

# ---- preflight --------------------------------------------------------------------------
[[ -x "$VENV/bin/python" ]] || { log "ERROR: VENV python not found at $VENV/bin/python"; exit 2; }
[[ -f "$CONFIG" ]] || { log "ERROR: CONFIG not found: $CONFIG"; exit 2; }

if [[ "$TP" -gt 1 ]]; then
  log "TP=$TP: request matching GPUs in sbatch (--gres) and ensure /dev/shm is mounted."
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]] && [[ "$TP" -gt "$SLURM_GPUS_ON_NODE" ]]; then
    log "ERROR: TP=$TP exceeds SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
    exit 2
  fi
fi

log "validating config alias=$RUN_CELLS_ALIAS SERVED=$SERVED"
if ! validate_out=$("$VENV/bin/python" - <<PY 2>&1)
from pathlib import Path
from inference.experiments.vllm_matrix import validate_launch_config
try:
    validate_launch_config(Path("$CONFIG"), "$RUN_CELLS_ALIAS", "$SERVED")
except ValueError as e:
    print(f"ERROR: {e}", flush=True)
    raise SystemExit(2) from e
print("config OK")
PY
then
  log "ERROR: config validation failed:"
  echo "$validate_out" | tee -a "$LAUNCH_LOG"
  exit 2
fi
log "config validation OK"

# shellcheck source=cluster_env.sh
source "$here/cluster_env.sh"
HF_HOME="$(resolve_hf_home)"
export HF_HOME
if [[ ! -d "$HF_HOME" ]]; then
  log "ERROR: HF_HOME=$HF_HOME does not exist (expired workspace?). Prefetch on a login node first."
  exit 2
fi

LOCAL_DIR="$HF_HOME/models/${MODEL##*/}"
if [[ -e "$MODEL" ]]; then
  MODEL_PATH="$MODEL"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  log "using explicit local weights: $MODEL_PATH (offline)"
elif [[ -d "$LOCAL_DIR" ]]; then
  MODEL_PATH="$LOCAL_DIR"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  log "using preloaded weights: $MODEL_PATH (offline)"
else
  MODEL_PATH="$MODEL"
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
  log "WARNING: $LOCAL_DIR missing — will PULL $MODEL from the hub (needs compute-node egress)."
fi

if command -v ss >/dev/null 2>&1; then
  if ss -ltnH 2>/dev/null | grep -qE "[:.]${PORT}([[:space:]]|\$)"; then
    log "ERROR: port $PORT already in use on $(hostname). Resubmit with PORT=<other>."
    exit 2
  fi
fi

vllm_args=(
  serve "$MODEL_PATH"
  --served-model-name "$SERVED"
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --seed "$SEED"
)
[[ -n "$MAXLEN" ]] && vllm_args+=(--max-model-len "$MAXLEN")

container_mounts="$HF_HOME:$HF_HOME"
if [[ "$TP" -gt 1 ]]; then
  container_mounts+=",/dev/shm:/dev/shm"
fi

# ---- server lifecycle -------------------------------------------------------------------
server_ready() {
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
    && curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -qF "\"$SERVED\""
}

server_log_fatal() {
  [[ -f "$SERVER_LOG" ]] || return 1
  grep -qE '(Traceback|CUDA out of memory|GatedRepoError|RuntimeError:|ERROR:)' "$SERVER_LOG"
}

start_server() {
  case "$RUNTIME" in
    container)
      case "$CLUSTER" in
        bwunicluster)
          srun --container-image="$IMAGE" \
               --container-mounts="$container_mounts" \
               --container-env=HF_HOME,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,HF_TOKEN \
               vllm "${vllm_args[@]}" >"$SERVER_LOG" 2>&1 &
          ;;
        helix)
          : "${SIF:?set SIF=<vllm-openai.sif> for CLUSTER=helix}"
          singularity exec --nv -B "$container_mounts" \
               --env "HF_HOME=$HF_HOME,HF_HUB_OFFLINE=${HF_HUB_OFFLINE},TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE},HF_TOKEN=${HF_TOKEN:-}" \
               "$SIF" vllm "${vllm_args[@]}" >"$SERVER_LOG" 2>&1 &
          ;;
        *) log "ERROR: unknown CLUSTER=$CLUSTER"; exit 2 ;;
      esac
      ;;
    venv)
      log "RUNTIME=venv: running vllm from \$VENV (no container)."
      "$VENV/bin/vllm" "${vllm_args[@]}" >"$SERVER_LOG" 2>&1 &
      ;;
    *) log "ERROR: unknown RUNTIME=$RUNTIME"; exit 2 ;;
  esac
  SERVER_PID=$!
}

wait_ready() {
  local deadline=$(( SECONDS + READINESS_TIMEOUT ))
  while (( SECONDS < deadline )); do
    if server_ready; then
      log "server READY at $VLLM_BASE_URL; /v1/models lists $SERVED"
      return 0
    fi
    if server_log_fatal; then
      log "ERROR: fatal error in server log. Last 40 lines of $SERVER_LOG:"
      tail -n 40 "$SERVER_LOG" | tee -a "$LAUNCH_LOG" || true
      return 1
    fi
    # Do NOT treat srun wrapper exit as fatal — Pyxis may return while vLLM is still loading.
    # Only fail early if nothing is listening AND the vllm process is gone.
    if ! server_ready && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      if ! pgrep -f "vllm serve.*--port ${PORT}" >/dev/null 2>&1; then
        log "ERROR: server process exited before readiness. Last 40 lines of $SERVER_LOG:"
        tail -n 40 "$SERVER_LOG" | tee -a "$LAUNCH_LOG" || true
        return 1
      fi
    fi
    sleep 5
  done
  log "ERROR: readiness timeout after ${READINESS_TIMEOUT}s. Last 40 lines of $SERVER_LOG:"
  tail -n 40 "$SERVER_LOG" | tee -a "$LAUNCH_LOG" || true
  return 1
}

# ---- run client -------------------------------------------------------------------------
log "MODEL=$MODEL SERVED=$SERVED ALIAS=$RUN_CELLS_ALIAS CLUSTER=$CLUSTER IMAGE=$IMAGE PORT=$PORT TP=$TP node=$(hostname)"
start_server
log "server wrapper pid=$SERVER_PID; logs -> $SERVER_LOG; waiting (<= ${READINESS_TIMEOUT}s)"
wait_ready || exit 1

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_BASE_URL

client_args=(
  --config "$CONFIG"
  --model-alias "$RUN_CELLS_ALIAS"
  --prompts-source "$PROMPTS_SOURCE"
  --skip-probe
)
[[ -n "${COLUMNS:-}" ]]         && client_args+=(--columns "$COLUMNS")
[[ -n "${EXPERIMENT_NAME:-}" ]] && client_args+=(--experiment-name "$EXPERIMENT_NAME")
[[ -n "${CSV_PATH:-}" ]]        && client_args+=(--csv-path "$CSV_PATH")
[[ -n "${LIMIT:-}" ]]           && client_args+=(--limit "$LIMIT")
client_args+=(--sample-per-group "$SAMPLE_PER_GROUP")

log "starting HOST client (VLLM_BASE_URL=$VLLM_BASE_URL): $VENV/bin/python scripts/run_cluster_direct_probing_stage1.py ${client_args[*]}"
"$VENV/bin/python" scripts/run_cluster_direct_probing_stage1.py "${client_args[@]}" &
CLIENT_PID=$!
CLIENT_RC=0
wait "$CLIENT_PID" || CLIENT_RC=$?
log "client exited rc=$CLIENT_RC (server will be stopped by the EXIT trap)"
exit "$CLIENT_RC"
