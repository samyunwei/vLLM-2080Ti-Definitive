#!/usr/bin/env bash
set -euo pipefail

APP=${APP:-/opt/vllm-2080ti}
cd "$APP"

export PATH="$APP/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$APP}"
export PROFILE_DIR="${PROFILE_DIR:-$APP/profiles}"
export LOG_DIR="${LOG_DIR:-$APP/run-logs}"
export MODEL_DIR="${MODEL_DIR:-/model}"
export PROFILE="${PROFILE:-qwen27b/safe/int4/fp16kv-256K-mtp3-text-only.env}"
export MODE="${MODE:-safe}"
export PORT="${PORT:-8000}"
export SERVICE_SCOPE="${SERVICE_SCOPE:-lan}"
export GPU_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_DEVICES}"
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

cmd="${1:-launcher}"
shift || true

case "$cmd" in
  launcher)
    args=("$@")
    [[ ${#args[@]} -gt 0 ]] || args=(--non-interactive)

    # launcher.sh starts the API server with nohup and writes PID/log state.
    ./launcher.sh "${args[@]}"

    state="${STATE_FILE:-$LOG_DIR/start-manager.state}"
    for _ in $(seq 1 60); do
      if [[ -f "$state" ]]; then
        # shellcheck disable=SC1090
        source "$state" || true
        [[ -n "${LAST_PID_FILE:-}" && -f "${LAST_PID_FILE:-}" ]] && break
      fi
      sleep 1
    done

    if [[ -z "${LAST_PID_FILE:-}" || ! -f "${LAST_PID_FILE:-}" ]]; then
      echo "ERROR: launcher returned but no PID file was found under $LOG_DIR" >&2
      exit 1
    fi

    server_pid="$(cat "$LAST_PID_FILE")"
    if [[ -z "$server_pid" || ! -d "/proc/$server_pid" ]]; then
      echo "ERROR: server process is not running; pid_file=$LAST_PID_FILE" >&2
      [[ -n "${LAST_LOG_FILE:-}" && -f "$LAST_LOG_FILE" ]] && tail -n 120 "$LAST_LOG_FILE" >&2 || true
      exit 1
    fi

    echo "Container supervisor: server_pid=$server_pid"
    [[ -n "${LAST_LOG_FILE:-}" ]] && echo "Container supervisor: log_file=$LAST_LOG_FILE"

    tail_pid=""
    if [[ -n "${LAST_LOG_FILE:-}" && -f "$LAST_LOG_FILE" ]]; then
      tail -n "${TAIL_LINES:-80}" -F "$LAST_LOG_FILE" &
      tail_pid=$!
    fi

    terminate() {
      echo "Container supervisor: stopping server pid=$server_pid"
      kill -TERM "$server_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      kill -KILL "$server_pid" 2>/dev/null || true
      [[ -n "$tail_pid" ]] && kill "$tail_pid" 2>/dev/null || true
    }
    trap terminate TERM INT

    while kill -0 "$server_pid" 2>/dev/null; do
      sleep 2
    done
    [[ -n "$tail_pid" ]] && kill "$tail_pid" 2>/dev/null || true
    ;;

  api)
    # Direct vLLM OpenAI-compatible API server mode. Pass all your own vLLM args after `api`.
    exec "$APP/.venv/bin/python" -m vllm.entrypoints.openai.api_server "$@"
    ;;

  check)
    exec bash -lc 'nvidia-smi && /opt/vllm-2080ti/.venv/bin/python - <<"PY"
import sys
import torch
import vllm
print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("vllm", getattr(vllm, "__version__", "unknown"))
PY'
    ;;

  print-config)
    exec ./launcher.sh --print-config
    ;;

  rebuild)
    exec bash -lc 'ASSUME_YES=1 PYTHON_VERSION="${PYTHON_VERSION:-3.12}" MAX_JOBS="${MAX_JOBS:-8}" ./build.sh'
    ;;

  bash|/bin/bash|sh|/bin/sh|python|python3|nvidia-smi)
    exec "$cmd" "$@"
    ;;

  *)
    exec "$cmd" "$@"
    ;;
esac
