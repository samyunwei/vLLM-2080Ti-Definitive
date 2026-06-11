#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL_HOST_DIR="${1:-${MODEL_HOST_DIR:-}}"
if [[ -z "$MODEL_HOST_DIR" ]]; then
  echo "Usage: $0 /path/to/model" >&2
  echo "       MODEL_HOST_DIR=/path/to/model $0" >&2
  exit 2
fi
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev-local}"
NAME="${NAME:-qwen27b-2080ti}"
PORT="${PORT:-8000}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
PROFILE="${PROFILE:-qwen27b/safe/int4/fp16kv-256K-mtp3-text-only.env}"
MODE="${MODE:-safe}"
SERVICE_SCOPE="${SERVICE_SCOPE:-lan}"

[[ -d "$MODEL_HOST_DIR" ]] || { echo "Model directory does not exist: $MODEL_HOST_DIR" >&2; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --runtime nvidia --gpus all \
  --name "$NAME" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v "$MODEL_HOST_DIR:/model:ro" \
  -v "$REPO_DIR/profiles:/opt/vllm-2080ti/profiles:ro" \
  -e MODEL_DIR=/model \
  -e PROFILE="$PROFILE" \
  -e MODE="$MODE" \
  -e PORT="$PORT" \
  -e GPU_UTIL=0.95 \
  -e SERVICE_SCOPE="$SERVICE_SCOPE" \
  -e GPU_DEVICES="$GPU_DEVICES" \
  -e CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
  "$IMAGE" launcher --non-interactive

echo "Logs: docker logs -f $NAME"
echo "API:  http://0.0.0.0:${PORT}/v1"
