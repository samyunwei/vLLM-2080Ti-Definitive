#!/usr/bin/env bash
set -euo pipefail
MODEL_HOST_DIR="${1:-${MODEL_HOST_DIR:-}}"
if [[ -z "$MODEL_HOST_DIR" ]]; then
  echo "Usage: $0 /path/to/model" >&2
  echo "       MODEL_HOST_DIR=/path/to/model $0" >&2
  exit 2
fi
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev-local}"
NAME="${NAME:-qwen27b-2080ti-bash}"
PORT="${PORT:-8000}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
MTP_K="${MTP_K:-3}"

[[ -d "$MODEL_HOST_DIR" ]] || { echo "Model directory does not exist: $MODEL_HOST_DIR" >&2; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1 || true

capture=$((MTP_K + 1))

docker run -it --rm \
  --runtime nvidia --gpus all \
  --name "$NAME" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v "$MODEL_HOST_DIR:/model:ro" \
  -e CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
  -e VLLM_QWOPUS_MTP_BF16_DRAFT=1 \
  "$IMAGE" /bin/bash

