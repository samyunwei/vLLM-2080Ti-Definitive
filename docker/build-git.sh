#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev}"
BUILDER="${BUILDER:-gpubuilder}"
VLLM_REF="${VLLM_REF:-sm75-tp2-cu128-stable}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MAX_JOBS="${MAX_JOBS:-8}"

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  docker buildx create \
    --name "$BUILDER" \
    --driver-opt "image=moby/buildkit:buildx-stable-1-gpu" \
    --bootstrap
fi

docker buildx inspect "$BUILDER" --bootstrap | tee "$SCRIPT_DIR/buildx-inspect.log"

docker buildx build \
  --builder "$BUILDER" \
  --allow device \
  --progress=plain \
  --load \
  --build-arg VLLM_REF="$VLLM_REF" \
  --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
  --build-arg MAX_JOBS="$MAX_JOBS" \
  -f "$SCRIPT_DIR/Dockerfile.git" \
  -t "$IMAGE" \
  "$SCRIPT_DIR" 2>&1 | tee "$SCRIPT_DIR/build.log"
