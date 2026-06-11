#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev-local}"
BUILDER="${BUILDER:-gpubuilder}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MAX_JOBS="${MAX_JOBS:-8}"
PATCH_ONLY="${PATCH_ONLY:-0}"
BASE_IMAGE="${BASE_IMAGE:-$IMAGE}"

if [[ ! -f "$REPO_DIR/build.sh" || ! -f "$REPO_DIR/pyproject.toml" || ! -d "$REPO_DIR/vllm" ]]; then
  echo "REPO_DIR does not look like vLLM-2080Ti-Definitive source: $REPO_DIR" >&2
  exit 1
fi

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  docker buildx create \
    --name "$BUILDER" \
    --driver-opt "image=moby/buildkit:buildx-stable-1-gpu" \
    --bootstrap
fi

docker buildx inspect "$BUILDER" --bootstrap | tee "$SCRIPT_DIR/buildx-inspect.log"

# Copy Dockerfile.local into a temporary context together with your source.
TMP_CONTEXT="$(mktemp -d)"
trap 'rm -rf "$TMP_CONTEXT"' EXIT
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude build \
  --exclude dist \
  --exclude build-logs \
  --exclude run-logs \
  --exclude .cache \
  "$REPO_DIR/" "$TMP_CONTEXT/"
if [[ "$PATCH_ONLY" == "1" ]]; then
  {
    printf '%s\n' '# syntax=docker/dockerfile:1'
    printf '%s\n' 'ARG BASE_IMAGE'
    printf '%s\n' 'FROM ${BASE_IMAGE}'
    printf '%s\n' 'WORKDIR /opt/vllm-2080ti'
    printf '%s\n' 'COPY . /opt/vllm-2080ti'
  } > "$TMP_CONTEXT/Dockerfile"
else
  cp "$SCRIPT_DIR/Dockerfile.local" "$TMP_CONTEXT/Dockerfile"
fi

if [[ "$PATCH_ONLY" == "1" ]]; then
  DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$IMAGE" \
    "$TMP_CONTEXT" 2>&1 | tee "$SCRIPT_DIR/build-local.log"
else
  docker buildx build \
    --builder "$BUILDER" \
    --allow device \
    --progress=plain \
    --load \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    -t "$IMAGE" \
    "$TMP_CONTEXT" 2>&1 | tee "$SCRIPT_DIR/build-local.log"
fi
