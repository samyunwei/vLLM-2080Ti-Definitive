#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev-local}"
BUILDER="${BUILDER:-gpubuilder}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MAX_JOBS="${MAX_JOBS:-8}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
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

# Copy source into a temporary context without git-ignored local artifacts.
TMP_CONTEXT="$(mktemp -d)"
trap 'rm -rf "$TMP_CONTEXT"' EXIT
if git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  (
    cd "$REPO_DIR"
    git ls-files -z --cached --modified --others --exclude-standard \
      | sort -zu \
      | rsync -a --from0 --files-from=- --ignore-missing-args ./ "$TMP_CONTEXT/"
  )
else
  rsync -a --delete \
    --filter=':- .gitignore' \
    --exclude .git \
    --exclude .venv \
    --exclude build \
    --exclude dist \
    --exclude build-logs \
    --exclude run-logs \
    --exclude .cache \
    "$REPO_DIR/" "$TMP_CONTEXT/"
fi
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
    --build-arg TORCH_BACKEND="$TORCH_BACKEND" \
    -t "$IMAGE" \
    "$TMP_CONTEXT" 2>&1 | tee "$SCRIPT_DIR/build-local.log"
fi
