#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev}"
OUT="${OUT:-vllm-2080ti-cu128.2-py312-dev.tar.gz}"
docker save "$IMAGE" | gzip -1 > "$OUT"
echo "Wrote: $OUT"
echo "Load: gunzip -c $OUT | docker load"
