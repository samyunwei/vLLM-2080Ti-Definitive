#!/usr/bin/env bash
# Official profile: profiles/qwen27b/fast/fp8/fp16kv-96K-mtp3-text-only.env
# Note: latest profiles do NOT include FP8 + FP16 KV + 256K + MTP3.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_HOST_DIR="${1:-${MODEL_HOST_DIR:-}}"
ROUTE_ID="qwen27b-fp8-fp16kv-96k-mtp3-fast"
PROFILE_REF="profiles/qwen27b/fast/fp8/fp16kv-96K-mtp3-text-only.env"
SERVED_NAME="${SERVED_NAME:-qwen27b-fp8-fp16kv-96K-mtp3-text-only-cu128}"
MODE="${MODE:-fast}"
MODEL_FAMILY="qwen"
MODEL_VARIANT="fp8"
QUANTIZATION="${QUANTIZATION:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MTP_K="${MTP_K:-3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"

source "$SCRIPT_DIR/run-api-qwen27b-common.sh"
