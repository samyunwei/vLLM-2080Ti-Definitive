#!/usr/bin/env bash
# Common Docker runner for Qwen3.6-27B vLLM-2080Ti profiles.
# Individual route scripts set the profile variables, then source this file.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

: "${ROUTE_ID:?ROUTE_ID is required}"
: "${PROFILE_REF:?PROFILE_REF is required}"
: "${SERVED_NAME:?SERVED_NAME is required}"
: "${MODE:?MODE is required}"
: "${MODEL_FAMILY:=qwen}"
: "${MODEL_VARIANT:=int4}"
if [[ -z "${MODEL_HOST_DIR:-}" ]]; then
  echo "Usage: $0 /path/to/model" >&2
  echo "       MODEL_HOST_DIR=/path/to/model $0" >&2
  exit 2
fi
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required}"
: "${GPU_UTIL:?GPU_UTIL is required}"
: "${MAX_BATCHED_TOKENS:?MAX_BATCHED_TOKENS is required}"
: "${MAX_NUM_SEQS:?MAX_NUM_SEQS is required}"
: "${MTP_K:?MTP_K is required}"

# Your pasted IMAGE value had two ':' separators, which Docker normally rejects.
# Override IMAGE=... if your local tag/name is different.
IMAGE="${IMAGE:-vllm-2080ti:cu128.2-py312-dev-local}"
NAME="${NAME:-${ROUTE_ID}-api}"
PORT="${PORT:-8000}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
# Optional raw Docker --gpus override. Leave empty for automatic safe quoting.
DOCKER_GPUS="${DOCKER_GPUS:-}"
TP_SIZE="${TP_SIZE:-}"
MODEL_CONTAINER_DIR="${MODEL_CONTAINER_DIR:-/model}"
PUBLISH_ADDR="${PUBLISH_ADDR:-0.0.0.0}"
API_HOST_IN_CONTAINER="${API_HOST_IN_CONTAINER:-0.0.0.0}"
SHM_SIZE="${SHM_SIZE:-16g}"
CACHE_HOST_DIR="${CACHE_HOST_DIR:-${HOME}/.cache/vllm-2080ti}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/workspace}"
# CUDA_VISIBLE_DEVICES inside the container is re-indexed to 0..TP_SIZE-1 by default.
CONTAINER_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
DRY_RUN="${DRY_RUN:-0}"

# Global service parameters mirrored from launcher.sh. Defaults can be overridden.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
REASONING_BUDGET="${REASONING_BUDGET:-}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
TOOL_PARSER_PLUGIN="${TOOL_PARSER_PLUGIN:-}"
CHAT_TEMPLATE_HOST_FILE="${CHAT_TEMPLATE_HOST_FILE:-}"
AUTO_CHAT_TEMPLATE="${AUTO_CHAT_TEMPLATE:-1}"
CHAT_TEMPLATE_CONTAINER_FILE="${CHAT_TEMPLATE_CONTAINER_FILE:-/templates/qwen-froggeric-v20.jinja}"

# vLLM extra knobs, normally empty for these text-only routes.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ADDITIONAL_CONFIG_JSON="${ADDITIONAL_CONFIG_JSON:-}"
HF_OVERRIDES_JSON="${HF_OVERRIDES_JSON:-}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
COMPILATION_CONFIG_JSON="${COMPILATION_CONFIG_JSON:-}"
MM_LIMIT_JSON="${MM_LIMIT_JSON:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
NO_ASYNC_SCHEDULING="${NO_ASYNC_SCHEDULING:-0}"
DISABLE_HYBRID_KV_CACHE_MANAGER="${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}"
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-0}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
QUANTIZATION="${QUANTIZATION:-}"

lowercase() {
  tr '[:upper:]' '[:lower:]' <<<"${1:-}"
}

normalize_gpu_devices() {
  local devices="${1// /}"
  devices="${devices%,}"
  echo "$devices"
}

seq_cuda_devices() {
  local n="$1" i out=""
  for ((i = 0; i < n; i++)); do
    if [[ -z "$out" ]]; then out="$i"; else out+=",$i"; fi
  done
  echo "$out"
}

gpu_count() {
  local devices part count=0
  devices="$(normalize_gpu_devices "$1")"
  if [[ "$devices" == "all" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L | wc -l | tr -d ' '
    else
      # Cannot infer inside every environment; caller can override TP_SIZE explicitly.
      echo 1
    fi
    return 0
  fi
  IFS=',' read -r -a _parts <<< "$devices"
  for part in "${_parts[@]}"; do
    [[ -n "$part" ]] && count=$((count + 1))
  done
  echo "$count"
}

append_docker_gpu_args() {
  local devices
  devices="$(normalize_gpu_devices "$GPU_DEVICES")"

  if [[ -n "$DOCKER_GPUS" ]]; then
    DOCKER_ARGS+=(--gpus "$DOCKER_GPUS")
    return 0
  fi

  if [[ "$devices" == "all" ]]; then
    DOCKER_ARGS+=(--gpus all)
    return 0
  fi

  if [[ "$devices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    # Docker parses --gpus with a CSV parser. A raw device=0,1 is split into
    # DeviceIDs=0 plus Count=1, causing: cannot set both Count and DeviceIDs.
    # The embedded quotes are intentional and match Docker's documented form:
    #   --gpus '"device=0,1"'
    DOCKER_ARGS+=(--gpus "\"device=${devices}\"")
    return 0
  fi

  echo "ERROR: GPU_DEVICES must be a comma-separated list like 0,1 or 'all'; got: $GPU_DEVICES" >&2
  exit 1
}

guess_quantization() {
  local dir quant_lc
  dir="$(lowercase "$1")"
  quant_lc="$(lowercase "${2:-}")"
  if [[ -n "$quant_lc" ]]; then
    echo "$2"
  elif [[ "$dir" == *fp8* ]]; then
    echo fp8
  elif [[ "$dir" == *gptq* ]]; then
    echo gptq_marlin
  elif [[ "$dir" == *awq* ]]; then
    echo awq_marlin
  elif [[ "$dir" == *quark* ]]; then
    echo quark
  else
    echo ""
  fi
}

resolve_auto_chat_template() {
  [[ -n "$CHAT_TEMPLATE_HOST_FILE" ]] && return 0
  [[ "$AUTO_CHAT_TEMPLATE" == "1" ]] || return 0
  local candidate
  for candidate in \
    "$REPO_DIR/profiles/templates/qwen-froggeric-v20.jinja" \
    "$PWD/profiles/templates/qwen-froggeric-v20.jinja" \
    "$SCRIPT_DIR/profiles/templates/qwen-froggeric-v20.jinja" \
    "$SCRIPT_DIR/templates/qwen-froggeric-v20.jinja" \
    "$HOME/vLLM-2080Ti-Definitive/profiles/templates/qwen-froggeric-v20.jinja"; do
    if [[ -f "$candidate" ]]; then
      CHAT_TEMPLATE_HOST_FILE="$candidate"
      return 0
    fi
  done
}

apply_mode_defaults() {
  case "$MODE" in
    safe)
      DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-0}"
      VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
      VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH="${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
      ;;
    fast)
      DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-1}"
      VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-nosync}"
      VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH="${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-1}"
      ;;
    normal)
      DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-1}"
      VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-nosync}"
      VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH="${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
      ;;
    *)
      echo "ERROR: MODE must be safe, normal, or fast; got: $MODE" >&2
      exit 1
      ;;
  esac
}

validate_inputs() {
  [[ -d "$MODEL_HOST_DIR" ]] || { echo "Model directory does not exist: $MODEL_HOST_DIR" >&2; exit 1; }
  GPU_DEVICES="$(normalize_gpu_devices "$GPU_DEVICES")"
  if [[ -z "$TP_SIZE" ]]; then TP_SIZE="$(gpu_count "$GPU_DEVICES")"; fi
  [[ "$TP_SIZE" =~ ^[0-9]+$ && "$TP_SIZE" -gt 0 ]] || { echo "Invalid TP_SIZE: $TP_SIZE" >&2; exit 1; }
  if [[ -z "$CONTAINER_CUDA_VISIBLE_DEVICES" ]]; then
    CONTAINER_CUDA_VISIBLE_DEVICES="$(seq_cuda_devices "$TP_SIZE")"
  fi
  mkdir -p "$CACHE_HOST_DIR/torchinductor" "$CACHE_HOST_DIR/triton"
  resolve_auto_chat_template
  if [[ -n "$CHAT_TEMPLATE_HOST_FILE" && ! -f "$CHAT_TEMPLATE_HOST_FILE" ]]; then
    echo "ERROR: CHAT_TEMPLATE_HOST_FILE does not exist: $CHAT_TEMPLATE_HOST_FILE" >&2
    exit 1
  fi
}

build_vllm_args() {
  local capture
  VLLM_ARGS=(
    --host "$API_HOST_IN_CONTAINER"
    --port "$PORT"
    --model "$MODEL_CONTAINER_DIR"
    --served-model-name "$SERVED_NAME"
    --dtype half
    --tensor-parallel-size "$TP_SIZE"
    --generation-config vllm
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  )

  if [[ "$MODEL_VARIANT" == "fp8" ]]; then
    QUANTIZATION="${QUANTIZATION:-fp8}"
  else
    QUANTIZATION="$(guess_quantization "$MODEL_HOST_DIR" "$QUANTIZATION")"
  fi

  [[ -n "$QUANTIZATION" ]] && VLLM_ARGS+=(--quantization "$QUANTIZATION")
  [[ -n "$KV_CACHE_DTYPE" ]] && VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  [[ "$ENFORCE_EAGER" == "1" ]] && VLLM_ARGS+=(--enforce-eager)
  [[ "$NO_ASYNC_SCHEDULING" == "1" ]] && VLLM_ARGS+=(--no-async-scheduling)
  [[ "$DISABLE_HYBRID_KV_CACHE_MANAGER" == "1" ]] && VLLM_ARGS+=(--disable-hybrid-kv-cache-manager)
  [[ "$DISABLE_PREFIX_CACHING" == "1" ]] && VLLM_ARGS+=(--no-enable-prefix-caching)
  [[ "$LANGUAGE_MODEL_ONLY" == "1" ]] && VLLM_ARGS+=(--language-model-only)
  [[ "$SKIP_MM_PROFILING" == "1" ]] && VLLM_ARGS+=(--skip-mm-profiling)
  [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]] && VLLM_ARGS+=(--disable-custom-all-reduce)
  [[ "$DISABLE_LOG_STATS" == "1" ]] && VLLM_ARGS+=(--disable-log-stats)
  [[ -n "$ATTENTION_BACKEND" ]] && VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")

  [[ -n "$REASONING_PARSER" ]] && VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
  [[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]] && VLLM_ARGS+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
  [[ -n "$TOOL_PARSER_PLUGIN" ]] && VLLM_ARGS+=(--tool-parser-plugin "$TOOL_PARSER_PLUGIN")
  [[ -n "$TOOL_CALL_PARSER" ]] && VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
  [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]] && VLLM_ARGS+=(--enable-auto-tool-choice)

  if [[ -n "$ADDITIONAL_CONFIG_JSON" ]]; then
    VLLM_ARGS+=(--additional-config "$ADDITIONAL_CONFIG_JSON")
  elif [[ "$MODEL_FAMILY" == qwen* ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  fi

  [[ -n "$HF_OVERRIDES_JSON" ]] && VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES_JSON")
  [[ -n "$MM_LIMIT_JSON" ]] && VLLM_ARGS+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")
  [[ -n "$CHAT_TEMPLATE_HOST_FILE" ]] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE_CONTAINER_FILE")

  capture=$((MTP_K + 1))
  if [[ -n "$SPECULATIVE_CONFIG" ]]; then
    VLLM_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  elif (( MTP_K > 0 )); then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
  fi

  if [[ -n "$COMPILATION_CONFIG_JSON" ]]; then
    VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG_JSON")
  elif [[ -n "$SPECULATIVE_CONFIG" || "$MTP_K" -gt 0 ]]; then
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_capture_sizes\":[${capture}],\"max_cudagraph_capture_size\":${capture}}")
  else
    VLLM_ARGS+=(--compilation-config '{"cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}')
  fi
}

build_docker_args() {
  DOCKER_ARGS=(
    run -d
    --name "$NAME"
  )
  append_docker_gpu_args
  DOCKER_ARGS+=(
    --ipc=host
    --shm-size "$SHM_SIZE"
    --ulimit memlock=-1:-1
    --ulimit stack=67108864
    -p "${PUBLISH_ADDR}:${PORT}:${PORT}"
    -v "$MODEL_HOST_DIR:$MODEL_CONTAINER_DIR:ro"
    -v "$CACHE_HOST_DIR/torchinductor:${CONTAINER_CACHE_DIR}/torchinductor-cache"
    -v "$CACHE_HOST_DIR/triton:${CONTAINER_CACHE_DIR}/triton-cache"
    -e CUDA_VISIBLE_DEVICES="$CONTAINER_CUDA_VISIBLE_DEVICES"
    -e NVIDIA_VISIBLE_DEVICES="$GPU_DEVICES"
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID
    -e CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
    -e CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.8}"
    -e CUDACXX="${CUDACXX:-/usr/local/cuda-12.8/bin/nvcc}"
    -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
    -e FLASHINFER_ENABLE_AOT="${FLASHINFER_ENABLE_AOT:-1}"
    -e PYTHONUNBUFFERED=1
    -e TORCHINDUCTOR_CACHE_DIR="${CONTAINER_CACHE_DIR}/torchinductor-cache"
    -e TRITON_CACHE_DIR="${CONTAINER_CACHE_DIR}/triton-cache"
    -e STABLE_ROOT="${STABLE_ROOT:-/workspace}"
    -e PYTHONPATH="${CONTAINER_PYTHONPATH:-/workspace:/opt/FlashQLA-SM70-SM75}"
    -e VLLM_SM75_SPEC_SYNC_MODE="$VLLM_SM75_SPEC_SYNC_MODE"
    -e VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH="$VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH"
  )

  [[ -n "$PYTHON_BIN" ]] && DOCKER_ARGS+=(-e PYTHON_BIN="$PYTHON_BIN")
  [[ -n "$REASONING_BUDGET" ]] && DOCKER_ARGS+=(-e VLLM_DEFAULT_THINKING_TOKEN_BUDGET="$REASONING_BUDGET")
  [[ "$MTP_K" -gt 0 ]] && DOCKER_ARGS+=(-e VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}")

  if [[ "$KV_CACHE_DTYPE" == "int8_per_token_head" ]]; then
    DOCKER_ARGS+=(
      -e VLLM_INT8KV_FA_PREFILL="${VLLM_INT8KV_FA_PREFILL:-1}"
      -e VLLM_INT8KV_FA_CONTINUATION_DEQUANT="${VLLM_INT8KV_FA_CONTINUATION_DEQUANT:-1}"
      -e VLLM_INT8KV_FA_CASCADE_DEQUANT="${VLLM_INT8KV_FA_CASCADE_DEQUANT:-1}"
      -e VLLM_INT8KV_FA_CASCADE_TILE_TOKENS="${VLLM_INT8KV_FA_CASCADE_TILE_TOKENS:-65536}"
    )
  fi

  if [[ -n "$CHAT_TEMPLATE_HOST_FILE" ]]; then
    DOCKER_ARGS+=(-v "$CHAT_TEMPLATE_HOST_FILE:$CHAT_TEMPLATE_CONTAINER_FILE:ro")
  fi

  DOCKER_ARGS+=(--entrypoint /bin/bash "$IMAGE")
}

print_summary() {
  cat <<SUMMARY_EOF
Route:        $ROUTE_ID
Profile:      $PROFILE_REF
Mode:         $MODE
Model path:   $MODEL_HOST_DIR -> $MODEL_CONTAINER_DIR
Image:        $IMAGE
Container:    $NAME
Port:         ${PUBLISH_ADDR}:${PORT}:${PORT}
GPUs / TP:    $GPU_DEVICES / $TP_SIZE
Container CUDA_VISIBLE_DEVICES: $CONTAINER_CUDA_VISIBLE_DEVICES
Served name:  $SERVED_NAME
Max len:      $MAX_MODEL_LEN
KV dtype:     ${KV_CACHE_DTYPE:-fp16/default}
Quantization: ${QUANTIZATION:-auto/not passed}
MTP_K:        $MTP_K
Reasoning:    ${REASONING_PARSER:-off}${REASONING_BUDGET:+, budget=$REASONING_BUDGET}
Tool call:    auto=${ENABLE_AUTO_TOOL_CHOICE}, parser=${TOOL_CALL_PARSER:-unset}
Chat template:${CHAT_TEMPLATE_HOST_FILE:- model default / not mounted}
SUMMARY_EOF
}

main() {
  apply_mode_defaults
  validate_inputs
  build_vllm_args
  build_docker_args

  print_summary
  echo

  local container_script
  container_script='set -euo pipefail
if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
  PYBIN="$PYTHON_BIN"
elif [[ -x /workspace/.venv/bin/python ]]; then
  PYBIN=/workspace/.venv/bin/python
else
  PYBIN=python
fi
exec "$PYBIN" -m vllm.entrypoints.openai.api_server "$@"'

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN docker command:"
    printf 'docker '
    printf '%q ' "${DOCKER_ARGS[@]}" -lc "$container_script" -- "${VLLM_ARGS[@]}"
    echo
    echo
    echo "vLLM args:"
    printf '  %q' "${VLLM_ARGS[@]}"
    echo
    exit 0
  fi

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker "${DOCKER_ARGS[@]}" -lc "$container_script" -- "${VLLM_ARGS[@]}"

  echo
  echo "Started. Logs: docker logs -f $NAME"
  echo "API: http://127.0.0.1:${PORT}/v1"
}

main "$@"
