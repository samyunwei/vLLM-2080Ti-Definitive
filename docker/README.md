# Docker 构建与运行脚本说明

这些脚本用于在 Docker 中构建和运行 `vLLM 2080 Ti Definitive Edition` 的 Qwen3.6-27B 路线。这里有两套运行入口：

- `run-api-qwen27b-*.sh`：直接拼 `vllm.entrypoints.openai.api_server` 参数，不读取 `profiles/*.env`。
- `run-launcher-qwen27b.sh`：进入容器后调用仓库的 `launcher.sh`，读取 `profiles/*.env`。

模型目录不写死在脚本里。启动时必须把模型目录作为第一个参数传入，或设置 `MODEL_HOST_DIR=/path/to/model`。

## 目录结构

```text
docker/
  build-local.sh                 # 从当前本地源码构建镜像
  build-git.sh                   # 从 GitHub 分支构建镜像
  export-image.sh                # 导出镜像 tar.gz
  Dockerfile.local               # 本地源码构建用 Dockerfile
  Dockerfile.git                 # git clone 构建用 Dockerfile
  docker-entrypoint.sh           # Dockerfile.git 使用的入口脚本
  run_scripts/
    run-api-qwen27b-*.sh         # 直接 API server 路线
    run-api-qwen27b-common.sh    # run-api 公共参数拼装
    run-launcher-qwen27b.sh      # launcher/profile 路线
    run_bash.sh                  # 进入容器 bash
```

## 构建镜像

### 从本地源码构建

在仓库根目录执行：

```bash
./docker/build-local.sh
```

或在 `docker/` 目录执行：

```bash
./build-local.sh
```

默认行为：

- 源码目录默认是 `docker/` 的上级目录，也就是仓库根目录。
- 镜像名默认是 `vllm-2080ti:cu128.2-py312-dev-local`。
- Python 默认 `3.12`。
- Docker 打包路线默认 `TORCH_BACKEND=cu128`，避免 uv 自动选择 CUDA 13 PyTorch wheel。
- build context 使用 `git ls-files --exclude-standard` 生成，`.gitignore` 忽略的本地文件不会打进镜像。

常用覆盖：

```bash
IMAGE=vllm-2080ti:my-local \
PYTHON_VERSION=3.12 \
MAX_JOBS=8 \
TORCH_BACKEND=cu128 \
./docker/build-local.sh
```

构建另一个源码目录：

```bash
./docker/build-local.sh /path/to/vLLM-2080Ti-Definitive
```

等价环境变量写法：

```bash
REPO_DIR=/path/to/vLLM-2080Ti-Definitive ./docker/build-local.sh
```

### 只把源码 patch 到已有镜像

如果已有可用的 cu128 镜像，只改了 Python、脚本、文档、profiles 等不需要重新编译的内容，可以用 `PATCH_ONLY=1`：

```bash
BASE_IMAGE=vllm-2080ti:cu128.2-py312-dev-local \
IMAGE=vllm-2080ti:cu128.2-py312-dev-local-patched \
PATCH_ONLY=1 \
./docker/build-local.sh
```

`PATCH_ONLY=1` 只生成类似下面的临时 Dockerfile：

```dockerfile
FROM ${BASE_IMAGE}
WORKDIR /opt/vllm-2080ti
COPY . /opt/vllm-2080ti
```

它不会运行 `build.sh`，也不会重新安装 torch / CUDA Python 依赖。改了 CUDA/C++ 扩展、依赖版本、编译配置时不要用这个模式。

### 从 GitHub 分支构建

```bash
./docker/build-git.sh
```

默认参数：

```bash
IMAGE=vllm-2080ti:cu128.2-py312-dev
VLLM_REF=sm75-tp2-cu128-stable
PYTHON_VERSION=3.12
MAX_JOBS=8
TORCH_BACKEND=cu128
```

常用覆盖：

```bash
IMAGE=vllm-2080ti:cu128.2-py312-dev \
VLLM_REF=your-branch-or-tag \
TORCH_BACKEND=cu128 \
./docker/build-git.sh
```

### TORCH_BACKEND 说明

仓库根目录的 `build.sh` 默认 `TORCH_BACKEND=auto`，方便裸机或非 Docker 场景自动选择 PyTorch CUDA 后端。

Docker 打包脚本默认传 `TORCH_BACKEND=cu128`，因为本镜像基于 `nvidia/cuda:12.8.2-devel-ubuntu22.04`，目标路线是 cu128。这样 requirements 里仍然可以保留 `torch==2.11.0`，由 uv 的 `--torch-backend=cu128` 选择 CUDA 12.8 对应的 PyTorch wheel，避免下载 `nvidia-*-cu13`。

需要临时恢复自动选择：

```bash
TORCH_BACKEND=auto ./docker/build-local.sh
```

## 导出与导入镜像

导出：

```bash
IMAGE=vllm-2080ti:cu128.2-py312-dev-local \
OUT=vllm-2080ti-cu128.2-py312-dev-local.tar.gz \
./docker/export-image.sh
```

导入：

```bash
gunzip -c vllm-2080ti-cu128.2-py312-dev-local.tar.gz | docker load
```

`export-image.sh` 默认导出 `vllm-2080ti:cu128.2-py312-dev` 到 `vllm-2080ti-cu128.2-py312-dev.tar.gz`。

## 运行方式选择

### 直接 API server 路线

`run-api-qwen27b-*.sh` 直接运行：

```bash
python -m vllm.entrypoints.openai.api_server ...
```

这些脚本不读取 `profiles/*.env`。每个 route 的上下文、GPU util、MTP、KV dtype、batch 参数都写在脚本内，并且可用环境变量覆盖。

示例：

```bash
./docker/run_scripts/run-api-qwen27b-fp8-fp16kv-96k-mtp3-fast.sh /path/to/Qwen3.6-27B-FP8
```

先打印 docker 命令和 vLLM 参数，不启动容器：

```bash
DRY_RUN=1 ./docker/run_scripts/run-api-qwen27b-fp8-fp16kv-96k-mtp3-fast.sh /path/to/Qwen3.6-27B-FP8
```

使用环境变量传模型目录：

```bash
MODEL_HOST_DIR=/path/to/Qwen3.6-27B-FP8 \
./docker/run_scripts/run-api-qwen27b-fp8-fp16kv-96k-mtp3-fast.sh
```

### launcher/profile 路线

`run-launcher-qwen27b.sh` 会启动容器并调用：

```bash
./launcher.sh --non-interactive
```

它通过 `PROFILE=...` 读取仓库 `profiles/` 里的 `.env`。适合验证 profile 路线、复用 launcher 的交互/非交互逻辑。

示例：

```bash
PROFILE=qwen27b/safe/int4/fp16kv-256K-mtp3-text-only.env \
MODE=safe \
./docker/run_scripts/run-launcher-qwen27b.sh /path/to/Qwen3.6-27B-AWQ
```

默认 profile：

```text
qwen27b/safe/int4/fp16kv-256K-mtp3-text-only.env
```

### 进入容器 bash

```bash
./docker/run_scripts/run_bash.sh /path/to/model
```

它只挂载模型目录并进入 `/bin/bash`，便于手工检查环境。

## run-api 路线表

| 脚本 | 模型类型 | 模式 | KV | Context | GPU util | Max seqs | Max batched tokens | MTP |
|---|---|---|---|---:|---:|---:|---:|---:|
| `run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh` | INT4 | safe | fp16/default | 259200 | 0.95 | 1 | 2048 | 3 |
| `run-api-qwen27b-int4-int8kv-256k-mtp3-fast.sh` | INT4 | fast | int8_per_token_head | 262144 | 0.90 | 1 | 2048 | 3 |
| `run-api-qwen27b-fp8-int8kv-256k-mtp3-fast.sh` | FP8 | fast | int8_per_token_head | 262144 | 0.958 | 1 | 2048 | 3 |
| `run-api-qwen27b-fp8-fp16kv-128k-mtp3-safe.sh` | FP8 | safe | fp16/default | 131072 | 0.92 | 1 | 2048 | 3 |
| `run-api-qwen27b-fp8-fp16kv-96k-mtp3-fast.sh` | FP8 | fast | fp16/default | 98304 | 0.92 | 1 | 2048 | 3 |
| `run-api-qwen27b-fp8-fp16kv-90k-mtp3-safe.sh` | FP8 | safe | fp16/default | 92160 | 0.95 | 4 | 8192 | 3 |

说明：`run-api-*` 是直接参数路线。表里的值来自脚本自身，不表示会读取同名 profile。

## run-api 必填参数

### 模型目录

必须传入模型目录：

```bash
./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
```

或：

```bash
MODEL_HOST_DIR=/path/to/model \
./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh
```

容器内默认挂载到 `/model`，可用 `MODEL_CONTAINER_DIR` 改变容器内路径。

## run-api 常用 Docker 参数

| 变量 | 默认值 | 作用 |
|---|---|---|
| `IMAGE` | `vllm-2080ti:cu128.2-py312-dev-local` | 使用的 Docker 镜像 |
| `NAME` | `${ROUTE_ID}-api` | 容器名 |
| `PORT` | `8000` | 宿主机和容器内 API 端口 |
| `PUBLISH_ADDR` | `0.0.0.0` | 端口绑定地址；本机访问可设 `127.0.0.1` |
| `GPU_DEVICES` | `0,1` | 宿主机 GPU 选择，支持 `0,1` 或 `all` |
| `DOCKER_GPUS` | 空 | 原样传给 Docker `--gpus`，设置后覆盖自动 GPU 参数 |
| `TP_SIZE` | GPU 数量自动推断 | vLLM tensor parallel size |
| `CONTAINER_CUDA_VISIBLE_DEVICES` | `0..TP_SIZE-1` | 容器内 CUDA 设备编号 |
| `SHM_SIZE` | `16g` | Docker `--shm-size` |
| `CACHE_HOST_DIR` | `$HOME/.cache/vllm-2080ti` | 宿主机 torchinductor/triton cache 根目录 |
| `CONTAINER_CACHE_DIR` | `/workspace` | 容器内 cache 根目录 |
| `DRY_RUN` | `0` | 设为 `1` 时只打印命令，不启动容器 |

GPU 选择示例：

```bash
GPU_DEVICES=0,1 ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
GPU_DEVICES=1,2 TP_SIZE=2 ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
GPU_DEVICES=all TP_SIZE=2 ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
```

脚本会把 `GPU_DEVICES=0,1` 转成 Docker 安全形式：

```bash
--gpus '"device=0,1"'
```

不要手写裸的 `--gpus device=0,1`，Docker 的 CSV 解析可能报 `cannot set both Count and DeviceIDs`。

## run-api 常用 vLLM 参数

| 变量 | 默认值来源 | 对应 vLLM 参数 / 作用 |
|---|---|---|
| `SERVED_NAME` | route 脚本 | `--served-model-name` |
| `MAX_MODEL_LEN` | route 脚本 | `--max-model-len` |
| `GPU_UTIL` | route 脚本 | `--gpu-memory-utilization` |
| `MAX_NUM_SEQS` | route 脚本 | `--max-num-seqs` |
| `MAX_BATCHED_TOKENS` | route 脚本 | `--max-num-batched-tokens` |
| `MTP_K` | route 脚本 | `--speculative-config {method: mtp, num_speculative_tokens: MTP_K}` |
| `KV_CACHE_DTYPE` | route 脚本或空 | `--kv-cache-dtype`，空表示 fp16/default |
| `QUANTIZATION` | FP8 route 默认 `fp8`；INT4 自动猜 | `--quantization` |
| `ATTENTION_BACKEND` | 空 | `--attention-backend` |
| `ADDITIONAL_CONFIG_JSON` | Qwen 默认 `{"gdn_prefill_backend":"flashqla_legacy"}` | `--additional-config` |
| `HF_OVERRIDES_JSON` | 空 | `--hf-overrides` |
| `MM_LIMIT_JSON` | 空 | `--limit-mm-per-prompt` |
| `SPECULATIVE_CONFIG` | 空 | 覆盖自动 MTP speculative config |
| `COMPILATION_CONFIG_JSON` | 空 | 覆盖自动 cudagraph capture config |
| `ENFORCE_EAGER` | `0` | `1` 时加 `--enforce-eager` |
| `NO_ASYNC_SCHEDULING` | `0` | `1` 时加 `--no-async-scheduling` |
| `DISABLE_HYBRID_KV_CACHE_MANAGER` | `0` | `1` 时加 `--disable-hybrid-kv-cache-manager` |
| `DISABLE_PREFIX_CACHING` | `0` | `1` 时加 `--no-enable-prefix-caching` |
| `LANGUAGE_MODEL_ONLY` | `1` | `1` 时加 `--language-model-only` |
| `SKIP_MM_PROFILING` | `1` | `1` 时加 `--skip-mm-profiling` |
| `DISABLE_CUSTOM_ALL_REDUCE` | route 脚本 | `1` 时加 `--disable-custom-all-reduce` |
| `DISABLE_LOG_STATS` | mode 默认 | `1` 时加 `--disable-log-stats` |

INT4 quantization 自动猜测规则：

- 模型路径包含 `fp8`：`fp8`
- 包含 `gptq`：`gptq_marlin`
- 包含 `awq`：`awq_marlin`
- 包含 `quark`：`quark`
- 否则不传 `--quantization`

可强制覆盖：

```bash
QUANTIZATION=awq_marlin ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/Qwen3.6-27B-AWQ
```

## mode 默认行为

| `MODE` | `DISABLE_LOG_STATS` | `VLLM_SM75_SPEC_SYNC_MODE` | `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH` |
|---|---:|---|---:|
| `safe` | `0` | `safe` | `0` |
| `normal` | `1` | `nosync` | `0` |
| `fast` | `1` | `nosync` | `1` |

route 脚本会设置默认 mode，你也可以用环境变量覆盖。

## reasoning / tool call / chat template

| 变量 | 默认值 | 作用 |
|---|---|---|
| `REASONING_PARSER` | `qwen3` | 传给 `--reasoning-parser`；设空可关闭 |
| `REASONING_BUDGET` | 空 | 设置 `VLLM_DEFAULT_THINKING_TOKEN_BUDGET` 环境变量 |
| `DEFAULT_CHAT_TEMPLATE_KWARGS` | 空 | 传给 `--default-chat-template-kwargs` |
| `ENABLE_AUTO_TOOL_CHOICE` | `1` | `1` 时加 `--enable-auto-tool-choice` |
| `TOOL_CALL_PARSER` | `qwen3_coder` | 传给 `--tool-call-parser` |
| `TOOL_PARSER_PLUGIN` | 空 | 传给 `--tool-parser-plugin` |
| `AUTO_CHAT_TEMPLATE` | `1` | 自动寻找 Froggeric chat template |
| `CHAT_TEMPLATE_HOST_FILE` | 空 | 显式指定宿主机 chat template |
| `CHAT_TEMPLATE_CONTAINER_FILE` | `/templates/qwen-froggeric-v20.jinja` | 容器内挂载路径 |

自动 chat template 搜索顺序：

```text
<repo>/profiles/templates/qwen-froggeric-v20.jinja
$PWD/profiles/templates/qwen-froggeric-v20.jinja
<run_scripts>/profiles/templates/qwen-froggeric-v20.jinja
<run_scripts>/templates/qwen-froggeric-v20.jinja
$HOME/vLLM-2080Ti-Definitive/profiles/templates/qwen-froggeric-v20.jinja
```

关闭自动 tool call：

```bash
ENABLE_AUTO_TOOL_CHOICE=0 TOOL_CALL_PARSER='' ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
```

默认关闭 thinking：

```bash
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' \
./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
```

设置 thinking budget：

```bash
REASONING_BUDGET=2048 ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
```

不用仓库 chat template，改用模型自带模板：

```bash
AUTO_CHAT_TEMPLATE=0 ./docker/run_scripts/run-api-qwen27b-int4-fp16kv-256k-mtp3-safe.sh /path/to/model
```

## INT8 KV 相关环境变量

当 `KV_CACHE_DTYPE=int8_per_token_head` 时，脚本会额外设置：

| 变量 | 默认值 |
|---|---:|
| `VLLM_INT8KV_FA_PREFILL` | `1` |
| `VLLM_INT8KV_FA_CONTINUATION_DEQUANT` | `1` |
| `VLLM_INT8KV_FA_CASCADE_DEQUANT` | `1` |
| `VLLM_INT8KV_FA_CASCADE_TILE_TOKENS` | `65536` |

MTP 路线默认设置：

```bash
VLLM_QWOPUS_MTP_BF16_DRAFT=1
```

## launcher 脚本参数

`run-launcher-qwen27b.sh` 常用变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `IMAGE` | `vllm-2080ti:cu128.2-py312-dev-local` | Docker 镜像 |
| `NAME` | `qwen27b-2080ti` | 容器名 |
| `PORT` | `8000` | API 端口 |
| `GPU_DEVICES` | `0,1` | 传给 `GPU_DEVICES` 和 `CUDA_VISIBLE_DEVICES` |
| `PROFILE` | `qwen27b/safe/int4/fp16kv-256K-mtp3-text-only.env` | launcher profile |
| `MODE` | `safe` | launcher mode |
| `SERVICE_SCOPE` | `lan` | launcher 服务范围 |

它会把仓库根目录的 `profiles/` 只读挂载到容器内：

```text
/opt/vllm-2080ti/profiles
```

## 常见检查命令

查看镜像环境：

```bash
docker run --rm --runtime nvidia --gpus all vllm-2080ti:cu128.2-py312-dev-local check
```

查看容器日志：

```bash
docker logs -f qwen27b-fp8-fp16kv-96k-mtp3-fast-api
```

停止容器：

```bash
docker rm -f qwen27b-fp8-fp16kv-96k-mtp3-fast-api
```

API 地址：

```text
http://127.0.0.1:8000/v1
```
