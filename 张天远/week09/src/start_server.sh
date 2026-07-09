#!/bin/bash
# 启动 vLLM OpenAI 兼容 server（多模型支持）
#
# 用法：
#   MODEL_PATH=/path/to/model bash start_server.sh
#   MODEL_PATH=/path/to/model SERVED_NAME=my-model bash start_server.sh
#
# 默认模型：Qwen2-0.5B-Instruct

set -e

# ── 配置（环境变量可覆盖）───────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/mnt/m/huggingface_cache/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH" | sed 's/-Instruct$//' | tr '[:upper:]' '[:lower:]')}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
DTYPE="${DTYPE:-float16}"

# ── 激活 venv ────────────────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    VENV="${VENV_PATH:-~/vllm_env}"
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
    fi
fi

# ── 防止 WSL 下 torch/numpy OpenMP 冲突 ─────────────────────
export KMP_DUPLICATE_LIB_OK=TRUE

echo "============================================"
echo "  启动 vLLM OpenAI Server"
echo "  模型路径: $MODEL_PATH"
echo "  对外名称: $SERVED_NAME"
echo "  端口:     $PORT"
echo "  max_len:  $MAX_MODEL_LEN"
echo "  gpu_util: $GPU_MEM_UTIL"
echo "============================================"
echo ""
echo "测试命令："
echo "  curl http://localhost:${PORT}/v1/models"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --enforce-eager \
    --host 0.0.0.0
