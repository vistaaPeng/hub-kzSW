#!/bin/bash
# 启动 vLLM OpenAI 兼容 HTTP 服务（WSL2 内执行）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.py" 2>/dev/null || true

MODEL_PATH="/mnt/e/DeepLearning/week9/pretrain_models/Qwen2-0.5B-Instruct"
SERVED_NAME="qwen2-0.5b"
PORT=8000
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.45
DTYPE="float16"

if [ -z "$VIRTUAL_ENV" ]; then
    source ~/vllm_env/bin/activate
fi
export KMP_DUPLICATE_LIB_OK=TRUE

echo "启动 vLLM Server: $MODEL_PATH"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --enforce-eager \
    --host 0.0.0.0
