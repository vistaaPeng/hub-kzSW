#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  vLLM 部署实验 — Seetacloud 云端一键脚本
#
#  用法:
#    bash run_all_cloud.sh                    # 跑全部 3 个模型
#    bash run_all_cloud.sh --models 0.5b,1b   # 只跑指定模型
#
#  前提：
#    1. GPU 实例已开机（4090D 24GB）
#    2. 模型已缓存在 /root/autodl-tmp/huggingface_cache/hub/
#    3. 项目代码已在 /root/vllm_deployment/
# ═══════════════════════════════════════════════════════════════════════

set -e
set -o pipefail

# ═════════════════════════════════════════════════════════════════════════
#  模型注册表
# ═════════════════════════════════════════════════════════════════════════
declare -A MODEL_MAP
MODEL_MAP[0.5b]="Qwen2-0.5B-Instruct|/root/autodl-tmp/huggingface_cache/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d|qwen2-0.5b"
MODEL_MAP[1b]="MiniCPM5-1B|/root/autodl-tmp/huggingface_cache/hub/models--openbmb--MiniCPM5-1B/snapshots/4e9de7a0778dc1c362e983e6858f0e77542cbdca|minicpm5-1b"
MODEL_MAP[7b]="Qwen2.5-7B-Instruct|/root/autodl-tmp/huggingface_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28|qwen2.5-7b"

# ═════════════════════════════════════════════════════════════════════════
#  配置
# ═════════════════════════════════════════════════════════════════════════
PROJ_DIR="/root/vllm_deployment"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/outputs}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
VENV_DIR="/root/vllm_env"
PYTHON="/root/miniconda3/bin/python"

# 默认：跑 0.5B + 1B（7B 可选）
MODEL_KEYS="${MODELS:-0.5b,1b}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODEL_KEYS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra KEYS <<< "$MODEL_KEYS"

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/run_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "════════════════════════════════════════════════"
echo "  vLLM 部署实验 — Seetacloud 云端"
echo "  时间: $(date)"
echo "  模型: $MODEL_KEYS"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════════════"

# ═════════════════════════════════════════════════════════════════════
#  Step 0: 环境准备
# ═════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 0: 环境准备"

# 关闭 mmap（某些云文件系统不兼容）
export HF_HUB_DISABLE_MMAP=1

# 创建 venv
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  ✓ venv 创建: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip

# 安装依赖
pip install -q -r "$PROJ_DIR/requirements.txt"
echo "  ✓ 依赖安装完成"

# 验证 CUDA
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
print(f'CUDA OK | {gpu} | {vram}GB VRAM')
import vllm
print(f'vLLM {vllm.__version__}')
"

# ═════════════════════════════════════════════════════════════════════
#  解析模型路径
# ═════════════════════════════════════════════════════════════════════
MODEL_PATHS=""
FIRST_MODEL_PATH=""
FIRST_MODEL_NAME=""
for key in "${KEYS[@]}"; do
    entry="${MODEL_MAP[$key]}"
    if [ -z "$entry" ]; then
        echo "  ✗ 未知模型: $key"
        continue
    fi
    IFS='|' read -r name path served <<< "$entry"
    if [ -d "$path" ]; then
        MODEL_PATHS="$MODEL_PATHS $path"
        [ -z "$FIRST_MODEL_PATH" ] && FIRST_MODEL_PATH="$path" && FIRST_MODEL_NAME="$served"
        echo "  ✓ $name ($key)"
    else
        echo "  ✗ $name 路径不存在: $path"
    fi
done

if [ -z "$FIRST_MODEL_PATH" ]; then
    echo "FATAL: 无可用的模型"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════════
#  Phase 1: 吞吐对比（不需要 server，50条基准）
# ═════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 1: 基础吞吐对比 (50 条, ~10min)"

cd "$PROJ_DIR/src"
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

python bench_throughput.py \
    --models $MODEL_PATHS \
    --n-prompts 50 \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "${LOG_DIR}/bench_${TS}.log"

echo "  ✓ 吞吐对比完成"


# ═════════════════════════════════════════════════════════════════════
#  Phase 2: 框架优势压测（200条高并发 + 延迟分布）
# ═════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 2: 4090D 框架优势压测 (200 条高并发, ~15min)"
echo "    看点: GPU 利用率 / 延迟 P50 P95 P99 / 并发上限"

STRESS_MODEL="${FIRST_MODEL_PATH}"
STRESS_NAME="${FIRST_MODEL_NAME}"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

python bench_stress.py \
    --model "$STRESS_MODEL" \
    --n-prompts 200 \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "${LOG_DIR}/stress_${TS}.log"

echo "  ✓ 压测完成"
echo "  产出:"
ls -la "$OUT_DIR"/stress_*.png 2>/dev/null || echo "  (无 PNG)"


# ═════════════════════════════════════════════════════════════════════
#  Phase 3: Demo 脚本（逐模型跑，每个需重启 server）
# ═════════════════════════════════════════════════════════════════════

for key in "${KEYS[@]}"; do
    entry="${MODEL_MAP[$key]}"
    [ -z "$entry" ] && continue
    IFS='|' read -r name path served <<< "$entry"
    [ ! -d "$path" ] && continue

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Phase 2: Demo — $name ($served)"
    echo "════════════════════════════════════════════════"

    # 启动 server
    cd "$PROJ_DIR/src"
    echo "  启动 vLLM server ..."
    MODEL_PATH="$path" SERVED_NAME="$served" \
        VENV_PATH="$VENV_DIR" \
        GPU_MEM_UTIL=0.85 MAX_MODEL_LEN=2048 \
        bash start_server.sh &
    SERVER_PID=$!

    # 等待就绪
    for i in $(seq 1 90); do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "  ✓ Server 就绪 (${i}s)"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ✗ Server 启动失败，查看日志"
            break
        fi
        sleep 2
    done
    sleep 3  # 多等一会确保完全就绪

    # 跑 demo
    export VLLM_MODEL="$served"
    MODEL_LOG="${LOG_DIR}/demo_${served}_${TS}.log"

    {
        echo "=== $(date) demo_guided_choice.py ==="
        python demo_guided_choice.py
        echo ""

        echo "=== $(date) demo_guided_regex.py ==="
        python demo_guided_regex.py
        echo ""

        echo "=== $(date) demo_guided_json.py ==="
        python demo_guided_json.py
        echo ""

        echo "=== $(date) demo_response_format.py ==="
        python demo_response_format.py
        echo ""

        echo "=== $(date) demo_function_call.py ★ ==="
        python demo_function_call.py --tool both
        echo ""
    } 2>&1 | tee "$MODEL_LOG"

    echo "  ✓ Demo 完成 → $MODEL_LOG"

    # 停 server
    kill $SERVER_PID 2>/dev/null || true
    sleep 5
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 2
    echo "  ✓ Server 已停止"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done


# ═════════════════════════════════════════════════════════════════════
#  Phase 3: 汇总
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════"
echo "  实验完成！"
echo "  产出: $OUT_DIR"
echo "  日志: $LOG_DIR"
echo "════════════════════════════════════════════════"
echo ""
ls -la "$OUT_DIR/" 2>/dev/null || echo "(空)"
echo ""
echo "主日志: $MAIN_LOG"
echo ""
echo "拉回本地："
echo "  scp -P 28197 root@connect.cqa1.seetacloud.com:\"$OUT_DIR/*\" ."
echo "  scp -P 28197 root@connect.cqa1.seetacloud.com:\"$MAIN_LOG\" ."
