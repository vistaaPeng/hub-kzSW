#!/bin/bash
# 在 WSL2 Ubuntu 内执行：安装 vLLM 并跑完整 benchmark（无需 sudo）
set -e

PROJECT="/mnt/e/DeepLearning/week9/week9 大模型应用补充知识/work9"
MODEL="/mnt/e/DeepLearning/week9/pretrain_models/Qwen2-0.5B-Instruct"

need_sudo=0
for cmd in gcc g++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        need_sudo=1
        break
    fi
done

if [ "$need_sudo" = "1" ]; then
    echo "=== 需要安装 build-essential（会提示输入 Ubuntu 密码）==="
    sudo apt update
    sudo apt install -y python3-pip python3-venv build-essential
else
    echo "=== 跳过 apt（编译工具已就绪）==="
fi

echo "=== 创建虚拟环境 ==="
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate
python -m pip install -U pip setuptools wheel

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "=== 安装 Python 依赖（约 5~15 分钟）==="
pip install -r "$PROJECT/requirements.txt"

echo "=== 验证 CUDA ==="
export KMP_DUPLICATE_LIB_OK=TRUE
python -c "import vllm, torch; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== 运行吞吐对比 benchmark ==="
cd "$PROJECT/src"
python bench_throughput.py || {
    echo ""
    echo "vLLM benchmark 失败（GTX 1060 sm_61 可能不支持 vLLM）"
    echo "尝试仅跑 transformers baseline..."
    python bench_throughput.py --skip-vllm
}

echo ""
echo "=== 完成！结果在 work9/outputs/ ==="
echo "启动 vLLM 服务：cd $PROJECT/src && bash start_server.sh"
