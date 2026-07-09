#!/bin/bash
# 分步安装 vLLM（优化下载源）
set -e

PROJECT="/mnt/e/DeepLearning/week9/week9 大模型应用补充知识/work9"

source ~/vllm_env/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
export PIP_DEFAULT_TIMEOUT=600

echo "=== [1/4] 安装 PyTorch（PyTorch 官方 CUDA 源）==="
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

echo "=== [2/4] 安装 vLLM ==="
pip install vllm==0.9.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== [3/4] 安装其余依赖 ==="
pip install transformers==4.52.4 accelerate openai matplotlib numpy requests \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== [4/4] 验证环境 ==="
python -c "
import vllm, torch
print('vLLM:', vllm.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "=== 运行 benchmark ==="
cd "$PROJECT/src"
if python bench_throughput.py; then
    echo "三路 benchmark 完成"
else
    echo "vLLM benchmark 失败，回退 transformers baseline..."
    python bench_throughput.py --skip-vllm
fi

echo "=== 完成！结果目录: $PROJECT/outputs/ ==="
