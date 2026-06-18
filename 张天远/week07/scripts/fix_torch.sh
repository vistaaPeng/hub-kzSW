#!/bin/bash
set -e

echo "=== 修复 torch 环境 ==="

# 清理损坏的 pip 残留
echo "[1/4] 清理残留..."
rm -rf /root/miniconda3/lib/python3.12/site-packages/~orch*

# 卸载旧版本
echo "[2/4] 卸载旧 torch/torchcrf..."
pip uninstall torch torchcrf -y 2>/dev/null || true

# 装 CUDA torch（--no-deps 防止拉 CPU 版）
echo "[3/4] 安装 CUDA torch + pytorch-crf..."
pip install torch==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
pip install pytorch-crf --no-deps

# 验证
echo "[4/4] 验证..."
python -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
import torchcrf
print('torchcrf OK')
"

echo ""
echo "=== 修复完成 ==="
echo "现在可以运行: bash scripts/cloud_run_all.sh"
