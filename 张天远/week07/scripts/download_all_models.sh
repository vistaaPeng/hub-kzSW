#!/bin/bash
# 在无卡实例上运行：下载所有模型（含完整权重），再镜像克隆到 GPU 实例
# 目的：避免 GPU 计费时等待下载

set -e

echo "=== 下载所有模型权重（预计 10-20 分钟）==="

# BERT + RoBERTa：tokenizer + 模型权重
for m in "bert-base-chinese" "hfl/chinese-roberta-wwm-ext"; do
    echo ""
    echo "--- 下载 $m ---"
    python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('$m', trust_remote_code=True)
AutoModel.from_pretrained('$m')
print('OK')
"
done

# MiniCPM5-1B：tokenizer + 完整模型（~2GB）
echo ""
echo "--- 下载 openbmb/MiniCPM5-1B ---"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('openbmb/MiniCPM5-1B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('openbmb/MiniCPM5-1B', trust_remote_code=True)
print('OK')
"

# Qwen2.5-7B QLoRA：tokenizer + 4-bit 量化权重（~14GB）
echo ""
echo "--- 下载 Qwen/Qwen2.5-7B-Instruct (4-bit) ---"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype='float16', bnb_4bit_quant_type='nf4')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', quantization_config=bnb, trust_remote_code=True)
print('OK')
"

echo ""
echo "=== 全部下载完成 ===\necho \"现在可以镜像克隆到 GPU 实例了\""
