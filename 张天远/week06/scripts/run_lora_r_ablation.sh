#!/bin/bash
# run_lora_r_ablation.sh — LoRA r 消融实验 (r=4/8/16/32)
# 在云端 run_all_cloud.sh 之后跑，总耗时 ~30min
set -e

cd /root/autodl-tmp/text_classification

echo "========================================"
echo " LoRA r 消融实验 — 5K条, 3 epoch"
echo "========================================"

# 1. r=4
echo "[1/4] r=4..."
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 5000 --epochs 3 --lora_r 4 --lora_alpha 8
cp -r outputs/sft_adapter outputs/sft_adapter_r4

# 2. r=8
echo "[2/4] r=8 (baseline)..."
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 5000 --epochs 3 --lora_r 8 --lora_alpha 16
cp -r outputs/sft_adapter outputs/sft_adapter_r8

# 3. r=16
echo "[3/4] r=16..."
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 5000 --epochs 3 --lora_r 16 --lora_alpha 32
cp -r outputs/sft_adapter outputs/sft_adapter_r16

# 4. r=32
echo "[4/4] r=32..."
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 5000 --epochs 3 --lora_r 32 --lora_alpha 64
cp -r outputs/sft_adapter outputs/sft_adapter_r32

# 5. 评估对比（每次保存独立结果文件）
echo ""
echo "=== 评估对比 ==="

python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --ckpt_dir outputs/sft_adapter_r4  --num_samples 200 --seed 42
cp outputs/llm_sft_results.json outputs/llm_sft_results_r4.json

python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --ckpt_dir outputs/sft_adapter_r8  --num_samples 200 --seed 42
cp outputs/llm_sft_results.json outputs/llm_sft_results_r8.json

python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --ckpt_dir outputs/sft_adapter_r16 --num_samples 200 --seed 42
cp outputs/llm_sft_results.json outputs/llm_sft_results_r16.json

python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --ckpt_dir outputs/sft_adapter_r32 --num_samples 200 --seed 42
cp outputs/llm_sft_results.json outputs/llm_sft_results_r32.json

# 6. 打包
echo ""
echo "=== 打包结果 ==="
tar -czf /root/autodl-tmp/r_ablation_results.tar.gz \
    outputs/sft_adapter_r4 outputs/sft_adapter_r8 \
    outputs/sft_adapter_r16 outputs/sft_adapter_r32 \
    outputs/llm_sft_results_r4.json outputs/llm_sft_results_r8.json \
    outputs/llm_sft_results_r16.json outputs/llm_sft_results_r32.json

echo "生成 /root/autodl-tmp/r_ablation_results.tar.gz"

echo ""
echo "结果对比："
for r in 4 8 16 32; do
    acc=$(python -c "import json; d=json.load(open('outputs/llm_sft_results_r${r}.json')); print(f'{d[\"accuracy\"]:.2%}')")
    echo "  r=$r: $acc"
done
