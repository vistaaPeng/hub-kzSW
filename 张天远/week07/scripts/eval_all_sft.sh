#!/bin/bash
set -e

MINICPM_MODEL="openbmb/MiniCPM5-1B"
QLORA_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "=== 重跑 SFT 评估（4 组）==="

echo ""
echo "[1/4] MiniCPM5-1B + peoples_daily"
python src_llm/evaluate_sft.py \
    --model_path "$MINICPM_MODEL" --dataset peoples_daily \
    --ckpt_dir outputs/sft_minicpm5_peoples_daily/sft_adapter --n_samples 200

echo ""
echo "[2/4] MiniCPM5-1B + cluener2020"
python src_llm/evaluate_sft.py \
    --model_path "$MINICPM_MODEL" --dataset cluener2020 \
    --ckpt_dir outputs/sft_minicpm5_cluener2020/sft_adapter --n_samples 200

echo ""
echo "[3/4] Qwen2.5-7B QLoRA + peoples_daily"
python src_llm/evaluate_sft.py \
    --model_path "$QLORA_MODEL" --dataset peoples_daily \
    --ckpt_dir outputs/sft_peoples_daily_qlora_r8 --n_samples 200

echo ""
echo "[4/4] Qwen2.5-7B QLoRA + cluener2020"
python src_llm/evaluate_sft.py \
    --model_path "$QLORA_MODEL" --dataset cluener2020 \
    --ckpt_dir outputs/sft_cluener2020_qlora_r8 --n_samples 200

echo ""
echo "=== 评估完成 ==="
echo "结果文件："
ls -1 outputs/logs/eval_MiniCPM5* outputs/logs/eval_Qwen* 2>/dev/null
