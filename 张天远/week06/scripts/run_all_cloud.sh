#!/bin/bash
# run_all_cloud.sh — 文本分类三种方案 云端完整训练+评估，跑完自动关机
# 使用方法: bash run_all_cloud.sh
# 注意: 请先确认 conda/native python 环境已激活

set -e  # 任一步出错就停，方便排查

cd /root/autodl-tmp/text_classification

# ── 数据确认 ──────────────────────────────────────────────
echo "=== 检查数据 ==="
ls data/train.json data/val.json data/test.json data/label_map.json || {
    echo "数据缺失，请先上传 data.tar.gz 并解压"
    exit 1
}

# ── 1. BERT 普通 loss ─────────────────────────────────────
echo "[1/8] BERT 微调 - 普通 CrossEntropyLoss"
python src/train.py --bert_path bert-base-chinese --epochs 3 --batch_size 16 --max_length 64

# ── 2. BERT 加权 loss ────────────────────────────────────
echo "[2/8] BERT 微调 - 加权 CrossEntropyLoss"
python src/train.py --bert_path bert-base-chinese --epochs 3 --batch_size 16 --max_length 64 --use_class_weight

# ── 3. SFT+LoRA ───────────────────────────────────────────
echo "[3/8] LLM SFT + LoRA - Qwen2-0.5B, r=8, 5K条"
python src_llm/train_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_train 5000 --epochs 3

# ── 4. LLM 零样本 ─────────────────────────────────────────
echo "[4/8] LLM 零样本 - 200 条评估"
python src_llm/classify_llm.py --model_path Qwen/Qwen2-0.5B-Instruct --num_samples 200 --seed 42

# ── 5. SFT 评估 ───────────────────────────────────────────
echo "[5/8] LLM SFT+LoRA - 200 条评估"
python src_llm/evaluate_sft.py --model_path Qwen/Qwen2-0.5B-Instruct --num_samples 200 --seed 42

# ── 6. BERT 评估 ──────────────────────────────────────────
echo "[6/8] BERT 评估 - cls"
python src/evaluate.py --pool cls --bert_path bert-base-chinese

# ── 7. BERT 加权评估 ─────────────────────────────────────
echo "[7/8] BERT 评估 - cls_weighted"
python src/evaluate.py --pool cls_weighted --bert_path bert-base-chinese

# ── 8. 消融对比 ─────────────────────────────────────────
echo "[8/8] 加权 Loss 消融实验"
python src/compare_class_weight.py --pool cls --bert_path bert-base-chinese

# ── 打包结果 ──────────────────────────────────────────────
echo "=== 打包结果 ==="
tar -czf /root/autodl-tmp/results.tar.gz outputs/
echo "结果已打包到 /root/autodl-tmp/results.tar.gz"
cp /root/autodl-tmp/results.tar.gz  /root/autodl-fs/
# ── 关机 ────────────────────────────────────────────────
echo "=== 全部完成，30 秒后关机 ==="
sleep 30
shutdown -h now
