#!/bin/bash
# 一键跑通三方案全流程（约 45 分钟）
set -e

PROJECT="/Users/songqingbin/PycharmProjects/hub-kzSW/宋庆彬/week06/my_text_classification"
PYTHON="/Users/songqingbin/anaconda3/envs/deep_learn/bin/python"

echo "============================================"
echo " 文本分类三方案对比 - 全流程"
echo "============================================"

# 1. 数据准备
echo ""
echo "[1/5] 数据下载..."
cd "$PROJECT/data_pipeline"
$PYTHON download.py

echo ""
echo "[2/5] 数据探索..."
$PYTHON explore.py

# 2. BERT Fine-tune（~10 min）
echo ""
echo "[3/5] BERT Fine-tune 训练（预计 ~10 min）..."
cd "$PROJECT/bert_finetune"
$PYTHON train.py --num_train 10000 --epochs 2 --pool cls
$PYTHON evaluate.py --pool cls

# 3. LLM Zero-shot（~2 min）
echo ""
echo "[4/5] LLM Zero-shot 评估（预计 ~2 min）..."
cd "$PROJECT/llm_zero_shot"
$PYTHON classify.py --num_samples 100

# 4. LLM SFT（~30 min）
echo ""
echo "[5/5] LLM SFT 训练 + 评估（预计 ~30 min）..."
cd "$PROJECT/llm_sft"
$PYTHON train.py --num_train 2000 --epochs 2
$PYTHON evaluate.py --num_samples 100

# 5. 对比
echo ""
echo "============================================"
echo " 三方方案对比"
echo "============================================"
cd "$PROJECT"
$PYTHON compare.py

echo ""
echo "全部完成！"
