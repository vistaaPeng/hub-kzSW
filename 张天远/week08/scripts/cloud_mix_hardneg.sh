#!/bin/bash
# BQ→AFQMC Hard Neg + 50/50 混合采样 — 测试"分布偏移"假说
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{
set -e
cd /root/autodl-tmp/text_match
echo "日志: $LOGFILE"

echo "=== Step 1: 存档 checkpoint ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_mix_bk_bi.pt

echo "=== Step 2: BQ→AFQMC 挖掘 + 混合采样训练 (mix=0.5) ==="
/root/miniconda3/bin/python src/cross_dataset_hard_neg.py --train_on afqmc --mine_from bq_corpus --top_k 10 --epochs 3 --mix_ratio 0.5

echo "=== Step 3: 恢复 ==="
cp outputs/checkpoints/_mix_bk_bi.pt outputs/checkpoints/biencoder_cosine_best.pt
rm outputs/checkpoints/_mix_bk_bi.pt

echo "=== DONE ==="
} 2>&1 | tee "$LOGFILE"
