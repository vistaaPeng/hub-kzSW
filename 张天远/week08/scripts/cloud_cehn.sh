#!/bin/bash
# CrossEncoder 挖掘 Hard Negative（AFQMC 同域）
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{
set -e
cd /root/autodl-tmp/text_match
echo "日志: $LOGFILE"

echo "=== Step 1: 存档 checkpoint ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_cehn_bk_bi.pt

echo "=== Step 2: CE 挖掘 + TripletLoss 训练 ==="
/root/miniconda3/bin/python src/ce_hard_neg_mining.py --train_on afqmc --mine_from afqmc --bi_top_k 50 --epochs 3

echo "=== Step 3: 恢复 ==="
cp outputs/checkpoints/_cehn_bk_bi.pt outputs/checkpoints/biencoder_cosine_best.pt
rm outputs/checkpoints/_cehn_bk_bi.pt

echo "=== DONE ==="
} 2>&1 | tee "$LOGFILE"
