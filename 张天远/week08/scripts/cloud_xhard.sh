#!/bin/bash
# 跨数据集 Hard Negative Mining: AFQMC基模 → LCQMC挖难负例 → AFQMC训练
set -e
cd /root/autodl-tmp/text_match
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "日志: $LOGFILE"

echo "=== Step 1: 存档 AFQMC checkpoint ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_xhard_bk_bi.pt

echo "=== Step 2: 跨数据集挖掘 + TripletLoss 训练 ==="
/root/miniconda3/bin/python src/cross_dataset_hard_neg.py --top_k 10 --epochs 3

echo "=== Step 3: 恢复 AFQMC checkpoint ==="
cp outputs/checkpoints/_xhard_bk_bi.pt outputs/checkpoints/biencoder_cosine_best.pt
rm outputs/checkpoints/_xhard_bk_bi.pt

echo ""
echo "=== DONE ==="
echo "拉回: scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/logs/biencoder_triplet_xhard_log.json ."
