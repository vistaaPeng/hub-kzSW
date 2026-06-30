#!/bin/bash
# LCQMC 同域 Hard Neg — 跨数据集反向验证
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{ set -e; cd /root/autodl-tmp/text_match; echo "日志: $LOGFILE"
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_c2_bk.pt
cp outputs/checkpoints/biencoder_cosine_lcqmc_best.pt outputs/checkpoints/biencoder_cosine_best.pt
/root/miniconda3/bin/python src/cross_dataset_hard_neg.py --train_on lcqmc --mine_from lcqmc --top_k 10 --epochs 3
cp outputs/checkpoints/_c2_bk.pt outputs/checkpoints/biencoder_cosine_best.pt; rm outputs/checkpoints/_c2_bk.pt
echo "=== DONE ==="; } 2>&1 | tee "$LOGFILE"
