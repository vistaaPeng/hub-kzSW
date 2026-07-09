#!/bin/bash
# BQ→AFQMC 小 margin (0.1) Hard Neg — 测试"约束可行性"
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{ set -e; cd /root/autodl-tmp/text_match; echo "日志: $LOGFILE"
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_c3_bk.pt
/root/miniconda3/bin/python src/cross_dataset_hard_neg.py --train_on afqmc --mine_from bq_corpus --margin 0.1 --top_k 10 --epochs 3
cp outputs/checkpoints/_c3_bk.pt outputs/checkpoints/biencoder_cosine_best.pt; rm outputs/checkpoints/_c3_bk.pt
echo "=== DONE ==="; } 2>&1 | tee "$LOGFILE"
