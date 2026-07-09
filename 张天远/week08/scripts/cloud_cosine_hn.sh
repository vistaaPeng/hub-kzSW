#!/bin/bash
# BQ→AFQMC CosineEmbeddingLoss + 纯 HN — 测试"梯度主导"假说
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{ set -e; cd /root/autodl-tmp/text_match; echo "日志: $LOGFILE"
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_c1_bk.pt
/root/miniconda3/bin/python src/cross_dataset_hard_neg.py --train_on afqmc --mine_from bq_corpus --loss cosine --top_k 10 --epochs 3
cp outputs/checkpoints/_c1_bk.pt outputs/checkpoints/biencoder_cosine_best.pt; rm outputs/checkpoints/_c1_bk.pt
echo "=== DONE ==="; } 2>&1 | tee "$LOGFILE"
