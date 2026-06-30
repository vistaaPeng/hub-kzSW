#!/bin/bash
# 诊断：BiEncoder 两种评测协议对比 (BQ)
set -e
cd /root/autodl-tmp/text_match

echo "=== Step 1: Swap in BQ BiEncoder ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_diag_tmp.pt
cp outputs/checkpoints/biencoder_cosine_bq_corpus_best.pt outputs/checkpoints/biencoder_cosine_best.pt
echo "  BQ checkpoint active"

echo "=== Step 2: Run diagnostic ==="
/root/miniconda3/bin/python src/diagnose_ranking.py --data_dir data/bq_corpus --k 100

echo "=== Step 3: Restore AFQMC BiEncoder ==="
cp outputs/checkpoints/_diag_tmp.pt outputs/checkpoints/biencoder_cosine_best.pt
rm outputs/checkpoints/_diag_tmp.pt
echo "  Restored"

echo "=== DONE ==="
cat outputs/logs/diagnose_bq_corpus.json
