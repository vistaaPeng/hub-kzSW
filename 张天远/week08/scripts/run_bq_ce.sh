#!/bin/bash
# BQ CrossEncoder 训练 + 防污染
# 跑完后手动拉回: scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/logs/crossencoder_bq_corpus_log.json .
set -e
cd /root/autodl-tmp/text_match

echo "=== Step 1: Archive AFQMC CE ==="
cp outputs/checkpoints/crossencoder_best.pt outputs/checkpoints/crossencoder_afqmc_best.pt
cp outputs/logs/crossencoder_log.json        outputs/logs/crossencoder_afqmc_log.json
echo "  AFQMC archived ok"

echo "=== Step 2: Train BQ CrossEncoder 3ep ==="
/root/miniconda3/bin/python src/train_crossencoder.py --data_dir data/bq_corpus --epochs 3

echo "=== Step 3: Save BQ results ==="
cp outputs/checkpoints/crossencoder_best.pt outputs/checkpoints/crossencoder_bq_corpus_best.pt
cp outputs/logs/crossencoder_log.json        outputs/logs/crossencoder_bq_corpus_log.json
echo "  BQ saved ok"

echo "=== Step 4: Restore AFQMC CE ==="
cp outputs/checkpoints/crossencoder_afqmc_best.pt outputs/checkpoints/crossencoder_best.pt
cp outputs/logs/crossencoder_afqmc_log.json        outputs/logs/crossencoder_log.json
echo "  AFQMC restored ok"

echo ""
echo "=== DONE - BQ CrossEncoder result ==="
python3 -c "
import json
with open('outputs/logs/crossencoder_bq_corpus_log.json') as f:
    data = json.load(f)
for ep in data:
    print(f'  epoch {ep[\"epoch\"]}: loss={ep[\"train_loss\"]:.4f}  val_acc={ep[\"val_acc\"]:.4f}  F1={ep[\"val_f1\"]:.4f}  thr={ep[\"threshold\"]}')
"
