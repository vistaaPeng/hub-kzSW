#!/bin/bash
# InfoNCE 基线 — AFQMC 3ep, temperature=0.05
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{ set -e; cd /root/autodl-tmp/text_match; echo "日志: $LOGFILE"

echo "=== InfoNCE 基线 (AFQMC, 3ep, τ=0.05) ==="
/root/miniconda3/bin/python src/train_biencoder.py --loss infonce --temperature 0.05 --epochs 3

echo ""
echo "=== 对比 ==="
echo "  CosineEmbeddingLoss: F1≈0.677 (已知基线)"
echo "  InfoNCE:             F1=$(python3 -c "import json;d=json.load(open('outputs/logs/biencoder_cosine_log.json'));print(f'{max(e[\"val_f1\"] for e in d):.4f}')")"

echo "=== DONE ==="
} 2>&1 | tee "$LOGFILE"
