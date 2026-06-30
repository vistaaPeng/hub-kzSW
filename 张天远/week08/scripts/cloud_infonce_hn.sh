#!/bin/bash
# InfoNCE + BQ负例 + batch=128 — 大batch in-batch negatives
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{ set -e; cd /root/autodl-tmp/text_match; echo "日志: $LOGFILE"

echo "=== Step 1: 增强训练集 ==="
/root/miniconda3/bin/python -c "
import json, random; random.seed(42)
afqmc = [json.loads(l) for l in open('data/afqmc/train.jsonl')]
bq    = [json.loads(l) for l in open('data/bq_corpus/train.jsonl')]
bq_s2 = list(set(r['sentence2'] for r in bq))
enhanced = list(afqmc)
for r in afqmc:
    if r['label'] == 1:
        enhanced.append({'sentence1': r['sentence1'], 'sentence2': random.choice(bq_s2), 'label': 0})
with open('data/afqmc/train_infonce_bq.jsonl', 'w') as f:
    for r in enhanced: f.write(json.dumps(r, ensure_ascii=False)+'\n')
print(f'增强: {len(enhanced)} 对 ({len(afqmc)} 原始 + {len(enhanced)-len(afqmc)} BQ负例)')
"

echo "=== Step 2: InfoNCE batch=128 基线 ==="
/root/miniconda3/bin/python src/train_biencoder.py --loss infonce --temperature 0.05 --epochs 3 --batch_size 128
cp outputs/logs/biencoder_infonce_log.json outputs/logs/biencoder_infonce_b128_log.json

echo "=== Step 3: InfoNCE batch=128 + BQ增强 ==="
mv data/afqmc/train.jsonl data/afqmc/train_orig.jsonl
cp data/afqmc/train_infonce_bq.jsonl data/afqmc/train.jsonl
/root/miniconda3/bin/python src/train_biencoder.py --loss infonce --temperature 0.05 --epochs 3 --batch_size 128
mv data/afqmc/train_orig.jsonl data/afqmc/train.jsonl
mv outputs/logs/biencoder_infonce_log.json outputs/logs/biencoder_infonce_bqhn_b128_log.json

echo "=== DONE ==="
} 2>&1 | tee "$LOGFILE"
