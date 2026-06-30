#!/bin/bash
# InfoNCE 温度搜索: τ=0.1, 0.2 — batch=128
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{ set -e; cd /root/autodl-tmp/text_match; echo "日志: $LOGFILE"

for tau in 0.1 0.2; do
    echo "=== InfoNCE b=128 τ=$tau ==="
    /root/miniconda3/bin/python src/train_biencoder.py --loss infonce --temperature $tau --epochs 3 --batch_size 128
    cp outputs/logs/biencoder_infonce_log.json outputs/logs/biencoder_infonce_b128_t${tau/./_}.json
done

echo "=== DONE ==="
} 2>&1 | tee "$LOGFILE"
