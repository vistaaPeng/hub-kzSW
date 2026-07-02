#!/bin/bash
# 云端：重训 AFQMC BiCosine + 两阶段检索
set -e
cd /root/autodl-tmp/text_match
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOGFILE" 2>&1
echo "日志: $LOGFILE"

echo "=== Step 1: 重训 AFQMC BiCosine (3ep) ==="
/root/miniconda3/bin/python src/train_biencoder.py --loss cosine --epochs 3

echo "=== Step 2: 存档为纯净 core ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/biencoder_cosine_clean.pt
cp outputs/logs/biencoder_cosine_log.json        outputs/logs/biencoder_cosine_clean_log.json
echo "  clean core saved"

echo "=== Step 3: 跑两阶段检索 ==="
/root/miniconda3/bin/python src/two_stage_retrieval.py --k 100 --data_dir data/afqmc

echo ""
echo "=== DONE ==="
echo "拉回本地："
echo "  scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/logs/two_stage_afqmc.json ."
echo "  scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/checkpoints/biencoder_cosine_clean.pt ."
