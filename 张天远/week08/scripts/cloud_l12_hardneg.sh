#!/bin/bash
# 12 层 BERT 同数据集离线 Hard Neg —— 直接测试"容量瓶颈"假说
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
{
set -e
cd /root/autodl-tmp/text_match
echo "日志: $LOGFILE"

echo "=== Step 1: 存档 4 层 checkpoint ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_l12hn_bk_bi.pt

echo "=== Step 2: 换入 12 层 BiEncoder ==="
cp outputs/checkpoints/biencoder_cosine_L12_best.pt outputs/checkpoints/biencoder_cosine_best.pt

echo "=== Step 3: 离线挖掘 + TripletLoss 重训练 ==="
/root/miniconda3/bin/python src/hard_negative_mining.py --top_k 10 --epochs 3

echo "=== Step 4: 保存 12 层结果 ==="
cp outputs/checkpoints/biencoder_triplet_hardneg_best.pt outputs/checkpoints/biencoder_triplet_hardneg_L12_best.pt
cp outputs/logs/biencoder_triplet_hardneg_log.json        outputs/logs/biencoder_triplet_hardneg_L12_log.json

echo "=== Step 5: 恢复 4 层 checkpoint ==="
cp outputs/checkpoints/_l12hn_bk_bi.pt outputs/checkpoints/biencoder_cosine_best.pt
rm outputs/checkpoints/_l12hn_bk_bi.pt

echo ""
echo "=== DONE ==="
echo "拉回: scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/logs/biencoder_triplet_hardneg_L12_log.json ."
} 2>&1 | tee "$LOGFILE"
