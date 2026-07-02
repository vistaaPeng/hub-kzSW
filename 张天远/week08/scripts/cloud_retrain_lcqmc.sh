#!/bin/bash
# 云端：重训 LCQMC BiCosine + 两阶段检索
set -e
cd /root/autodl-tmp/text_match
LOGFILE="outputs/logs/cloud_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOGFILE" 2>&1
echo "日志: $LOGFILE"

echo "=== Step 1: 存档 AFQMC checkpoint ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_bk_afqmc_bi.pt
cp outputs/checkpoints/crossencoder_best.pt       outputs/checkpoints/_bk_afqmc_ce.pt

echo "=== Step 2: 重训 LCQMC BiCosine (3ep) ==="
/root/miniconda3/bin/python src/train_biencoder.py --loss cosine --data_dir data/lcqmc --epochs 3

echo "=== Step 3: 换入 LCQMC CrossEncoder ==="
cp outputs/checkpoints/crossencoder_lcqmc_best.pt outputs/checkpoints/crossencoder_best.pt

echo "=== Step 4: 跑两阶段检索 ==="
/root/miniconda3/bin/python src/two_stage_retrieval.py --k 100 --data_dir data/lcqmc

echo "=== Step 5: 恢复 AFQMC checkpoint ==="
cp outputs/checkpoints/_bk_afqmc_bi.pt outputs/checkpoints/biencoder_cosine_best.pt
cp outputs/checkpoints/_bk_afqmc_ce.pt outputs/checkpoints/crossencoder_best.pt
rm outputs/checkpoints/_bk_afqmc_bi.pt outputs/checkpoints/_bk_afqmc_ce.pt

echo ""
echo "=== DONE ==="
echo "拉回: scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/logs/two_stage_lcqmc.json ."
