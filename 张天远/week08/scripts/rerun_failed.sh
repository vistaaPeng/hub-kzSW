#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  上云前强制检查清单 (Week08 教训) — 每次打开这个文件先看这里
# ═══════════════════════════════════════════════════════════════════════
#  □ 1. 每个消融实验后是否 `cp _core.pt → best.pt` 恢复核心？
#  □ 2. 跨数据集实验 checkpoint 是否重命名加数据集后缀？
#  □ 3. `&&` 链中的 cp 是否在训练成功后才执行？
#  □ 4. 新增脚本是否已 scp 到云上？
#  □ 5. train_biencoder.py / train_crossencoder.py 是否最新版？
# ═══════════════════════════════════════════════════════════════════════
# ============================================================================
# Week08 重跑脚本 — 修复 5 个异常实验 + 结果验证
# ============================================================================
# 每个实验分三步：前置检查 → 执行 → 结果验证
# 任一环节失败都有明确错误信息，不会静默跳过
#
# 使用：
#   bash scripts/rerun_failed.sh
# ============================================================================

set -e  # 与 cloud_run_all 不同——这里每个实验都不能静默失败

cd "$(dirname "$0")/.."
PY=/root/miniconda3/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_DIR="outputs/logs"
CKPT_DIR="outputs/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo " Week08 异常修复重跑 — $TIMESTAMP"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

# ── 前置检查 ──────────────────────────────────────────────────────────────

echo ""
echo "── 前置：验证代码修复是否就绪 ─────────────────────"

# 检查 1: two_stage_retrieval.py 是否包含批量修复
if grep -q 's1_list' src/two_stage_retrieval.py 2>/dev/null; then
    echo "  ✓ two_stage_retrieval.py 批量修复已就绪"
else
    echo "  ✗ two_stage_retrieval.py 缺少批量修复！请先 scp 上传修复后的文件"
    exit 1
fi

# 检查 2: train_biencoder.py 是否包含 --hard_neg 参数
if grep -q 'hard_neg' src/train_biencoder.py 2>/dev/null; then
    echo "  ✓ train_biencoder.py --hard_neg 参数已就绪"
else
    echo "  ✗ train_biencoder.py 缺少 --hard_neg！请先 scp 上传修复后的文件"
    exit 1
fi

# 检查 3: BQ 数据是否就绪
if [ -f "data/bq_corpus/train.jsonl" ]; then
    BQ_ROWS=$(wc -l < data/bq_corpus/train.jsonl)
    echo "  ✓ BQ 数据就绪（$BQ_ROWS 条）"
else
    echo "  ✗ data/bq_corpus/train.jsonl 不存在！"
    exit 1
fi

echo "  前置检查全部通过"

# ══════════════════════════════════════════════════════════════════════════
# 实验 1: BQ BiEncoder Cosine
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 1/7: BQ BiEncoder Cosine ────────────────────"

# 先备份当前的 log（防止覆写后无法对比）
if [ -f "$LOG_DIR/biencoder_cosine_log.json" ]; then
    cp "$LOG_DIR/biencoder_cosine_log.json" "/tmp/biencoder_cosine_backup_$TIMESTAMP.json"
    echo "  已备份当前 log"
fi

echo "  开始训练..."
$PY src/train_biencoder.py --loss cosine --data_dir data/bq_corpus --epochs 3 --num_hidden_layers 4

# 立即复制——不等任何其他操作
cp "$CKPT_DIR/biencoder_cosine_best.pt" "$CKPT_DIR/biencoder_cosine_bq_corpus_best.pt"
cp "$LOG_DIR/biencoder_cosine_log.json"  "$LOG_DIR/biencoder_cosine_bq_corpus_log.json"

# 恢复 AFQMC 核心 checkpoint（后续脚本依赖 biencoder_cosine_best.pt = AFQMC）
if [ -f "$CKPT_DIR/biencoder_cosine_best_core.pt" ]; then
    cp "$CKPT_DIR/biencoder_cosine_best_core.pt" "$CKPT_DIR/biencoder_cosine_best.pt"
    cp "$LOG_DIR/biencoder_cosine_log_core.json"  "$LOG_DIR/biencoder_cosine_log.json"
    echo "  已恢复 AFQMC 核心 checkpoint → biencoder_cosine_best.pt"
else
    echo "  ⚠️  biencoder_cosine_best_core.pt 不存在，无法恢复 AFQMC checkpoint"
    echo "     biencoder_cosine_best.pt 目前是 BQ 模型！"
fi

# 验证：BQ Cosine 的 F1 不应与 AFQMC Cosine 完全相同
BQ_F1=$($PY -c "import json; d=json.load(open('$LOG_DIR/biencoder_cosine_bq_corpus_log.json')); print(d[-1]['val_f1'])")
AFQMC_CORE_F1=$($PY -c "import json; d=json.load(open('$LOG_DIR/biencoder_cosine_log_core.json')); print(d[-1]['val_f1'])" 2>/dev/null || echo "0")

echo "  BQ Cosine F1: $BQ_F1"
echo "  AFQMC Core F1: $AFQMC_CORE_F1"

if [ "$BQ_F1" = "$AFQMC_CORE_F1" ]; then
    echo ""
    echo "  ⚠️  WARNING: BQ F1 与 AFQMC Core F1 完全相同 ($BQ_F1)！"
    echo "  可能原因："
    echo "    1. --data_dir 参数未生效，实际训练用了 AFQMC 数据"
    echo "    2. BQ 数据集特征与 AFQMC 高度相似"
    echo "  建议：检查训练日志中的 train dataset size 是否 ≈ 69K（BQ）而非 34K（AFQMC）"
else
    echo "  ✓ 验证通过：BQ F1 ($BQ_F1) ≠ AFQMC F1 ($AFQMC_CORE_F1)"
fi

echo "  → biencoder_cosine_bq_corpus_best.pt / biencoder_cosine_bq_corpus_log.json"

# ══════════════════════════════════════════════════════════════════════════
# 实验 2: 两阶段 AFQMC
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 2/7: 两阶段 AFQMC ──────────────────────────"

$PY src/two_stage_retrieval.py --k 100

if [ -f "$LOG_DIR/two_stage_afqmc.json" ]; then
    echo "  ✓ 日志已生成 → two_stage_afqmc.json"
elif [ -f "$LOG_DIR/two_stage_retrieval.json" ]; then
    echo "  ✓ 日志已生成 → two_stage_retrieval.json"
else
    echo "  ⚠️  日志文件未找到，检查 outputs/logs/ 下 two_stage*"
fi

# ══════════════════════════════════════════════════════════════════════════
# 实验 3: CrossEncoder 5-epoch
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 3/7: CrossEncoder 5-epoch ───────────────────"

$PY src/train_crossencoder.py --epochs 5 --num_hidden_layers 4

# 立即复制
cp "$CKPT_DIR/crossencoder_best.pt" "$CKPT_DIR/crossencoder_e5_best.pt"
cp "$LOG_DIR/crossencoder_log.json"  "$LOG_DIR/crossencoder_e5_log.json"

# 验证：e5 日志应有 5 个 epoch
E5_EPOCHS=$($PY -c "import json; print(len(json.load(open('$LOG_DIR/crossencoder_e5_log.json'))))")
echo "  e5 日志 epoch 数: $E5_EPOCHS"
if [ "$E5_EPOCHS" -lt 5 ]; then
    echo "  ✗ FAIL: 预期 5 epoch，实际只有 $E5_EPOCHS。训练可能提前失败。"
else
    echo "  ✓ 验证通过：完整 5 epoch"
fi

# 恢复核心 checkpoint（不影响后续使用）
if [ -f "$CKPT_DIR/crossencoder_best_core.pt" ]; then
    cp "$CKPT_DIR/crossencoder_best_core.pt" "$CKPT_DIR/crossencoder_best.pt"
    cp "$LOG_DIR/crossencoder_log_core.json"  "$LOG_DIR/crossencoder_log.json"
    echo "  核心 checkpoint 已恢复"
fi

# ══════════════════════════════════════════════════════════════════════════
# 实验 4: 离线 Hard Neg（调大 top_k）
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 4/7: 离线 Hard Neg (top_k=20) ──────────────"

$PY src/hard_negative_mining.py --top_k 20 --epochs 1 --margin 0.3

cp "$CKPT_DIR/biencoder_triplet_hardneg_best.pt" "$CKPT_DIR/biencoder_triplet_hardneg_v2_best.pt"
cp "$LOG_DIR/biencoder_triplet_hardneg_log.json"  "$LOG_DIR/biencoder_triplet_hardneg_v2_log.json"

# 验证：threshold 不应为 1.0
HN_THR=$($PY -c "import json; d=json.load(open('$LOG_DIR/biencoder_triplet_hardneg_v2_log.json')); print(d[-1].get('threshold', '?'))")
HN_F1=$($PY -c "import json; d=json.load(open('$LOG_DIR/biencoder_triplet_hardneg_v2_log.json')); print(d[-1].get('val_f1', '?'))")
echo "  threshold=$HN_THR  F1=$HN_F1"
if [ "$HN_THR" = "1.0" ]; then
    echo "  ⚠️  threshold 仍为 1.0，向量空间可能仍塌缩。尝试减小 margin 到 0.1"
else
    echo "  ✓ threshold 正常"
fi

# ══════════════════════════════════════════════════════════════════════════
# 实验 5: 在线 Hard Neg（验证新代码）
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 5/7: 在线 Hard Neg ─────────────────────────"

# 验证 --hard_neg 参数确实被接受
$PY src/train_biencoder.py --help 2>&1 | grep -q 'hard_neg' || {
    echo "  ✗ --hard_neg 参数未在 train_biencoder.py 中注册！"
    exit 1
}

$PY src/train_biencoder.py --loss triplet --hard_neg online --epochs 3 --num_hidden_layers 4

cp "$CKPT_DIR/biencoder_triplet_best.pt" "$CKPT_DIR/biencoder_triplet_online_v2_best.pt"
cp "$LOG_DIR/biencoder_triplet_log.json"  "$LOG_DIR/biencoder_triplet_online_v2_log.json"

ON_F1=$($PY -c "import json; d=json.load(open('$LOG_DIR/biencoder_triplet_online_v2_log.json')); print(d[-1]['val_f1'])")
RANDOM_F1=$($PY -c "import json; d=json.load(open('$LOG_DIR/biencoder_triplet_log_core.json')); print(d[-1]['val_f1'])" 2>/dev/null || echo "0")
echo "  在线 Hard Neg F1: $ON_F1"
echo "  随机 Triplet F1:  $RANDOM_F1"

if [ "$ON_F1" = "$RANDOM_F1" ]; then
    echo "  ⚠️  在线 Hard Neg F1 与随机 Triplet 相同 ($ON_F1)"
    echo "  可能原因："
    echo "    1. 文件上传后未覆盖（旧版 train_biencoder.py 没有 --hard_neg 逻辑）"
    echo "    2. 在线难负例在当前 batch size 下没有提供足够多样的负例"
else
    echo "  ✓ 在线 Hard Neg 产生了不同的 F1"
fi

# ══════════════════════════════════════════════════════════════════════════
# 实验 6: FAISS demo AFQMC（受 margin ablation 污染）
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 6/7: FAISS demo AFQMC ──────────────────────"

$PY src/faiss_demo.py --data_dir data/afqmc --n_queries 200 --nlist 50

if [ -f "$LOG_DIR/faiss_demo_afqmc.json" ]; then
    echo "  ✓ 日志已生成 → faiss_demo_afqmc.json"
else
    echo "  ⚠️  日志未生成，检查 outputs/logs/"
fi

# ══════════════════════════════════════════════════════════════════════════
# 实验 7: 领域迁移 3×3（受 BQ Cosine + AFQMC checkpoint 污染）
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "── 实验 7/7: 领域迁移 3×3 ──────────────────────────"

$PY src/domain_transfer.py --datasets afqmc bq_corpus lcqmc

if [ -f "$LOG_DIR/domain_transfer.json" ]; then
    echo "  ✓ 日志已生成 → domain_transfer.json"
else
    echo "  ⚠️  日志未生成，检查 outputs/logs/"
fi
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo " 重跑完成 — $(date)"
echo "============================================"
echo ""
echo "新增产出（7 个实验）："
echo "  $LOG_DIR/biencoder_cosine_bq_corpus_log.json"
echo "  $LOG_DIR/two_stage_afqmc.json (或 two_stage_retrieval.json)"
echo "  $LOG_DIR/crossencoder_e5_log.json"
echo "  $LOG_DIR/biencoder_triplet_hardneg_v2_log.json"
echo "  $LOG_DIR/biencoder_triplet_online_v2_log.json"
echo "  $LOG_DIR/faiss_demo_afqmc.json"
echo "  $LOG_DIR/domain_transfer.json"
echo ""
echo "拉回本地："
echo "  scp -P 28197 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/text_match/outputs/logs/biencoder_cosine_bq_corpus_log.json E:\\npl\\workspaces\\npl_tran\\text_match\\outputs\\logs\\"
echo "  ... (其余同理)"
echo ""
echo "⚠️  关注上面带 WARNING 的实验——它们可能需要进一步排查而非仅重跑。"
