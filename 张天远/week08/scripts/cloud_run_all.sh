#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  上云前强制检查清单 (Week08 教训)
# ═══════════════════════════════════════════════════════════════════════
#  □ 1. 每个消融实验后是否 `cp _core.pt → best.pt` 恢复核心 checkpoint？
#       → margin/epoch/pool 消融都会覆盖核心同名文件
#       → 缺失恢复 = 后续所有实验静默拿到错误模型
#  □ 2. 跨数据集实验后 checkpoint 是否重命名加数据集后缀？
#       → BQ/LCQMC 训练后必须 cp 到 _bq_corpus/_lcqmc 版本
#  □ 3. `run_step` 内 `&&` 链中的 cp 是否在训练成功后才执行？
#       → 训练失败时旧日志会被错误复制
#  □ 4. 新增脚本是否已 scp 到云上？（先本地改，再上云）
#  □ 5. train_biencoder.py / train_crossencoder.py 是否最新版？
# ═══════════════════════════════════════════════════════════════════════
# ============================================================================
# 实验清单：
#  核心实验（7 组）：
#   1. 数据探索
#   2. BM25 传统基线
#   3. BiEncoder CosineEmbeddingLoss（4层 3ep）
#   4. BiEncoder TripletLoss（4层 3ep）
#   5. CrossEncoder（4层 3ep）
#   6. 存档核心 checkpoint（防消融覆盖）
#   7. 三方对比 + Bad Case
#  消融实验（7 组）：
#   8. 池化 cls
#   9. 池化 max
#  10. 层数 12 层
#  11. CrossEncoder epoch=1
#  12. CrossEncoder epoch=5
#  13. margin=0.1
#  14. margin=0.5
#  扩展实验（21 组）：
#  15. 两阶段检索（AFQMC）
#  16a. LCQMC BM25         16b. LCQMC BiCosine   16c. LCQMC Triplet
#  16d. LCQMC CrossEnc     16e. LCQMC 规模消融   16f. 混合训练
#  16g. LCQMC SimCSE       16h. LCQMC 两阶段     16i. FAISS 演示
#  16j. BQ BM25            16k. BQ BiCosine      16l. BQ Triplet
#  16m. 领域迁移 3×3
#  17. SimCSE（AFQMC）
#  18. Hard Neg（离线）     18b. Hard Neg（在线）
#  19. LLM Few-shot
#  20a. SFT epoch=1         20b. SFT epoch=3      20c. SFT epoch=5
#  20d. SFT 评估
#
# 使用：
#   bash scripts/cloud_run_all.sh
#   bash scripts/cloud_run_all.sh --dry-run
# ============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

LOG_DIR="$PROJ_DIR/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cloud_run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

MARKER_DIR="$PROJ_DIR/markers"
mkdir -p "$MARKER_DIR"
CKPT_DIR="$PROJ_DIR/outputs/checkpoints"
LOG_DIR_="$PROJ_DIR/outputs/logs"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

TOTAL_EXPS=35

# ── 工具函数 ──────────────────────────────────────────────────────────────

interrupted=false
trap 'interrupted=true; echo ""; echo "[!] 收到中断信号，当前实验未完成，下次从这继续。"' INT TERM

mark_done() { touch "$MARKER_DIR/$1.done"; }
is_done()  { [[ -f "$MARKER_DIR/$1.done" ]]; }
skip()     { echo "  ⏭ 跳过（已完成）"; }

run_step() {
    local id="$1"; local desc="$2"; shift 2
    if is_done "$id"; then skip; return 0; fi
    echo ""
    echo "── $desc ──────────────────────────────────────────────"
    echo "   ID: $id   $(date)"
    if $DRY_RUN; then echo "   [dry-run] $*"; return 0; fi
    if "$@"; then mark_done "$id"; echo "  ✓ 完成 → $id"
    else echo "  ✗ 失败 ($?) — 跳过，继续下一个实验"; fi
}

# ── 环境配置 ──────────────────────────────────────────────────────────────

echo "============================================"
echo " Week08 文本匹配 全实验跑批 — $(date)"
echo " 已完成: $(ls "$MARKER_DIR"/*.done 2>/dev/null | wc -l) / $TOTAL_EXPS"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

export HF_HUB_DISABLE_MMAP=1

if ! $DRY_RUN; then
    echo "  检查缺失依赖..."
    python -c "
pkg_to_import = {'scikit-learn':'sklearn','matplotlib':'matplotlib','tqdm':'tqdm','peft':'peft','datasets':'datasets','faiss-cpu':'faiss'}
missing = []
for pkg, mod in pkg_to_import.items():
    try: __import__(mod)
    except ImportError: missing.append(pkg)
if missing:
    import subprocess, sys
    print(f'  安装: {\" \".join(missing)}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
else:
    print('  全部依赖已就绪')
"
fi

export HF_ENDPOINT=https://hf-mirror.com

# ── 模型预热 ──────────────────────────────────────────────────────────────

run_step "download_models" "下载/预热模型" bash -c '
for m in "bert-base-chinese" "Qwen/Qwen2-0.5B-Instruct"; do
    echo "  加载 $m ..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"$m\", trust_remote_code=True)" 2>&1 | tail -1
done
echo "模型就绪"
'

run_step "download_data" "下载 AFQMC + LCQMC 数据" bash -c '
python src/download_data.py 2>&1
'

# ══════════════════════════════════════════════════════════════════════════
# 核心实验
# ══════════════════════════════════════════════════════════════════════════

run_step "explore_data" "1. 数据探索" \
    python src/explore_data.py

run_step "bm25_baseline" "2. BM25 传统基线" \
    python src/bm25_baseline.py

run_step "biencoder_cosine" "3. BiEncoder CosineEmbeddingLoss（4层, 3 epoch）" \
    python src/train_biencoder.py --loss cosine --epochs 3 --num_hidden_layers 4

run_step "biencoder_triplet" "4. BiEncoder TripletLoss（4层, 3 epoch）" \
    python src/train_biencoder.py --loss triplet --epochs 3 --num_hidden_layers 4

run_step "crossencoder" "5. CrossEncoder（4层, 3 epoch）" \
    python src/train_crossencoder.py --epochs 3 --num_hidden_layers 4

# ══════════════════════════════════════════════════════════════════════════
# 存档核心 checkpoint（消融实验会覆盖同名文件）
# ══════════════════════════════════════════════════════════════════════════

run_step "archive_core" "6. 存档核心 checkpoint" bash -c '
echo "存档核心 checkpoint 和训练日志..."
for ckpt in biencoder_cosine_best.pt biencoder_triplet_best.pt crossencoder_best.pt; do
    if [ -f "outputs/checkpoints/$ckpt" ]; then
        cp -n "outputs/checkpoints/$ckpt" "outputs/checkpoints/${ckpt%.pt}_core.pt" 2>/dev/null && echo "  ✓ $ckpt → ${ckpt%.pt}_core.pt"
    fi
done
for log in biencoder_cosine_log.json biencoder_triplet_log.json crossencoder_log.json; do
    if [ -f "outputs/logs/$log" ]; then
        cp -n "outputs/logs/$log" "outputs/logs/${log%.json}_core.json" 2>/dev/null && echo "  ✓ $log → ${log%.json}_core.json"
    fi
done
echo "存档完成——后续消融实验可安全覆盖默认文件名"
'

# 对比 + Bad Case（依赖核心 checkpoint）

run_step "compare_methods" "7a. 三方对比" \
    python src/compare_methods.py

run_step "analyze_badcases" "7b. Bad Case 分析" \
    python src/analyze_badcases.py

# ══════════════════════════════════════════════════════════════════════════
# 消融实验
# ══════════════════════════════════════════════════════════════════════════

run_step "biencoder_cosine_cls" "8. 消融：池化 cls" \
    python src/train_biencoder.py --loss cosine --pool cls --epochs 3 --num_hidden_layers 4

run_step "biencoder_cosine_max" "9. 消融：池化 max" \
    python src/train_biencoder.py --loss cosine --pool max --epochs 3 --num_hidden_layers 4

run_step "biencoder_cosine_L12" "10. 消融：12 层 BiEncoder Cosine" \
    python src/train_biencoder.py --loss cosine --num_hidden_layers 12 --epochs 3 --batch_size 16

# CrossEncoder epoch 消融
run_step "crossencoder_e1" "11. 消融：CrossEncoder 1 epoch" bash -c '
python src/train_crossencoder.py --epochs 1 --num_hidden_layers 4 && \
cp outputs/checkpoints/crossencoder_best.pt outputs/checkpoints/crossencoder_e1_best.pt && \
cp outputs/logs/crossencoder_log.json    outputs/logs/crossencoder_e1_log.json && \
cp outputs/checkpoints/crossencoder_best_core.pt outputs/checkpoints/crossencoder_best.pt && \
echo "  → crossencoder_e1_best.pt / crossencoder_e1_log.json  (核心 checkpoint 已恢复)"
'

run_step "crossencoder_e5" "12. 消融：CrossEncoder 5 epoch" bash -c '
python src/train_crossencoder.py --epochs 5 --num_hidden_layers 4 && \
cp outputs/checkpoints/crossencoder_best.pt outputs/checkpoints/crossencoder_e5_best.pt && \
cp outputs/logs/crossencoder_log.json    outputs/logs/crossencoder_e5_log.json && \
# 恢复核心 checkpoint（epoch=3）
cp outputs/checkpoints/crossencoder_best_core.pt outputs/checkpoints/crossencoder_best.pt && \
cp outputs/logs/crossencoder_log_core.json      outputs/logs/crossencoder_log.json && \
echo "  → crossencoder_e5_best.pt / crossencoder_e5_log.json  (核心 checkpoint 已恢复)"
'

# margin 消融
run_step "biencoder_cosine_margin01" "13. 消融：margin=0.1" bash -c '
python src/train_biencoder.py --loss cosine --margin 0.1 --epochs 3 --num_hidden_layers 4 && \
cp outputs/logs/biencoder_cosine_log.json outputs/logs/biencoder_cosine_margin01_log.json && \
cp outputs/checkpoints/biencoder_cosine_best_core.pt outputs/checkpoints/biencoder_cosine_best.pt && \
echo "  → biencoder_cosine_margin01_log.json  (核心 checkpoint 已恢复)"
'

run_step "biencoder_cosine_margin05" "14. 消融：margin=0.5" bash -c '
python src/train_biencoder.py --loss cosine --margin 0.5 --epochs 3 --num_hidden_layers 4 && \
cp outputs/logs/biencoder_cosine_log.json outputs/logs/biencoder_cosine_margin05_log.json && \
cp outputs/checkpoints/biencoder_cosine_best_core.pt outputs/checkpoints/biencoder_cosine_best.pt && \
echo "  → biencoder_cosine_margin05_log.json  (核心 checkpoint 已恢复)"
'

# ══════════════════════════════════════════════════════════════════════════
# 扩展实验
# ══════════════════════════════════════════════════════════════════════════

run_step "two_stage_retrieval" "15. 两阶段检索（BiEncoder 召回 → CrossEncoder 精排）" \
    python src/two_stage_retrieval.py --k 100

# LCQMC 跨数据集系列
run_step "lcqmc_bm25" "16a. LCQMC BM25 基线" \
    python src/bm25_baseline.py --data_dir data/lcqmc

run_step "lcqmc_biencoder" "16b. LCQMC BiEncoder Cosine（4层, 3 epoch）" bash -c '
python src/train_biencoder.py --loss cosine --data_dir data/lcqmc --epochs 3 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/biencoder_cosine_lcqmc_best.pt && \
cp outputs/logs/biencoder_cosine_log.json    outputs/logs/biencoder_cosine_lcqmc_log.json && \
echo "  → biencoder_cosine_lcqmc_best.pt / biencoder_cosine_lcqmc_log.json"
'

# LCQMC Length Bias 分析
run_step "lcqmc_length_bias" "16c2. LCQMC Length Bias 分析（分桶 F1）" \
    python src/length_bias_analysis.py --data_dir data/lcqmc

run_step "lcqmc_triplet" "16c. LCQMC BiEncoder Triplet（4层, 3 epoch）" bash -c '
python src/train_biencoder.py --loss triplet --data_dir data/lcqmc --epochs 3 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_triplet_best.pt outputs/checkpoints/biencoder_triplet_lcqmc_best.pt && \
cp outputs/logs/biencoder_triplet_log.json    outputs/logs/biencoder_triplet_lcqmc_log.json && \
echo "  → biencoder_triplet_lcqmc_best.pt / biencoder_triplet_lcqmc_log.json"
'

run_step "lcqmc_crossencoder" "16d. LCQMC CrossEncoder（4层, 1 epoch）" bash -c '
python src/train_crossencoder.py --data_dir data/lcqmc --epochs 1 --num_hidden_layers 4 && \
cp outputs/checkpoints/crossencoder_best.pt outputs/checkpoints/crossencoder_lcqmc_best.pt && \
cp outputs/logs/crossencoder_log.json    outputs/logs/crossencoder_lcqmc_log.json && \
echo "  → crossencoder_lcqmc_best.pt / crossencoder_lcqmc_log.json"
'

# LCQMC 数据规模消融
run_step "lcqmc_scale" "16e. LCQMC 数据规模消融（1K→5K→10K→50K→100K）" \
    python src/data_scale_ablation.py --data_dir data/lcqmc --epochs 2

# LCQMC 预训练 → AFQMC 微调（混合训练）
run_step "hybrid_train" "16f. 混合训练（LCQMC 预训 2 epoch → AFQMC 微调 1 epoch）" bash -c '
# Step 1: LCQMC 预训练 2 epoch
python src/train_biencoder.py --loss cosine --data_dir data/lcqmc --epochs 2 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/biencoder_lcqmc_pretrain.pt && \
echo "  → LCQMC 预训练完成: biencoder_lcqmc_pretrain.pt"

# Step 2: 加载 LCQMC 权重，在 AFQMC 上微调 1 epoch
python src/train_biencoder.py --loss cosine --data_dir data/afqmc --epochs 1 --num_hidden_layers 4 \
    --resume_from outputs/checkpoints/biencoder_lcqmc_pretrain.pt && \
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/biencoder_hybrid_best.pt && \
cp outputs/logs/biencoder_cosine_log.json    outputs/logs/biencoder_hybrid_log.json && \
echo "  → 混合训练完成: biencoder_hybrid_best.pt / biencoder_hybrid_log.json"
'

# LCQMC SimCSE（无监督，天然受益于更多数据）
run_step "lcqmc_simcse" "16g. LCQMC SimCSE 无监督对比学习（3 epoch）" bash -c '
python src/train_simcse.py --data_dir data/lcqmc --epochs 3 --batch_size 64 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_simcse_best.pt outputs/checkpoints/biencoder_simcse_lcqmc_best.pt && \
cp outputs/logs/biencoder_simcse_log.json    outputs/logs/biencoder_simcse_lcqmc_log.json && \
echo "  → biencoder_simcse_lcqmc_best.pt / biencoder_simcse_lcqmc_log.json"
'

# LCQMC 两阶段检索
run_step "lcqmc_two_stage" "16h. LCQMC 两阶段检索（BiEncoder 召回 → CrossEncoder 精排）" \
    python src/two_stage_retrieval.py --k 100 --data_dir data/lcqmc

# FAISS 检索演示
run_step "faiss_demo" "16i. FAISS 检索演示（精确 vs 近似，nprobe 对比）" bash -c '
python src/faiss_demo.py --data_dir data/afqmc --n_queries 200 --nlist 50 && \
python src/faiss_demo.py --data_dir data/lcqmc --n_queries 200 --nlist 200
'

# BQ Corpus 基线系列（金融银行，86K，填补 AFQMC→LCQMC 中间的规模缺口）
run_step "bq_bm25" "16j. BQ BM25 基线" \
    python src/bm25_baseline.py --data_dir data/bq_corpus

run_step "bq_biencoder" "16k. BQ BiEncoder Cosine（4层, 3 epoch）" bash -c '
python src/train_biencoder.py --loss cosine --data_dir data/bq_corpus --epochs 3 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/biencoder_cosine_bq_corpus_best.pt && \
cp outputs/logs/biencoder_cosine_log.json    outputs/logs/biencoder_cosine_bq_corpus_log.json && \
echo "  → biencoder_cosine_bq_corpus_best.pt / biencoder_cosine_bq_corpus_log.json"
'

run_step "bq_triplet" "16l. BQ BiEncoder Triplet（4层, 3 epoch）" bash -c '
python src/train_biencoder.py --loss triplet --data_dir data/bq_corpus --epochs 3 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_triplet_best.pt outputs/checkpoints/biencoder_triplet_bq_corpus_best.pt && \
cp outputs/logs/biencoder_triplet_log.json    outputs/logs/biencoder_triplet_bq_corpus_log.json && \
echo "  → biencoder_triplet_bq_corpus_best.pt / biencoder_triplet_bq_corpus_log.json"
'

# 领域迁移升级为 3×3 矩阵
run_step "domain_transfer_3x3" "16m. 领域迁移 3×3（AFQMC ↔ BQ ↔ LCQMC）" \
    python src/domain_transfer.py --datasets afqmc bq_corpus lcqmc

# SimCSE 无监督对比学习
run_step "simcse" "17. SimCSE 无监督对比学习（3 epoch, batch=64）" \
    python src/train_simcse.py --epochs 3 --batch_size 64 --num_hidden_layers 4

# Hard Negative Mining
run_step "hard_neg_mining" "18. Hard Negative Mining（离线）+ TripletLoss 重训练" \
    python src/hard_negative_mining.py --top_k 5 --epochs 1

# Online Hard Negative Mining（in-batch 在线难负例）
run_step "hard_neg_online" "18b. Online Hard Negative Mining（TripletLoss + in-batch）" bash -c '
python src/train_biencoder.py --loss triplet --hard_neg online --epochs 3 --num_hidden_layers 4 && \
cp outputs/checkpoints/biencoder_triplet_best.pt outputs/checkpoints/biencoder_triplet_online_best.pt && \
cp outputs/logs/biencoder_triplet_log.json    outputs/logs/biencoder_triplet_online_log.json && \
echo "  → biencoder_triplet_online_best.pt / biencoder_triplet_online_log.json"
'

# LLM Few-shot（需 API key，跳过如果没有）
run_step "llm_fewshot" "19. LLM Few-shot 对比（4 示例，100 条采样）" bash -c '
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "[SKIP] DASHSCOPE_API_KEY 未设置，跳过 LLM 实验"
    exit 0
fi
python src_llm/llm_compare.py --num_samples 100 --few_shot --few_shot_n 4
'

# LLM SFT
# LLM SFT — epoch 消融
run_step "sft_lora_e1" "20a. LLM SFT LoRA（1 epoch）" bash -c '
python src_llm/train_sft.py --epochs 1 --num_train 5000 && \
cp -r outputs/sft_adapter outputs/sft_adapter_e1 && \
cp outputs/logs/train_sft.json outputs/logs/train_sft_e1.json && \
echo "  → sft_adapter_e1 / train_sft_e1.json"
'

run_step "sft_lora_e3" "20b. LLM SFT LoRA（3 epoch）" bash -c '
python src_llm/train_sft.py --epochs 3 --num_train 5000 && \
cp -r outputs/sft_adapter outputs/sft_adapter_e3 && \
cp outputs/logs/train_sft.json outputs/logs/train_sft_e3.json && \
echo "  → sft_adapter_e3 / train_sft_e3.json"
'

run_step "sft_lora_e5" "20c. LLM SFT LoRA（5 epoch）" bash -c '
python src_llm/train_sft.py --epochs 5 --num_train 5000 && \
cp -r outputs/sft_adapter outputs/sft_adapter_e5 && \
cp outputs/logs/train_sft.json outputs/logs/train_sft_e5.json && \
echo "  → sft_adapter_e5 / train_sft_e5.json"
'

run_step "sft_evaluate" "20d. SFT 多方评估（1/3/5 epoch 对比）" \
    python src_llm/evaluate_sft.py

# ══════════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════════

if $DRY_RUN; then
    echo ""; echo "[dry-run] 跳过汇总"; exit 0
fi

echo ""
echo "============================================"
echo " 全部实验完成 — $(date)"
echo " 完成: $(ls "$MARKER_DIR"/*.done 2>/dev/null | wc -l) / $TOTAL_EXPS"
echo "============================================"
echo ""
echo "产出清单："
echo "  outputs/checkpoints/"
echo "    biencoder_cosine_best.pt          (核心)"
echo "    biencoder_triplet_best.pt"
echo "    crossencoder_best.pt              (核心, epoch=3)"
echo "    biencoder_cosine_cls_best.pt      (池化消融)"
echo "    biencoder_cosine_max_best.pt"
echo "    biencoder_cosine_L12_best.pt      (层数消融)"
echo "    crossencoder_e1_best.pt           (epoch 消融)"
echo "    crossencoder_e5_best.pt"
echo "    biencoder_cosine_lcqmc_best.pt    (跨数据集)"
echo "    biencoder_triplet_lcqmc_best.pt"
echo "    crossencoder_lcqmc_best.pt"
echo "    biencoder_lcqmc_pretrain.pt       (混合训练预训)"
echo "    biencoder_hybrid_best.pt          (混合训练最终)"
echo "    biencoder_simcse_lcqmc_best.pt    (LCQMC SimCSE)"
echo "    biencoder_simcse_best.pt         (SimCSE)"
echo "    biencoder_cosine_bq_corpus_best.pt (BQ Cosine)"
echo "    biencoder_triplet_bq_corpus_best.pt (BQ Triplet)"
echo "    biencoder_triplet_online_best.pt  (在线 Hard Neg)"
echo "    biencoder_triplet_hardneg_best.pt (离线 Hard Neg)"
echo "  outputs/logs/"
echo "    bm25_afqmc.json                  (AFQMC 传统基线)"
echo "    bm25_lcqmc.json                  (LCQMC 传统基线)"
echo "    bm25_bq_corpus.json               (BQ 传统基线)"
echo "    crossencoder_e1_log.json          (1-epoch Acc 陷阱)"
echo "    crossencoder_e5_log.json"
echo "    biencoder_cosine_margin01_log.json (margin 消融)"
echo "    biencoder_cosine_margin05_log.json"
echo "    biencoder_cosine_lcqmc_log.json    (跨数据集)"
echo "    biencoder_triplet_lcqmc_log.json"
echo "    crossencoder_lcqmc_log.json"
echo "    data_scale_lcqmc.json             (数据规模消融)"
echo "    domain_transfer.json              (领域迁移 3×3)"
echo "    biencoder_hybrid_log.json         (混合训练)"
echo "    biencoder_simcse_lcqmc_log.json   (LCQMC SimCSE)"
echo "    biencoder_simcse_log.json          (SimCSE AFQMC)"
echo "    biencoder_triplet_hardneg_log.json (离线 Hard Neg)"
echo "    biencoder_triplet_online_log.json (在线 Hard Neg)"
echo "    two_stage_retrieval.json          (两阶段 AFQMC)"
echo "    two_stage_lcqmc.json              (两阶段 LCQMC)"
echo "    faiss_demo_afqmc.json             (FAISS AFQMC)"
echo "    faiss_demo_lcqmc.json             (FAISS LCQMC)"
echo "    biencoder_cosine_bq_corpus_log.json (BQ Cosine)"
echo "    biencoder_triplet_bq_corpus_log.json (BQ Triplet)"
echo "    llm_compare_zeroshot.json         (LLM 零样本)"
echo "    llm_compare_fewshot.json          (LLM 少样本)"
echo "    train_sft_e1.json                 (SFT epoch=1)"
echo "    train_sft_e3.json                 (SFT epoch=3)"
echo "    train_sft_e5.json                 (SFT epoch=5)"
echo "  outputs/sft_adapter_e1/  outputs/sft_adapter_e3/  outputs/sft_adapter_e5/"
echo "  outputs/figures/"
echo ""
echo "拉回本地："
echo "  cp -r outputs/checkpoints/ /root/autodl-fs/text_match_checkpoints/"
echo "  cp -r outputs/logs/       /root/autodl-fs/text_match_logs/"
echo "  cp -r outputs/figures/    /root/autodl-fs/text_match_figures/"
echo "  cp -r outputs/sft_adapter_e*/ /root/autodl-fs/text_match_sft/"
echo "============================================"
