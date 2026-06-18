#!/bin/bash
# ============================================================================
# NER 全实验云端跑批 — 支持断点续传 & 手动中断恢复
# ============================================================================
# 特性：
#   - 每个实验完成后写 markers/EXP_ID.done，中断后重跑自动跳过
#   - Ctrl+C 优雅退出：当前实验未完成不写 marker，下次重跑这个
#   - 中间某个实验失败不会影响后续实验（继续跑下一个）
#   - 可单独删某个 marker 重跑那个实验
#
# 使用：
#   bash scripts/cloud_run_all.sh              # 全量跑
#   bash scripts/cloud_run_all.sh --dry-run    # 仅打印不执行
#   bash scripts/cloud_run_all.sh --resume     # 显式续传（默认行为）
#
# 手动重跑某个实验：
#   rm -f markers/roberta_peoples_daily_crf.done
#   bash scripts/cloud_run_all.sh
# ============================================================================

set -o pipefail  # 管道中任一命令失败则整体失败，但不 set -e —— 允许单个实验失败

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# 终端日志自动保存
LOG_DIR="$PROJ_DIR/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cloud_run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

MARKER_DIR="$PROJ_DIR/markers"
mkdir -p "$MARKER_DIR"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

# ── 工具函数 ──────────────────────────────────────────────────────────────

interrupted=false
trap 'interrupted=true; echo ""; echo "[!] 收到中断信号，当前实验未完成，下次从这继续。"' INT TERM

mark_done() { touch "$MARKER_DIR/$1.done"; }
is_done()  { [[ -f "$MARKER_DIR/$1.done" ]]; }
skip()     { echo "  ⏭ 跳过（已完成）"; }

run_step() {
    # usage: run_step EXP_ID "描述" command...
    local id="$1"; local desc="$2"; shift 2

    if is_done "$id"; then
        skip
        return 0
    fi

    echo ""
    echo "── $desc ──────────────────────────────────────────────"
    echo "   ID: $id"
    echo "   $(date)"

    if $DRY_RUN; then
        echo "   [dry-run] $*"
        return 0
    fi

    if "$@"; then
        mark_done "$id"
        echo "  ✓ 完成 → $id"
    else
        echo "  ✗ 失败 ($?) — 跳过，继续下一个实验"
    fi
}

# ── 环境配置 ──────────────────────────────────────────────────────────────

echo "============================================"
echo " NER 全实验跑批 — $(date)"
echo " 已完成实验: $(ls "$MARKER_DIR"/*.done 2>/dev/null | wc -l) / 23"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

conda activate py312 2>/dev/null || true

if ! $DRY_RUN; then
    echo "  检查缺失依赖..."
    # 云镜像自带 torch/transformers，只补缺的
    python -c "
missing = []
for pkg in ['torch', 'transformers', 'seqeval', 'tqdm', 'peft', 'bitsandbytes', 'accelerate']:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)
# pytorch-crf 导入名是 torchcrf（不是 pytorch_crf）
try:
    import torchcrf
except ImportError:
    missing.append('pytorch-crf')
if missing:
    import subprocess, sys
    # 先装非 pytorch-crf 的包（正常安装）
    normal = [p for p in missing if p != 'pytorch-crf']
    if normal:
        print(f'  安装: {\" \".join(normal)}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + normal)
    # pytorch-crf 单独装，--no-deps 防止拉起 CPU torch
    if 'pytorch-crf' in missing:
        print('  安装: pytorch-crf (--no-deps)')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'pytorch-crf'])
else:
    print('  全部依赖已就绪')
"
fi

# ── 模型预热（下载到 HF 缓存）─────────────────────────────────────────────

run_step "download_models" "下载模型" bash -c '
models=("bert-base-chinese" "hfl/chinese-roberta-wwm-ext" "openbmb/MiniCPM5-1B" "Qwen/Qwen2.5-7B-Instruct")
for m in "${models[@]}"; do
    echo "  下载 $m ..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"$m\", trust_remote_code=True)" 2>&1 | tail -1
done
echo "模型下载完成"
'

# ══════════════════════════════════════════════════════════════════════════
# BERT 实验（4 组）
# ══════════════════════════════════════════════════════════════════════════

BERT_MODEL="bert-base-chinese"

for ds in "cluener2020" "peoples_daily"; do
    for crf in "" "--use_crf"; do
        crf_suffix="${crf:+_crf}"
        crf_suffix="${crf_suffix:-_linear}"
        exp_id="bert_${ds}${crf_suffix}"
        desc="BERT + ${ds} + ${crf_suffix#_}"

        run_step "$exp_id" "$desc" bash -c "
            python src/train.py --bert_path '$BERT_MODEL' --dataset '$ds' --epochs 3 --batch_size 32 $crf && \
            python src/evaluate.py --bert_path '$BERT_MODEL' --dataset '$ds' $crf
        "
    done
done

# ══════════════════════════════════════════════════════════════════════════
# RoBERTa 实验（4 组）
# ══════════════════════════════════════════════════════════════════════════

RB_MODEL="hfl/chinese-roberta-wwm-ext"

for ds in "cluener2020" "peoples_daily"; do
    for crf in "" "--use_crf"; do
        crf_suffix="${crf:+_crf}"
        crf_suffix="${crf_suffix:-_linear}"
        exp_id="roberta_${ds}${crf_suffix}"
        desc="RoBERTa + ${ds} + ${crf_suffix#_}"

        run_step "$exp_id" "$desc" bash -c "
            python src/train.py --bert_path '$RB_MODEL' --dataset '$ds' --epochs 3 --batch_size 32 $crf && \
            python src/evaluate.py --bert_path '$RB_MODEL' --dataset '$ds' $crf
        "
    done
done

# ══════════════════════════════════════════════════════════════════════════
# Epoch 消融实验：BERT vs RoBERTa CRF（6 组）
# ══════════════════════════════════════════════════════════════════════════

# 先存档 epoch=3 的结果，防止被后续实验覆盖
for model in "bert-base-chinese" "hfl/chinese-roberta-wwm-ext"; do
    if echo "$model" | grep -q "roberta"; then
        ckpt_suffix="_roberta"; eval_suffix="roberta_"
    else
        ckpt_suffix=""; eval_suffix=""
    fi
    cp -n outputs/checkpoints/best${ckpt_suffix}_crf.pt outputs/checkpoints/best${ckpt_suffix}_crf_e3.pt 2>/dev/null || true
    cp -n outputs/logs/eval${ckpt_suffix}_crf_validation.json outputs/logs/eval${ckpt_suffix}_crf_validation_e3.json 2>/dev/null || true
    cp -n outputs/logs/train${ckpt_suffix}_crf.json outputs/logs/train${ckpt_suffix}_crf_e3.json 2>/dev/null || true
done

for model in "bert-base-chinese" "hfl/chinese-roberta-wwm-ext"; do
    if echo "$model" | grep -q "roberta"; then
        model_tag="roberta"; ckpt_suffix="_roberta"; eval_suffix="roberta_"
    else
        model_tag="bert"; ckpt_suffix=""; eval_suffix=""
    fi

    for epochs in 5 7 10; do
        exp_id="${model_tag}_cluener2020_crf_e${epochs}"
        desc="${model_tag} + cluener2020 + CRF + epochs=${epochs}"

        run_step "$exp_id" "$desc" bash -c "
            python src/train.py --bert_path '$model' --dataset cluener2020 --epochs $epochs --batch_size 32 --use_crf && \
            python src/evaluate.py --bert_path '$model' --dataset cluener2020 --use_crf && \
            cp outputs/checkpoints/best${ckpt_suffix}_crf.pt outputs/checkpoints/best${ckpt_suffix}_crf_e${epochs}.pt && \
            cp outputs/logs/eval${ckpt_suffix}_crf_validation.json outputs/logs/eval${ckpt_suffix}_crf_validation_e${epochs}.json && \
            cp outputs/logs/train${ckpt_suffix}_crf.json outputs/logs/train${ckpt_suffix}_crf_e${epochs}.json
        "
    done
done

# ══════════════════════════════════════════════════════════════════════════
# MiniCPM5-1B LoRA 实验（2 组）— 1B 参数，直接 LoRA 不用量化
# ══════════════════════════════════════════════════════════════════════════

MINICPM_MODEL="openbmb/MiniCPM5-1B"

for ds in "peoples_daily" "cluener2020"; do
    exp_id="minicpm5_${ds}"
    desc="MiniCPM5-1B LoRA + ${ds}"

    run_step "$exp_id" "$desc" bash -c "
        python src_llm/train_sft.py \\
            --model_path '$MINICPM_MODEL' --dataset '$ds' \\
            --num_train 5000 --epochs 3 --batch_size 4 --grad_accum 4 \\
            --lora_r 8 --output_dir 'outputs/sft_minicpm5_${ds}' && \\
        python src_llm/evaluate_sft.py \\
            --model_path '$MINICPM_MODEL' --dataset '$ds' \\
            --ckpt_dir 'outputs/sft_minicpm5_${ds}/sft_adapter' --n_samples 200
    "
done

# ══════════════════════════════════════════════════════════════════════════
# MiniCPM5-1B LoRA r=64 + 全量数据（2 组）
# ══════════════════════════════════════════════════════════════════════════

for ds in "peoples_daily" "cluener2020"; do
    exp_id="minicpm5_r64_${ds}"
    desc="MiniCPM5-1B LoRA r=64 + ${ds} (全量数据)"

    run_step "$exp_id" "$desc" bash -c "
        python src_llm/train_sft.py \\
            --model_path '$MINICPM_MODEL' --dataset '$ds' \\
            --num_train -1 --epochs 3 --batch_size 4 --grad_accum 4 \\
            --lora_r 64 --output_dir 'outputs/sft_minicpm5_r64_${ds}' && \\
        python src_llm/evaluate_sft.py \\
            --model_path '$MINICPM_MODEL' --dataset '$ds' \\
            --ckpt_dir 'outputs/sft_minicpm5_r64_${ds}/sft_adapter' --n_samples 200
    "
done

# ══════════════════════════════════════════════════════════════════════════
# QLoRA 退化验证：peoples_daily + 强制输出实体（1 组）
# ══════════════════════════════════════════════════════════════════════════

QLORA_MODEL="/root/autodl-tmp/huggingface_cache/Qwen2.5-7B-Instruct/Qwen/Qwen2.5-7B-Instruct"

run_step "qlora_peoples_daily_force" "QLoRA Qwen2.5-7B + peoples_daily (强制至少输出一个实体)" bash -c "
    python src_llm/train_sft_qlora.py \
        --model_name '$QLORA_MODEL' --dataset peoples_daily \
        --epochs 3 --num_train 5000 --batch_size 2 --grad_accum 4 \
        --output_tag 'peoples_daily_qlora_force' \
        --prompt_extra '重要提示：请务必在文本中至少识别出一个实体，不要输出空的实体列表。如果没有明显实体，请根据上下文推断最可能的实体类型。' && \
    python src_llm/evaluate_sft.py \
        --model_path '$QLORA_MODEL' --dataset peoples_daily \
        --ckpt_dir 'outputs/sft_peoples_daily_qlora_force' --n_samples 200
"

# ══════════════════════════════════════════════════════════════════════════
# QLoRA 实验（2 组）
# ══════════════════════════════════════════════════════════════════════════

QLORA_MODEL="/root/autodl-tmp/huggingface_cache/Qwen2.5-7B-Instruct/Qwen/Qwen2.5-7B-Instruct"

for ds in "peoples_daily" "cluener2020"; do
    exp_id="qlora_${ds}"
    desc="QLoRA Qwen2.5-7B + ${ds}"

    run_step "$exp_id" "$desc" bash -c "
        python src_llm/train_sft_qlora.py \
            --model_name '$QLORA_MODEL' --dataset '$ds' \
            --epochs 3 --num_train 5000 --batch_size 2 --grad_accum 4 \
            --output_tag '${ds}_qlora_r8' && \
        python src_llm/evaluate_sft.py \
            --model_path '$QLORA_MODEL' --dataset '$ds' \
            --ckpt_dir 'outputs/sft_${ds}_qlora_r8' --n_samples 200
    "
done

# ══════════════════════════════════════════════════════════════════════════
# QLoRA 全量数据验证：peoples_daily 用 20864 条（1 组）
# ══════════════════════════════════════════════════════════════════════════

run_step "qlora_peoples_daily_full" "QLoRA Qwen2.5-7B + peoples_daily (全量数据 20864 条)" bash -c "
    python src_llm/train_sft_qlora.py \\
        --model_name '$QLORA_MODEL' --dataset peoples_daily \\
        --epochs 3 --num_train -1 --batch_size 2 --grad_accum 4 \\
        --output_tag 'peoples_daily_qlora_full' && \\
    python src_llm/evaluate_sft.py \\
        --model_path '$QLORA_MODEL' --dataset peoples_daily \\
        --ckpt_dir 'outputs/sft_peoples_daily_qlora_full' --n_samples 200
"

# ══════════════════════════════════════════════════════════════════════════
# QLoRA 全量数据验证：cluener2020 用 10748 条（1 组）
# ══════════════════════════════════════════════════════════════════════════

run_step "qlora_cluener2020_full" "QLoRA Qwen2.5-7B + cluener2020 (全量数据 10748 条)" bash -c "
    python src_llm/train_sft_qlora.py \\
        --model_name '$QLORA_MODEL' --dataset cluener2020 \\
        --epochs 3 --num_train -1 --batch_size 2 --grad_accum 4 \\
        --output_tag 'cluener2020_qlora_full' && \\
    python src_llm/evaluate_sft.py \\
        --model_path '$QLORA_MODEL' --dataset cluener2020 \\
        --ckpt_dir 'outputs/sft_cluener2020_qlora_full' --n_samples 200
"

# ══════════════════════════════════════════════════════════════════════════
# 汇总 & 关机
# ══════════════════════════════════════════════════════════════════════════

if $DRY_RUN; then
    echo ""
    echo "[dry-run] 跳过"
    exit 0
fi

echo ""
echo "============================================"
echo " 全部实验完成 — $(date)"
echo " 完成实验: $(ls "$MARKER_DIR"/*.done 2>/dev/null | wc -l) / 23"
echo "============================================"
echo ""
echo " 增量结果在 outputs/logs/ 下"
echo ""
echo " 拷贝所需文件到 /root/autodl-fs/："
echo "   cp outputs/logs/eval_<新增实验>.json /root/autodl-fs/"
echo "   shutdown -h now"
echo "============================================"
