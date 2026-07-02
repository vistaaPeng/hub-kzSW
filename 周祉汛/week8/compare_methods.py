"""
多方法效果对比脚本

对比三种文本匹配方式在 bq_corpus 和 lcqmc 两个数据集上的效果：
  1. BiEncoder + CosineEmbeddingLoss
  2. BiEncoder + TripletLoss
  3. CrossEncoder + CrossEntropyLoss

使用方式：
  # 评估已有 checkpoint
  python compare_methods.py

  # 指定数据集和评估集
  python compare_methods.py --datasets lcqmc bq_corpus --split validation --batch_size 64

  # 仅评估不训练
  python compare_methods.py --eval_only

依赖：
  pip install torch transformers scikit-learn matplotlib tqdm
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import PairDataset, TripletDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder

# ── 基础路径配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PRETRAINED_BERT = BASE_DIR.parent / "models" / "bert-base-chinese"
DATA_HOME = BASE_DIR / "data"
CHECKPOINT_HOME = BASE_DIR / "outputs" / "checkpoints"
FIGURE_HOME = BASE_DIR / "outputs" / "figures"
LOG_HOME = BASE_DIR / "outputs" / "logs"

SUPPORTED_DATASETS = ["lcqmc", "bq_corpus"]

# ── 方法定义（统一配置） ──────────────────────────────────────────────────
class MethodSpec:
    def __init__(self, key, label, ckpt_pattern, mtype, color):
        self.key = key
        self.label = label
        self.ckpt_pattern = ckpt_pattern  # 用于生成带数据集后缀的名称
        self.mtype = mtype
        self.color = color

METHOD_CONFIGS = [
    MethodSpec(
        key="biencoder_cosine",
        label="BiEncoder (Cosine)",
        ckpt_pattern="biencoder_cosine_{dataset}_best.pt",
        mtype="biencoder",
        color="#2196F3",
    ),
    MethodSpec(
        key="biencoder_triplet",
        label="BiEncoder (Triplet)",
        ckpt_pattern="biencoder_triplet_{dataset}_best.pt",
        mtype="biencoder",
        color="#4CAF50",
    ),
    MethodSpec(
        key="crossencoder",
        label="CrossEncoder",
        ckpt_pattern="crossencoder_{dataset}_best.pt",
        mtype="crossencoder",
        color="#FF9800",
    ),
]

# ── 训练调度 ──────────────────────────────────────────────────────────────
def launch_training(dataset_name, num_layers=4, max_epochs=3, batch_size=32):
    """为指定数据集启动所有方法的训练（如果 checkpoint 缺失）"""
    import subprocess
    import sys

    for spec in METHOD_CONFIGS:
        ckpt_file = CHECKPOINT_HOME / spec.ckpt_pattern.format(dataset=dataset_name)
        if ckpt_file.exists():
            print(f"  [{spec.key}] checkpoint 已存在，跳过训练")
            continue

        print(f"\n{'='*55}")
        print(f"开始训练 {spec.key} 在 {dataset_name} 数据集上...")

        if spec.mtype == "biencoder":
            loss_type = "triplet" if "triplet" in spec.key else "cosine"
            cmd = [
                sys.executable, "train_biencoder.py",
                "--dataset", dataset_name,
                "--loss", loss_type,
                "--num_hidden_layers", str(num_layers),
                "--epochs", str(max_epochs),
                "--batch_size", str(batch_size),
            ]
        else:  # crossencoder
            cmd = [
                sys.executable, "train_crossencoder.py",
                "--dataset", dataset_name,
                "--num_hidden_layers", str(num_layers),
                "--epochs", str(max_epochs),
                "--batch_size", str(batch_size),
            ]

        ret = subprocess.run(cmd, cwd=str(BASE_DIR / "src"))
        if ret.returncode != 0:
            print(f"  [!] 训练失败: {spec.key}")
        else:
            print(f"  [OK] 训练完成: {spec.key}")

# ── 评估加载器 ────────────────────────────────────────────────────────────
def load_and_evaluate(spec, dataset_name, tokenizer, device, split_name, batch_size):
    """加载指定方法的 checkpoint，并返回评估指标字典"""
    # 尝试带数据集名的专用 checkpoint
    ckpt_file = CHECKPOINT_HOME / spec.ckpt_pattern.format(dataset=dataset_name)
    if not ckpt_file.exists():
        # 尝试通用名（回退）
        fallback = CHECKPOINT_HOME / spec.ckpt_pattern.replace("_{dataset}", "")
        if fallback.exists():
            ckpt_file = fallback
        else:
            print(f"  [SKIP] checkpoint 不存在: {ckpt_file}")
            return None

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    data_path = DATA_HOME / dataset_name / f"{split_name}.jsonl"

    if spec.mtype == "biencoder":
        model = build_biencoder(
            bert_path=str(PRETRAINED_BERT),
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds = PairDataset(data_path, tokenizer, max_length=saved_args.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)
    else:
        model = build_crossencoder(
            bert_path=str(PRETRAINED_BERT),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    return metrics

# ── 绘图功能 ──────────────────────────────────────────────────────────────
def draw_accuracy_f1_bars(aggregated, save_path, main_title):
    """为每个数据集绘制 Accuracy 与 F1 的双柱对比图"""
    dataset_names = list(aggregated.keys())
    method_labels = [s.label for s in METHOD_CONFIGS]

    x_pos = np.arange(len(method_labels))
    bar_width = 0.35

    fig, axes = plt.subplots(1, len(dataset_names), figsize=(6 * len(dataset_names), 5), sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        results = aggregated[ds_name]
        acc_vals = [results.get(s.key, {}).get("accuracy", 0) for s in METHOD_CONFIGS]
        f1_vals  = [results.get(s.key, {}).get("f1", 0) for s in METHOD_CONFIGS]
        colors   = [s.color for s in METHOD_CONFIGS]

        bars_acc = ax.bar(x_pos - bar_width/2, acc_vals, bar_width, label="Accuracy",
                          color=colors, alpha=0.85)
        bars_f1  = ax.bar(x_pos + bar_width/2, f1_vals, bar_width, label="F1 (weighted)",
                          color=colors, alpha=0.5, hatch="//", edgecolor="white")

        # 数值标注
        for bar in bars_acc:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_f1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"{ds_name.upper()}")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(main_title, y=1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

def draw_grouped_comparison(aggregated, save_path):
    """绘制两个子图：Accuracy 和 F1 的分组柱状图"""
    dataset_names = list(aggregated.keys())
    method_count = len(METHOD_CONFIGS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：Accuracy
    ax_acc = axes[0]
    x_ticks = np.arange(len(dataset_names))
    width = 0.25
    for i, spec in enumerate(METHOD_CONFIGS):
        accs = [aggregated[ds].get(spec.key, {}).get("accuracy", 0) for ds in dataset_names]
        ax_acc.bar(x_ticks + i * width, accs, width, label=spec.label,
                   color=spec.color, alpha=0.85)
    ax_acc.set_xticks(x_ticks + width)
    ax_acc.set_xticklabels([d.upper() for d in dataset_names])
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Comparison")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.set_ylim(0, 1.1)

    # 子图2：F1
    ax_f1 = axes[1]
    for i, spec in enumerate(METHOD_CONFIGS):
        f1s = [aggregated[ds].get(spec.key, {}).get("f1", 0) for ds in dataset_names]
        ax_f1.bar(x_ticks + i * width, f1s, width, label=spec.label,
                  color=spec.color, alpha=0.85)
    ax_f1.set_xticks(x_ticks + width)
    ax_f1.set_xticklabels([d.upper() for d in dataset_names])
    ax_f1.set_ylabel("F1 (weighted)")
    ax_f1.set_title("F1 Comparison")
    ax_f1.legend(fontsize=8)
    ax_f1.grid(axis="y", alpha=0.3)
    ax_f1.set_ylim(0, 1.1)

    fig.suptitle("Text Matching Methods Comparison on BQ_CORPUS vs LCQMC", y=1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

# ── 命令行参数解析 ──────────────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(description="三种文本匹配方法在两个数据集上的效果对比")
    parser.add_argument("--datasets", default="lcqmc,bq_corpus", help="逗号分隔的数据集")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_only", action="store_true", help="仅评估，不训练")
    parser.add_argument("--num_layers", default=4, type=int, help="BERT 层数")
    parser.add_argument("--epochs", default=3, type=int, help="训练轮数")
    return parser.parse_args()

# ── 主入口 ────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()
    dataset_list = [d.strip() for d in args.datasets.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"运行设备: {device}")
    print(f"目标数据集: {dataset_list}  评估集: {args.split}")

    # 确保目录存在
    CHECKPOINT_HOME.mkdir(parents=True, exist_ok=True)
    FIGURE_HOME.mkdir(parents=True, exist_ok=True)
    LOG_HOME.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(str(PRETRAINED_BERT))

    # 阶段1：训练（如果需要）
    if not args.eval_only:
        for ds in dataset_list:
            launch_training(ds, args.num_layers, args.epochs, args.batch_size)

    # 阶段2：评估所有方法
    aggregated_metrics = {}
    for ds in dataset_list:
        print(f"\n{'='*55}")
        print(f"评估数据集: {ds.upper()}")
        print(f"{'='*55}")

        ds_results = {}
        for spec in METHOD_CONFIGS:
            print(f"\n加载 {spec.key} ...")
            m = load_and_evaluate(spec, ds, tokenizer, device, args.split, args.batch_size)
            if m is not None:
                ds_results[spec.key] = m
                extra_info = f"threshold={m['threshold']:.2f}" if spec.mtype == "biencoder" else "argmax"
                print(f"  Accuracy: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  ({extra_info})")
        aggregated_metrics[ds] = ds_results

    # 保存 JSON 结果
    log_file = LOG_HOME / "method_comparison.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {log_file}")

    # 绘图
    draw_accuracy_f1_bars(
        aggregated_metrics,
        FIGURE_HOME / "method_comparison_by_dataset.png",
        main_title="Text Matching Methods (4-layer BERT, 3 epochs)"
    )
    draw_grouped_comparison(
        aggregated_metrics,
        FIGURE_HOME / "method_comparison_grouped.png"
    )

    # 控制台汇总表格
    print(f"\n{'='*75}")
    print(f"{'数据集':<12} {'方法':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*75}")
    for ds in dataset_list:
        for spec in METHOD_CONFIGS:
            m = aggregated_metrics[ds].get(spec.key, {})
            if m:
                print(f"  {ds:<10} {spec.label:<22} {m['accuracy']:>10.4f} "
                      f"{m['f1']:>10.4f} {m.get('precision', 0):>10.4f} {m.get('recall', 0):>10.4f}")
    print(f"{'='*75}")

    # 最佳方法总结
    for ds in dataset_list:
        best_acc_key = max(aggregated_metrics[ds].items(), key=lambda x: x[1]["accuracy"])[0]
        best_f1_key  = max(aggregated_metrics[ds].items(), key=lambda x: x[1]["f1"])[0]
        print(f"\n{ds.upper()} 结论:")
        print(f"  最高 Accuracy: {best_acc_key} ({aggregated_metrics[ds][best_acc_key]['accuracy']:.4f})")
        print(f"  最高 F1:       {best_f1_key} ({aggregated_metrics[ds][best_f1_key]['f1']:.4f})")

if __name__ == "__main__":
    main()
