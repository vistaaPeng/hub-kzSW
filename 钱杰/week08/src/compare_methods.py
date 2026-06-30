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

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
BERT_PATH  = ROOT.parent / "models" / "bert-base-chinese"
DATA_ROOT  = ROOT / "data"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
FIG_DIR    = ROOT / "outputs" / "figures"
LOG_DIR    = ROOT / "outputs" / "logs"

DATASETS   = ["lcqmc", "bq_corpus"]
METHODS    = [
    {
        "key":   "biencoder_cosine",
        "label": "BiEncoder (Cosine)",
        "ckpt": "biencoder_cosine_best.pt",
        "type": "biencoder",
        "color": "#2196F3",
    },
    {
        "key":   "biencoder_triplet",
        "label": "BiEncoder (Triplet)",
        "ckpt": "biencoder_triplet_best.pt",
        "type": "biencoder",
        "color": "#4CAF50",
    },
    {
        "key":   "crossencoder",
        "label": "CrossEncoder",
        "ckpt": "crossencoder_best.pt",
        "type": "crossencoder",
        "color": "#FF9800",
    },
]


# ── 训练脚本调用 ──────────────────────────────────────────────────────────

def train_all_methods(dataset, num_layers=4, epochs=3, batch_size=32):
    """训练指定数据集上的所有方法"""
    import subprocess
    import sys

    results = {}
    for method in METHODS:
        ckpt_path = CKPT_DIR / method["ckpt"]
        if ckpt_path.exists():
            print(f"  [{method['key']}] checkpoint 已存在，跳过训练")
            continue

        print(f"\n{'='*55}")
        print(f"训练 {method['key']} 在 {dataset} 上...")

        if method["type"] == "biencoder":
            loss_type = "triplet" if "triplet" in method["key"] else "cosine"
            cmd = [
                sys.executable, "train_biencoder.py",
                "--dataset", dataset,
                "--loss", loss_type,
                "--num_hidden_layers", str(num_layers),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
            ]
        else:
            cmd = [
                sys.executable, "train_crossencoder.py",
                "--dataset", dataset,
                "--num_hidden_layers", str(num_layers),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
            ]

        # 在 src 目录下运行
        result = subprocess.run(cmd, cwd=str(ROOT / "src"))
        if result.returncode != 0:
            print(f"  [!] 训练失败: {method['key']}")
        else:
            print(f"  [OK] 训练完成: {method['key']}")

    return results


# ── 加载并评估单个方法 ─────────────────────────────────────────────────────

def load_and_eval(method, dataset, tokenizer, device, split, batch_size):
    ckpt_name = method["ckpt"].replace("_best.pt", f"_{dataset}_best.pt")
    # 检查带数据集名的 checkpoint
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        # 回退到通用 checkpoint 名
        ckpt_path = CKPT_DIR / method["ckpt"]

    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    data_path = DATA_ROOT / dataset / f"{split}.jsonl"

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds     = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)
    else:
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    return metrics


# ── 对比可视化 ────────────────────────────────────────────────────────────

def plot_comparison_bar(all_results, save_path, title):
    """准确率 / F1 对比柱状图"""
    datasets = list(all_results.keys())
    methods_labels = [m["label"] for m in METHODS]

    x = np.arange(len(methods_labels))
    w = 0.35

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        results = all_results[dataset]
        accs = [results.get(m["key"], {}).get("accuracy", 0) for m in METHODS]
        f1s  = [results.get(m["key"], {}).get("f1", 0) for m in METHODS]
        colors = [m["color"] for m in METHODS]

        bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.85)
        bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color=colors, alpha=0.5,
                       hatch="//", edgecolor="white")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in METHODS], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"{dataset.upper()}")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_grouped_comparison(all_results, save_path):
    """分组对比图：每个指标一个子图"""
    datasets = list(all_results.keys())
    n_methods = len(METHODS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    x = np.arange(len(datasets))
    w = 0.25
    for i, method in enumerate(METHODS):
        accs = [all_results[d].get(method["key"], {}).get("accuracy", 0) for d in datasets]
        ax.bar(x + i * w, accs, w, label=method["label"], color=method["color"], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    # F1
    ax = axes[1]
    for i, method in enumerate(METHODS):
        f1s = [all_results[d].get(method["key"], {}).get("f1", 0) for d in datasets]
        ax.bar(x + i * w, f1s, w, label=method["label"], color=method["color"], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("F1 Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    fig.suptitle("Text Matching Methods Comparison on BQ_CORPUS vs LCQMC", y=1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="三种文本匹配方法在两个数据集上的效果对比")
    parser.add_argument("--datasets",   default="lcqmc,bq_corpus", help="逗号分隔的数据集")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_only",  action="store_true", help="仅评估，不训练")
    parser.add_argument("--num_layers", default=4, type=int, help="BERT 层数")
    parser.add_argument("--epochs",     default=3, type=int, help="训练轮数")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"设备: {device}")
    print(f"数据集: {datasets}  评估集: {args.split}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 训练（如需要）
    if not args.eval_only:
        for dataset in datasets:
            train_all_methods(dataset, args.num_layers, args.epochs, args.batch_size)

    # 评估
    all_results = {}
    for dataset in datasets:
        print(f"\n{'='*55}")
        print(f"评估数据集: {dataset.upper()}")
        print(f"{'='*55}")

        results = {}
        for method in METHODS:
            print(f"\n加载 {method['key']}...")
            metrics = load_and_eval(method, dataset, tokenizer, device, args.split, args.batch_size)
            if metrics:
                results[method["key"]] = metrics
                extra = f"threshold={metrics['threshold']:.2f}" if method["type"] == "biencoder" else "argmax"
                print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}  ({extra})")

        all_results[dataset] = results

    # 保存结果
    log_path = LOG_DIR / "method_comparison.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {log_path}")

    # 可视化
    plot_comparison_bar(
        all_results,
        FIG_DIR / "method_comparison_by_dataset.png",
        title="Text Matching Methods (4-layer BERT, 3 epochs)"
    )
    plot_grouped_comparison(
        all_results,
        FIG_DIR / "method_comparison_grouped.png"
    )

    # 控制台汇总表
    print(f"\n{'='*75}")
    print(f"{'数据集':<12} {'方法':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*75}")
    for dataset in datasets:
        for method in METHODS:
            m = all_results[dataset].get(method["key"], {})
            if m:
                print(f"  {dataset:<10} {method['label']:<22} {m['accuracy']:>10.4f} "
                      f"{m['f1']:>10.4f} {m.get('precision', 0):>10.4f} {m.get('recall', 0):>10.4f}")
    print(f"{'='*75}")

    # 找出每个数据集的最佳方法
    for dataset in datasets:
        best_acc_key = max(all_results[dataset].items(), key=lambda x: x[1]["accuracy"])[0]
        best_f1_key  = max(all_results[dataset].items(), key=lambda x: x[1]["f1"])[0]
        print(f"\n{dataset.upper()} 结论:")
        print(f"  最高 Accuracy: {best_acc_key} ({all_results[dataset][best_acc_key]['accuracy']:.4f})")
        print(f"  最高 F1:       {best_f1_key} ({all_results[dataset][best_f1_key]['f1']:.4f})")


if __name__ == "__main__":
    main()
