"""
多方法效果对比脚本

对比七种文本匹配方式在 LCQMC validation 集上的效果：
  1. 编辑距离（Edit Distance）— 基于字符序列的相似度
  2. 词向量（Word Vector）— 基于词嵌入的语义相似度
  3. TF-IDF — 基于词频统计的相似度
  4. BM25 — 改进的TF-IDF，考虑文档长度归一化
  5. BiEncoder + CosineEmbeddingLoss — 表示型双塔模型
  6. BiEncoder + TripletLoss — 对比学习
  7. CrossEncoder + CrossEntropyLoss — 交互型模型

教学重点：
  1. 传统方法 vs 深度学习方法的对比
  2. 不同传统方法的特点和适用场景
  3. BiEncoder vs CrossEncoder的精度/速度权衡
  4. CosineEmbeddingLoss vs TripletLoss（对比学习）的差异
  5. BiEncoder需要阈值搜索，CrossEncoder直接argmax

使用方式：
  python compare_methods.py
  python compare_methods.py --split validation --batch_size 64

依赖：
  pip install torch transformers scikit-learn matplotlib jieba gensim
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset, TraditionalDataset
from evaluate import eval_biencoder, eval_crossencoder, eval_traditional
from model import (
    build_edit_distance_model,
    build_word_vector_model,
    build_tfidf_model,
    build_bm25_model,
)
from biencoder import build_biencoder
from crossencoder import build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "lcqmc"
BERT_PATH  = ROOT.parent.parent.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
FIG_DIR    = ROOT / "outputs" / "figures"
LOG_DIR    = ROOT / "outputs" / "logs"

# ── 方法定义 ──────────────────────────────────────────────────────────────

METHODS = [
    {
        "key":       "edit_distance",
        "label":     "编辑距离\n(Edit Distance)",
        "type":      "traditional",
        "color":     "#9E9E9E",
        "build":     lambda: build_edit_distance_model(),
    },
    {
        "key":       "word_vector",
        "label":     "词向量\n(Word Vector)",
        "type":      "traditional",
        "color":     "#795548",
        "build":     lambda: build_word_vector_model(),
    },
    {
        "key":       "tfidf",
        "label":     "TF-IDF",
        "type":      "traditional",
        "color":     "#607D8B",
        "build":     lambda sentences: build_tfidf_model(sentences),
    },
    {
        "key":       "bm25",
        "label":     "BM25",
        "type":      "traditional",
        "color":     "#455A64",
        "build":     lambda sentences: build_bm25_model(sentences),
    },
    {
        "key":       "biencoder_cosine",
        "label":     "BiEncoder\n(CosineEmbeddingLoss)",
        "ckpt":      "biencoder_cosine_best.pt",
        "type":      "biencoder",
        "color":     "#2196F3",
    },
    {
        "key":       "biencoder_triplet",
        "label":     "BiEncoder\n(TripletLoss)",
        "ckpt":      "biencoder_triplet_best.pt",
        "type":      "biencoder",
        "color":     "#4CAF50",
    },
    {
        "key":       "crossencoder",
        "label":     "CrossEncoder\n(CrossEntropyLoss)",
        "ckpt":      "crossencoder_best.pt",
        "type":      "crossencoder",
        "color":     "#FF9800",
    },
]


# ── 加载并评估单个方法 ─────────────────────────────────────────────────────

def load_and_eval(method, tokenizer, device, split, batch_size, train_sentences=None):
    """加载并评估单个方法"""

    # ── 传统方法 ────────────────────────────────────────────────────────
    if method["type"] == "traditional":
        if method["key"] in ["tfidf", "bm25"]:
            model = method["build"](train_sentences)
        else:
            model = method["build"]()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = TraditionalDataset(data_path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_traditional(model, loader)

    # ── BiEncoder ────────────────────────────────────────────────────────
    elif method["type"] == "biencoder":
        ckpt_path = CKPT_DIR / method["ckpt"]
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
            return None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ckpt.get("args", {})

        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)

    # ── CrossEncoder ──────────────────────────────────────────────────────
    elif method["type"] == "crossencoder":
        ckpt_path = CKPT_DIR / method["ckpt"]
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
            return None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ckpt.get("args", {})

        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["model"] = model
    metrics["ckpt"] = ckpt if method["type"] != "traditional" else None
    return metrics


# ── 对比可视化 ────────────────────────────────────────────────────────────

def plot_comparison_bar(results, save_path):
    """准确率 / F1 对比柱状图"""
    names    = [m["label"]    for m in results]
    accs     = [m["accuracy"] for m in results]
    f1s      = [m["f1"]       for m in results]
    colors   = [m["color"]    for m in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color=colors, alpha=0.5,
                   hatch="//", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison on LCQMC Validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_sim_distributions(results_with_sim, save_path):
    """所有方法的相似度/概率分布叠放对比"""
    n = len(results_with_sim)
    if n == 0:
        return

    # 只保留有相似度数据的结果
    results_with_sim = [m for m in results_with_sim if "similarities" in m]
    n = len(results_with_sim)
    if n == 0:
        print("  跳过相似度分布图绘制（没有可用的相似度数据）")
        return

    # 分组显示
    fig, axes = plt.subplots(2, min(n, 4), figsize=(16, 8))
    if n <= 4:
        axes = axes.reshape(2, n)

    for idx, m in enumerate(results_with_sim[:min(n, 4)]):
        # 第一行：相似度分布
        ax = axes[0, idx] if n > 1 else axes[0]
        sims = np.array(m["similarities"])
        labels = np.array(m["labels"])
        ax.hist(sims[labels==1], bins=40, alpha=0.6, label="positive",
                color="#2196F3", density=True)
        ax.hist(sims[labels==0], bins=40, alpha=0.6, label="negative",
                color="#F44336", density=True)
        if "threshold" in m:
            ax.axvline(m["threshold"], color="black", linestyle="--",
                       label=f"threshold={m['threshold']:.2f}")
        ax.set_title(m["label"].replace("\n", " "), fontsize=9)
        ax.set_xlabel("Similarity")
        ax.legend(fontsize=8)

        # 第二行：预测分布
        ax2 = axes[1, idx] if n > 1 else axes[1]
        if "threshold" in m:
            preds = (sims >= m["threshold"]).astype(int)
        else:
            preds = np.argmax(m["logits"], axis=1) if "logits" in m else np.zeros(len(labels))

        correct = preds == labels
        ax2.bar(["正确", "错误"],
                [np.sum(correct), np.sum(~correct)],
                color=["#4CAF50", "#F44336"])
        ax2.set_title(f"预测结果 (Acc={m['accuracy']:.3f})", fontsize=9)
        ax2.set_ylabel("数量")

    fig.suptitle("Similarity Distribution & Prediction Results", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="七种文本匹配方法效果对比")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--skip_traditional", action="store_true",
                        help="跳过传统方法评估（速度较快）")
    parser.add_argument("--skip_deep", action="store_true",
                        help="跳过深度学习方法评估（需要checkpoint）")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  评估集: {args.split}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 只在需要深度学习方法时加载 tokenizer
    tokenizer = None
    if not args.skip_deep:
        tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # ── 加载训练集句子（用于TF-IDF/BM25训练）────────────────────────────
    train_sentences = []
    if not args.skip_traditional:
        print("\n加载训练集句子（用于TF-IDF/BM25训练）...")
        train_path = DATA_DIR / "train.jsonl"
        train_ds = TraditionalDataset(train_path)
        for batch in DataLoader(train_ds, batch_size=100, shuffle=False):
            train_sentences.extend(batch["sentence1"])
            train_sentences.extend(batch["sentence2"])
        print(f"  训练集句子数: {len(train_sentences):,}")

    # ── 逐方法评估 ────────────────────────────────────────────────────────
    all_results = []
    for m in METHODS:
        # 跳过逻辑
        if args.skip_traditional and m["type"] == "traditional":
            continue
        if args.skip_deep and m["type"] != "traditional":
            continue

        print(f"\n{'='*55}")
        print(f"评估 {m['key']} ...")

        metrics = load_and_eval(m, tokenizer, device, args.split, args.batch_size, train_sentences)
        if metrics is None:
            continue

        metrics.update({"label": m["label"], "color": m["color"], "key": m["key"],
                        "type": m["type"]})
        all_results.append(metrics)

    if not all_results:
        print("没有可用的评估结果。")
        return

    # ── 控制台对比表 ──────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"{'方法':<35} {'Accuracy':>9} {'F1(weighted)':>13} {'额外信息':>15}")
    print(f"{'-'*75}")
    for m in all_results:
        extra = (f"threshold={m['threshold']:.2f}"
                 if "threshold" in m else "argmax")
        print(f"  {m['key']:<33} {m['accuracy']:>9.4f} {m['f1']:>13.4f} {extra:>15}")

    print(f"\n{'─'*75}")
    print("结论速览：")
    best_acc = max(all_results, key=lambda x: x["accuracy"])
    best_f1  = max(all_results, key=lambda x: x["f1"])
    print(f"  最高 Accuracy : {best_acc['key']} ({best_acc['accuracy']:.4f})")
    print(f"  最高 F1       : {best_f1['key']}  ({best_f1['f1']:.4f})")

    # ── 传统方法 vs 深度学习方法对比 ──────────────────────────────────────
    traditional_results = [m for m in all_results if m["type"] == "traditional"]
    deep_results = [m for m in all_results if m["type"] != "traditional"]

    if traditional_results and deep_results:
        avg_trad_acc = np.mean([m["accuracy"] for m in traditional_results])
        avg_trad_f1  = np.mean([m["f1"] for m in traditional_results])
        avg_deep_acc = np.mean([m["accuracy"] for m in deep_results])
        avg_deep_f1  = np.mean([m["f1"] for m in deep_results])

        print(f"\n传统方法平均：Accuracy={avg_trad_acc:.4f}  F1={avg_trad_f1:.4f}")
        print(f"深度学习方法平均：Accuracy={avg_deep_acc:.4f}  F1={avg_deep_f1:.4f}")
        print(f"深度学习方法提升：Accuracy +{(avg_deep_acc - avg_trad_acc):.4f}  F1 +{(avg_deep_f1 - avg_trad_f1):.4f}")

    # ── BiEncoder vs CrossEncoder对比 ──────────────────────────────────────
    bi_results = [m for m in all_results if m["type"] == "biencoder"]
    cross_results = [m for m in all_results if m["type"] == "crossencoder"]

    if bi_results and cross_results:
        avg_bi_acc = np.mean([m["accuracy"] for m in bi_results])
        avg_bi_f1  = np.mean([m["f1"] for m in bi_results])
        avg_cross_acc = np.mean([m["accuracy"] for m in cross_results])
        avg_cross_f1  = np.mean([m["f1"] for m in cross_results])

        print(f"\nBiEncoder平均：Accuracy={avg_bi_acc:.4f}  F1={avg_bi_f1:.4f}")
        print(f"CrossEncoder平均：Accuracy={avg_cross_acc:.4f}  F1={avg_cross_f1:.4f}")
        print(f"CrossEncoder优势：Accuracy +{(avg_cross_acc - avg_bi_acc):.4f}  F1 +{(avg_cross_f1 - avg_bi_f1):.4f}")

    # ── CosineEmbeddingLoss vs TripletLoss对比 ──────────────────────────────
    if len(bi_results) == 2:
        a, b = bi_results
        delta_acc = b["accuracy"] - a["accuracy"]
        delta_f1  = b["f1"] - a["f1"]
        print(f"\n  Cosine vs Triplet (Δ):")
        print(f"    Accuracy: {delta_acc:+.4f}  F1: {delta_f1:+.4f}")
        if abs(delta_f1) < 0.01:
            print("    → 两种 Loss 差距不大")
        elif delta_f1 > 0:
            print("    → TripletLoss 更优，对比学习能更好地学习语义距离")
        else:
            print("    → CosineEmbeddingLoss 更优，直接对标签优化更稳定")

    # ── 保存对比日志 ──────────────────────────────────────────────────────
    SKIP_KEYS = {"model", "similarities", "labels", "logits", "ckpt"}

    def _to_py(v):
        if hasattr(v, "item"):
            return v.item()
        return v

    log = [{k: _to_py(v) for k, v in m.items() if k not in SKIP_KEYS}
           for m in all_results]
    log_path = LOG_DIR / "method_comparison.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志 → {log_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    plot_comparison_bar(all_results, FIG_DIR / "method_comparison_bar.png")

    results_with_sim = [m for m in all_results if "similarities" in m or "logits" in m]
    if results_with_sim:
        plot_sim_distributions(results_with_sim, FIG_DIR / "similarity_distributions.png")


if __name__ == "__main__":
    main()