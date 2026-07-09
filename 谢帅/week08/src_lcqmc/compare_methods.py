"""
多方法效果对比脚本（LCQMC）

对比四种文本匹配方式在 LCQMC validation/test 集上的效果：
  1. BiEncoder + CosineEmbeddingLoss
  2. BiEncoder + TripletLoss
  3. CrossEncoder + CrossEntropyLoss
  4. LLM SFT (Qwen2.5-0.5B LoRA)

教学重点：
  1. 三种 BERT 方式都用 12 层 BERT，3 epoch 训练，控制变量对比 Loss 函数差异
  2. BiEncoder 需阈值搜索，CrossEncoder 直接 argmax——评估方式本身就是知识点
  3. 速度/精度权衡：CrossEncoder 度最高但推理慢，BiEncoder 可向量化
  4. LLM SFT 作为第四种方法，展示微调大语言模型的效果
  5. 输出对比图：正负样本相似度分布（BiEncoder × 2）+ 混淆矩阵（CrossEncoder）

使用方式：
  python compare_methods.py
  python compare_methods.py --split validation --batch_size 64

依赖：
  pip install torch transformers scikit-learn matplotlib
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
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder, plot_similarity_distribution
from model import build_biencoder, build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
DATA_DIR   = ROOT.parent / "data" / "lcqmc"
BERT_PATH  = Path("E:/my/LLM/课程/pretrain_models/bert-base-chinese")
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
FIG_DIR    = ROOT / "outputs" / "figures"
LOG_DIR    = ROOT / "outputs" / "logs"
SFT_DIR    = ROOT / "outputs" / "sft_adapter"

METHODS = [
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
    {
        "key":       "llm_sft",
        "label":     "LLM SFT\n(Qwen2.5-0.5B LoRA)",
        "type":      "sft",
        "color":     "#9C27B0",
    },
]


# ── 加载并评估单个方法 ─────────────────────────────────────────────────────

def load_and_eval(method, tokenizer, device, split, batch_size, bert_path=None, data_dir=None):
    bert_path = bert_path or str(BERT_PATH)
    data_dir  = data_dir or str(DATA_DIR)
    ckpt_path = CKPT_DIR / method["ckpt"]
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved     = ckpt.get("args", {})

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=bert_path,
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = Path(data_dir) / f"{split}.jsonl"
        ds     = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)

    else:  # crossencoder
        model = build_crossencoder(
            bert_path=bert_path,
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = Path(data_dir) / f"{split}.jsonl"
        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["model"] = model
    metrics["ckpt"]  = ckpt
    return metrics


def load_sft_result(split="validation"):
    """
    加载 LLM SFT 评估结果（从 sft_results.json）。

    SFT 模型不输出相似度分数，直接输出分类标签，
    因此使用 f1_weighted 作为主要 F1 指标。
    """
    result_path = LOG_DIR / "sft_results.json"
    if not result_path.exists():
        print(f"  [SKIP] SFT 结果文件不存在: {result_path}")
        return None

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    # 查找对应 split 的结果
    if isinstance(data, dict):
        # 如果按 split 组织
        result = data.get(split, data)
    elif isinstance(data, list):
        # 如果是列表，取第一个
        result = data[0] if data else None
    else:
        result = data

    if result is None:
        print(f"  [SKIP] SFT 结果中无 {split} 数据")
        return None

    return {
        "f1":           result.get("f1_weighted", result.get("f1", 0.0)),
        "accuracy":     result.get("accuracy", 0.0),
        "type":         "sft",
    }


# ── 对比可视化 ────────────────────────────────────────────────────────────

def plot_comparison_bar(results, save_path):
    """准确率 / F1 对比柱状图"""
    names    = [m["label"]    for m in results]
    accs     = [m["accuracy"] for m in results]
    f1s      = [m["f1"]       for m in results]
    colors   = [m["color"]    for m in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color=colors, alpha=0.5,
                   hatch="//", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison on LCQMC (12-layer BERT, 3 epoch)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_sim_distributions(biencoder_results, save_path):
    """所有 BiEncoder 方法的相似度分布叠放对比"""
    n = len(biencoder_results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, biencoder_results):
        sims   = np.array(m["similarities"])
        labels = np.array(m["labels"])
        ax.hist(sims[labels==1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)
        ax.hist(sims[labels==0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)
        ax.axvline(m["threshold"], color="black", linestyle="--",
                   label=f"threshold={m['threshold']:.2f}")
        ax.set_title(m["label"].replace("\n", " "))
        ax.set_xlabel("Cosine Similarity")
        ax.legend(fontsize=8)

    fig.suptitle("BiEncoder Similarity Distribution (positive vs negative)", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LCQMC 四种文本匹配方法效果对比")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--bert_path",  default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=str)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  评估集: {args.split}")

    bert_path = args.bert_path
    data_dir  = args.data_dir
    tokenizer = BertTokenizer.from_pretrained(str(bert_path))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 逐方法评估 ────────────────────────────────────────────────────────
    all_results = []
    for m in METHODS:
        print(f"\n{'='*55}")
        print(f"加载 {m['key']} ...")

        if m["type"] == "sft":
            sft_result = load_sft_result(split=args.split)
            if sft_result is None:
                continue
            sft_result.update({
                "label": m["label"],
                "color": m["color"],
                "key":   m["key"],
                "type":  m["type"],
            })
            all_results.append(sft_result)
            continue

        metrics = load_and_eval(m, tokenizer, device, args.split, args.batch_size, bert_path=bert_path, data_dir=data_dir)
        if metrics is None:
            continue
        metrics.update({"label": m["label"], "color": m["color"], "key": m["key"],
                        "type": m["type"]})
        all_results.append(metrics)

    if not all_results:
        print("没有可用的 checkpoint 或结果，请先运行训练脚本。")
        return

    # ── 控制台对比表 ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'方法':<30} {'Accuracy':>9} {'F1(weighted)':>13} {'额外信息':>15}")
    print(f"{'-'*65}")
    for m in all_results:
        extra = (f"threshold={m['threshold']:.2f}"
                 if m["type"] == "biencoder" else
                 "argmax" if m["type"] == "crossencoder" else
                 "SFT direct")
        print(f"  {m['key']:<28} {m['accuracy']:>9.4f} {m['f1']:>13.4f} {extra:>15}")

    print(f"\n{'─'*65}")
    print("结论速览：")
    best_acc = max(all_results, key=lambda x: x["accuracy"])
    best_f1  = max(all_results, key=lambda x: x["f1"])
    print(f"  最高 Accuracy : {best_acc['key']} ({best_acc['accuracy']:.4f})")
    print(f"  最高 F1       : {best_f1['key']}  ({best_f1['f1']:.4f})")

    # 列出不同 Loss 的 BiEncoder 对比
    bi_results = [m for m in all_results if m["type"] == "biencoder"]
    if len(bi_results) == 2:
        a, b = bi_results
        delta_acc = b["accuracy"] - a["accuracy"]
        delta_f1  = b["f1"] - a["f1"]
        print(f"\n  Cosine vs Triplet (Δ):")
        print(f"    Accuracy: {delta_acc:+.4f}  F1: {delta_f1:+.4f}")
        if abs(delta_f1) < 0.01:
            print("    → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）")
        elif delta_f1 > 0:
            print("    → TripletLoss 更优，三元组对语义距离的约束更精确")
        else:
            print("    → CosineEmbeddingLoss 更优，LCQMC 数据量下直接对标签优化更稳定")

    # ── 保存对比日志 ──────────────────────────────────────────────────────
    SKIP_KEYS = {"model", "similarities", "labels", "logits", "ckpt"}

    def _to_py(v):
        if hasattr(v, "item"):  # numpy scalar / torch scalar
            return v.item()
        return v

    log = [{k: _to_py(v) for k, v in m.items() if k not in SKIP_KEYS}
           for m in all_results]
    log_path = LOG_DIR / "method_comparison.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志 → {log_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    plot_comparison_bar(all_results, FIG_DIR / "method_comparison_bar.png")

    bi_with_sim = [m for m in all_results if m["type"] == "biencoder" and "similarities" in m]
    if bi_with_sim:
        plot_sim_distributions(bi_with_sim, FIG_DIR / "biencoder_sim_distributions.png")


if __name__ == "__main__":
    main()
