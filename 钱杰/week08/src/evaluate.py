"""
文本匹配评估工具

教学重点：
  1. BiEncoder 评估：输出是相似度分数，需要阈值搜索才能变为预测标签
  2. 阈值搜索：在验证集上枚举阈值，选 F1 最高的
  3. CrossEncoder 评估与分类任务完全一样——直接取 argmax 即为预测标签

使用方式：
  python evaluate.py --model_type biencoder --ckpt ../outputs/checkpoints/biencoder_best.pt
  python evaluate.py --model_type crossencoder --ckpt ../outputs/checkpoints/crossencoder_best.pt

依赖：
  pip install torch transformers scikit-learn matplotlib
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import BertTokenizer


# ── 阈值搜索 ──────────────────────────────────────────────────────────────

def _find_best_threshold(sims, labels):
    """枚举阈值，找 weighted F1 最高的"""
    best_th, best_f1 = 0.0, 0.0
    for th in np.arange(-1.0, 1.01, 0.01):
        preds = (sims >= th).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th


# ── BiEncoder 评估 ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    """
    BiEncoder 评估：计算余弦相似度，搜索最优阈值

    返回 dict：
      similarities : list[float]  每对的余弦相似度
      labels       : list[int]    真实标签
      accuracy     : float        最优阈值下的准确率
      f1           : float        最优阈值下的 F1（weighted）
      precision    : float
      recall       : float
      threshold    : float        最优阈值
      auc          : float        ROC-AUC
    """
    model.eval()
    all_sims, all_labels = [], []

    for batch in loader:
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        all_sims.extend(sims)
        all_labels.extend(batch["label"].tolist())

    sims   = np.array(all_sims)
    labels = np.array(all_labels)

    if find_threshold:
        threshold = _find_best_threshold(sims, labels)

    preds     = (sims >= threshold).astype(int)
    accuracy  = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:
        auc = float("nan")

    return {
        "similarities": all_sims,
        "labels":       all_labels,
        "accuracy":     accuracy,
        "f1":           f1,
        "precision":    precision,
        "recall":       recall,
        "threshold":    float(threshold),
        "auc":          auc,
    }


# ── CrossEncoder 评估 ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_crossencoder(model, loader, device):
    """CrossEncoder 评估：直接 argmax 获取预测"""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, probs[:, 1].cpu().numpy())
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":  accuracy,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "auc":       auc,
        "predictions": all_preds,
        "labels":    all_labels,
    }


# ── 相似度分布可视化 ──────────────────────────────────────────────────────

@torch.no_grad()
def plot_similarity_distribution(model, loader, device, save_path, title="Similarity Distribution"):
    """绘制正负样本的相似度分布直方图"""
    model.eval()
    pos_sims, neg_sims = [], []

    for batch in loader:
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims   = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu()
        labels = batch["label"]

        pos_sims.extend(sims[labels == 1].tolist())
        neg_sims.extend(sims[labels == 0].tolist())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_sims, bins=50, alpha=0.6, label="positive", color="#2196F3", density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, label="negative", color="#F44336", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 主函数（独立运行评估）─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["biencoder", "crossencoder"], required=True)
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--split",      default="validation", choices=["validation", "test"])
    args = parser.parse_args()

    from dataset import PairDataset, CrossEncoderDataset
    from model import build_biencoder, build_crossencoder

    ROOT      = Path(__file__).parent.parent
    BERT_PATH = ROOT.parent / "models" / "bert-base-chinese"
    DATA_DIR  = Path(args.data_dir) if args.data_dir else ROOT / "data" / "lcqmc"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    if args.model_type == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds = PairDataset(DATA_DIR / f"{args.split}.jsonl", tokenizer,
                         max_length=saved_args.get("max_length", 64))
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        metrics = eval_biencoder(model, loader, device)
        print(f"\nBiEncoder {args.split} 结果:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.2f}")
    else:
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds = CrossEncoderDataset(DATA_DIR / f"{args.split}.jsonl", tokenizer,
                                  max_length=128)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        metrics = eval_crossencoder(model, loader, device)
        print(f"\nCrossEncoder {args.split} 结果:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
