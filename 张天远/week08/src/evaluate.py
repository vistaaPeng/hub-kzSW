"""
文本匹配评估工具（可作模块导入，也可独立运行）

教学重点：
  1. BiEncoder 评估的关键差异——输出是相似度分数，需要阈值搜索才能变为预测标签
  2. 阈值搜索 (threshold search)：在验证集上枚举阈值，选 F1 最高的
  3. CrossEncoder 评估与分类任务完全一样——直接取 argmax 即为预测标签
  4. 相似度分布图——正负样本在相似度轴上的分离程度是模型质量的直观反映

使用方式：
  # 独立运行（加载 checkpoint 评估）
  python evaluate.py --model_type biencoder --ckpt ../outputs/checkpoints/biencoder_best.pt
  python evaluate.py --model_type crossencoder --ckpt ../outputs/checkpoints/crossencoder_best.pt

  # 作为模块导入（训练脚本内调用）
  from evaluate import eval_biencoder, eval_crossencoder

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
    roc_auc_score,
)
from transformers import BertTokenizer

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"
BERT_PATH  = "bert-base-chinese"  # HF model ID, auto-uses HF_HOME cache
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FIG_DIR    = OUTPUT_DIR / "figures"


# ── BiEncoder 评估 ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    """
    BiEncoder 评估：计算每对句子的余弦相似度，然后在 val 集上搜索最优阈值。

    返回 dict：
      similarities : list[float]  每对的余弦相似度
      labels       : list[int]    真实标签
      accuracy     : float        最优阈值下的准确率
      f1           : float        最优阈值下的 F1（weighted）
      threshold    : float        最优阈值（若 find_threshold=True）
      auc          : float        ROC-AUC（不依赖阈值）
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

    preds    = (sims >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_pos   = f1_score(labels, preds, pos_label=1, zero_division=0)
    f1_neg   = f1_score(labels, preds, pos_label=0, zero_division=0)

    # 相似度诊断：正负样本分离度 + 分布宽度（检测坍缩）
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_mean = float(np.mean(sims[pos_mask])) if pos_mask.any() else float('nan')
    neg_mean = float(np.mean(sims[neg_mask])) if neg_mask.any() else float('nan')
    sim_gap  = pos_mean - neg_mean
    sim_std  = float(np.std(sims))

    # AUC：若 labels 只有一类（如 AFQMC test 全为 0），跳过
    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:
        auc = float("nan")

    return {
        "similarities": all_sims,
        "labels":       all_labels,
        "accuracy":     accuracy,
        "f1":           f1,
        "f1_pos":       f1_pos,
        "f1_neg":       f1_neg,
        "threshold":    threshold,
        "pos_mean_sim": pos_mean,
        "neg_mean_sim": neg_mean,
        "sim_gap":      sim_gap,
        "sim_std":      sim_std,
        "auc":          auc,
    }


def _find_best_threshold(sims, labels):
    """枚举 [0.0, 1.0] 区间 101 个候选阈值，返回使 weighted-F1 最高的那个。"""
    best_f1, best_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1    = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t
    return float(best_thresh)


# ── CrossEncoder 评估 ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_crossencoder(model, loader, device):
    """
    CrossEncoder 评估：与分类任务完全一致，取 argmax 得预测标签。

    返回 dict：
      accuracy : float
      f1       : float（weighted）
      logits   : list[list[float]]
      labels   : list[int]
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"]

        logits = model(input_ids, attention_mask, token_type_ids).cpu()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.tolist())

    preds    = np.argmax(all_logits, axis=1)
    labels   = np.array(all_labels)
    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "logits":   all_logits,
        "labels":   all_labels,
        "accuracy": accuracy,
        "f1":       f1,
    }


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_similarity_distribution(sims, labels, threshold, save_path, title="Similarity Distribution"):
    """
    Plot cosine similarity distribution histogram for positive/negative samples (BiEncoder only).

    Teaching value:
      - Ideal case: positive samples clustered near 1, negative near -1
      - Less overlap = stronger model discrimination
    """
    sims   = np.array(sims)
    labels = np.array(labels)

    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_sims, bins=50, alpha=0.6, label=f"Positive (n={len(pos_sims)})",
            color="#2196F3", density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, label=f"Negative (n={len(neg_sims)})",
            color="#F44336", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 独立运行入口 ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="文本匹配模型评估")
    parser.add_argument("--model_type", required=True, choices=["biencoder", "crossencoder"])
    parser.add_argument("--ckpt",       required=True, type=str, help="checkpoint .pt 文件路径")
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=str)
    parser.add_argument("--bert_path",  default=str(BERT_PATH), type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=64,  type=int, help="BiEncoder 单句最大长度")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"],
                        help="评估的数据集分割（AFQMC test 无正样本，建议用 validation）")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"加载 checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"训练信息: {ckpt.get('args', {})}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.model_type == "biencoder":
        from model import build_biencoder
        from dataset import PairDataset
        from torch.utils.data import DataLoader

        saved_args = ckpt.get("args", {})
        model = build_biencoder(
            bert_path=args.bert_path,
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds     = PairDataset(data_path, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        metrics = eval_biencoder(model, loader, device)
        print(f"\n{'='*50}")
        print(f"BiEncoder 评估结果（{args.split}，{len(ds)} 条）")
        print(f"  最优阈值: {metrics['threshold']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")
        print(f"  F1(pos) : {metrics['f1_pos']:.4f}   F1(neg) : {metrics['f1_neg']:.4f}")
        print(f"  相似度  : pos_mean={metrics['pos_mean_sim']:.3f}  neg_mean={metrics['neg_mean_sim']:.3f}  gap={metrics['sim_gap']:.3f}  std={metrics['sim_std']:.3f}")
        print(f"  AUC     : {metrics['auc']:.4f}")

        plot_similarity_distribution(
            metrics["similarities"], metrics["labels"], metrics["threshold"],
            save_path=FIG_DIR / f"biencoder_{args.split}_sim_dist.png",
            title=f"BiEncoder Similarity Distribution ({args.split})",
        )

        # 打印分类报告
        preds = (np.array(metrics["similarities"]) >= metrics["threshold"]).astype(int)
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")

    elif args.model_type == "crossencoder":
        from model import build_crossencoder
        from dataset import CrossEncoderDataset
        from torch.utils.data import DataLoader

        saved_args = ckpt.get("args", {})
        model = build_crossencoder(
            bert_path=args.bert_path,
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        metrics = eval_crossencoder(model, loader, device)
        print(f"\n{'='*50}")
        print(f"CrossEncoder 评估结果（{args.split}，{len(ds)} 条）")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")

        preds = np.argmax(metrics["logits"], axis=1)
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")


if __name__ == "__main__":
    main()
