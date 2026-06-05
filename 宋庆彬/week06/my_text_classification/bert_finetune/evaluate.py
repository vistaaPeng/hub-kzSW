"""BERT 模型评估 + 混淆矩阵"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer

from common.config import DATA_DIR, CKPT_DIR, FIG_DIR, BERT_MODEL_NAME
from common.metrics import compute_metrics
from common.utils import get_device, setup_plot_font
from dataset import build_dataloaders
from model import build_model


def plot_confusion_matrix(cm, id2name, save_path):
    setup_plot_font()
    class_names = [id2name[i] for i in range(len(id2name))]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.heatmap(cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 7})
    axes[0].set_title("混淆矩阵（绝对计数）")
    axes[0].set_xlabel("预测"); axes[0].set_ylabel("真实")
    axes[0].tick_params(axis="x", rotation=40)

    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 7}, vmin=0, vmax=1)
    axes[1].set_title("混淆矩阵（按行归一化）")
    axes[1].set_xlabel("预测"); axes[1].set_ylabel("真实")
    axes[1].tick_params(axis="x", rotation=40)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵 → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"设备: {device}")

    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    _, val_loader, _ = build_dataloaders(DATA_DIR, tokenizer, max_length=args.max_length, batch_size=args.batch_size)

    model = build_model(BERT_MODEL_NAME, num_labels, pool=args.pool)
    ckpt = torch.load(CKPT_DIR / f"best_{args.pool}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    print(f"加载 checkpoint: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), id2name)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}  |  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"\n分类报告:\n{metrics['report']}")

    plot_confusion_matrix(metrics["confusion_matrix"], id2name,
                          FIG_DIR / f"confusion_matrix_{args.pool}.png")


if __name__ == "__main__":
    main()
