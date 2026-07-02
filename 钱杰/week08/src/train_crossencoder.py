"""
CrossEncoder 训练脚本（交互型文本匹配）

使用方式：
  python train_crossencoder.py --dataset lcqmc

  # 自定义参数
  python train_crossencoder.py --dataset bq_corpus --num_hidden_layers 4 --epochs 3 --batch_size 16

依赖：
  pip install torch transformers scikit-learn tqdm
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_crossencoder_loaders
from evaluate import eval_crossencoder
from model import build_crossencoder


# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
BERT_PATH  = ROOT.parent / "models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"


# ── 训练一个 epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs, grad_accum):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [CrossEncoder]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


# ── 主训练流程 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"数据集: {args.dataset}  Epochs: {args.epochs}  Batch size: {args.batch_size}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 数据路径
    DATA_DIR = ROOT / "data" / args.dataset

    # 构建模型
    model = build_crossencoder(
        bert_path=str(BERT_PATH),
        dropout=args.dropout,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # 加载数据
    print(f"\n加载数据 from {DATA_DIR}:")
    train_loader, val_loader, test_loader = build_crossencoder_loaders(
        DATA_DIR, tokenizer, max_length=args.max_length, batch_size=args.batch_size)

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 保存配置
    save_args = {
        "dataset":           args.dataset,
        "num_hidden_layers": args.num_hidden_layers,
        "max_length":        args.max_length,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
    }
    torch.save({"args": save_args}, CKPT_DIR / "crossencoder_train_args.pt")

    # 训练
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum)

        val_metrics = eval_crossencoder(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "epoch":        epoch,
                "state_dict":   model.state_dict(),
                "val_f1":       val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "args":         save_args,
            }, CKPT_DIR / "crossencoder_best.pt")
            print(f"  [NEW BEST] F1={best_f1:.4f} -> crossencoder_best.pt")

    total_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {total_time:.1f}s  最佳 Val F1: {best_f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="CrossEncoder 训练")
    parser.add_argument("--dataset",          default="lcqmc", choices=["bq_corpus", "lcqmc"])
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--max_length",        default=128, type=int)
    parser.add_argument("--batch_size",        default=32, type=int)
    parser.add_argument("--epochs",           default=3, type=int)
    parser.add_argument("--lr",               default=2e-5, type=float)
    parser.add_argument("--weight_decay",      default=0.01, type=float)
    parser.add_argument("--dropout",           default=0.1, type=float)
    parser.add_argument("--grad_accum",       default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
