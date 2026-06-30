"""
BiEncoder 训练脚本（表示型文本匹配，Sentence-BERT 架构）

支持两种 Loss：
  1. CosineEmbeddingLoss — 直接优化余弦相似度
  2. TripletLoss — 约束三角关系

使用方式：
  # CosineEmbeddingLoss（默认）
  python train_biencoder.py --dataset lcqmc

  # TripletLoss
  python train_biencoder.py --dataset lcqmc --loss triplet

  # 自定义参数
  python train_biencoder.py --dataset bq_corpus --loss cosine --pool mean --num_hidden_layers 4 --epochs 3

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
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_pair_loaders, build_triplet_loader
from evaluate import eval_biencoder
from model import build_biencoder


# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
BERT_PATH  = ROOT.parent / "models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"


# ── 训练一个 epoch ────────────────────────────────────────────────────────

def train_one_epoch_cosine(model, loader, optimizer, scheduler, device,
                           epoch, total_epochs, margin, grad_accum):
    """CosineEmbeddingLoss 训练 loop"""
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Cosine]", leave=False)
    for step, batch in enumerate(pbar):
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
        labels = batch["label"].to(device)

        emb_a, emb_b = model(batch_a, batch_b)

        # label 0→-1，1→+1（cosine_embedding_loss 要求 target ∈ {-1, +1}）
        cos_target = (labels.float() * 2 - 1)
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=margin)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def train_one_epoch_triplet(model, loader, optimizer, scheduler, device,
                            epoch, total_epochs, margin, grad_accum):
    """TripletLoss 训练 loop"""
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Triplet]", leave=False)
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_p = {
            "input_ids":      batch["input_ids_p"].to(device),
            "attention_mask": batch["attention_mask_p"].to(device),
            "token_type_ids": batch["token_type_ids_p"].to(device),
        }
        batch_n = {
            "input_ids":      batch["input_ids_n"].to(device),
            "attention_mask": batch["attention_mask_n"].to(device),
            "token_type_ids": batch["token_type_ids_n"].to(device),
        }

        emb_a, emb_p = model(batch_a, batch_p)
        _,  emb_n    = model(batch_a, batch_n)  # 复用 batch_a 的编码

        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=margin)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss    += loss.item()
        total_samples += 1
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


# ── 主训练流程 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"数据集: {args.dataset}  Loss: {args.loss}  Epochs: {args.epochs}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 数据路径
    DATA_DIR = ROOT / "data" / args.dataset

    # 构建模型
    model = build_biencoder(
        bert_path=str(BERT_PATH),
        pool=args.pool,
        dropout=args.dropout,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # 加载数据
    print(f"\n加载数据 from {DATA_DIR}:")
    if args.loss == "triplet":
        train_loader, val_loader, test_loader = build_triplet_loader(
            DATA_DIR, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
        train_fn = train_one_epoch_triplet
    else:
        train_loader, val_loader, test_loader = build_pair_loaders(
            DATA_DIR, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
        train_fn = train_one_epoch_cosine

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 保存配置
    save_args = {
        "dataset":         args.dataset,
        "loss":            args.loss,
        "pool":            args.pool,
        "num_hidden_layers": args.num_hidden_layers,
        "max_length":      args.max_length,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
    }
    torch.save({"args": save_args}, CKPT_DIR / "train_args.pt")

    # 训练
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.margin, args.grad_accum)

        val_metrics = eval_biencoder(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1']:.4f}  "
              f"Threshold: {val_metrics['threshold']:.2f}")

        # 保存最佳模型
        ckpt_name = f"biencoder_{args.loss}_best.pt"
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "epoch":          epoch,
                "state_dict":     model.state_dict(),
                "val_f1":         val_metrics["f1"],
                "val_accuracy":   val_metrics["accuracy"],
                "args":           save_args,
            }, CKPT_DIR / ckpt_name)
            print(f"  [NEW BEST] F1={best_f1:.4f} -> {ckpt_name}")

    total_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {total_time:.1f}s  最佳 Val F1: {best_f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="BiEncoder 训练")
    parser.add_argument("--dataset",          default="lcqmc", choices=["bq_corpus", "lcqmc"])
    parser.add_argument("--loss",             default="cosine", choices=["cosine", "triplet"])
    parser.add_argument("--pool",             default="mean", choices=["cls", "mean", "max"])
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--max_length",       default=64, type=int)
    parser.add_argument("--batch_size",       default=32, type=int)
    parser.add_argument("--epochs",          default=3, type=int)
    parser.add_argument("--lr",               default=2e-5, type=float)
    parser.add_argument("--weight_decay",     default=0.01, type=float)
    parser.add_argument("--margin",           default=0.3, type=float)
    parser.add_argument("--grad_accum",       default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
