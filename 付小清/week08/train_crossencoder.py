"""
在 LCQMC / BQ Corpus 上训练 CrossEncoder

使用方式：
  cd work8
  python train_crossencoder.py --dataset bq_corpus
  python train_crossencoder.py --dataset lcqmc --epochs 3
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

from dataset import build_crossencoder_loaders, resolve_data_dir
from evaluate import eval_crossencoder, ckpt_dir, log_dir
from model import build_crossencoder

PROJECT_ROOT = Path(__file__).parent.parent
BERT_PATH = PROJECT_ROOT.parent / "pretrain_models" / "bert-base-chinese"


def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs, grad_accum):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [CrossEncoder]", leave=False)
    for step, batch in enumerate(pbar):
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        )
        labels = batch["label"].to(device)
        loss = criterion(logits, labels)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


def parse_args():
    parser = argparse.ArgumentParser(description="work8 CrossEncoder 训练")
    parser.add_argument("--dataset", required=True, choices=["lcqmc", "bq_corpus"])
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--head_lr_mult", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.dataset)
    out_ckpt = ckpt_dir(args.dataset)
    out_log = log_dir(args.dataset)
    out_ckpt.mkdir(parents=True, exist_ok=True)
    out_log.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"数据集: {args.dataset}  层数: {args.num_hidden_layers}  Epochs: {args.epochs}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")
    train_loader, val_loader, _ = build_crossencoder_loaders(
        data_dir, tokenizer, args.max_length, args.batch_size,
    )

    print("\n构建模型...")
    model = build_crossencoder(
        bert_path=args.bert_path,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": args.lr},
        {"params": list(model.dropout.parameters()) + list(model.classifier.parameters()),
         "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"总训练步数: {total_steps}  Warmup: {warmup_steps}")

    criterion = nn.CrossEntropyLoss()
    ckpt_path = out_ckpt / "crossencoder_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum,
        )
        val_metrics = eval_crossencoder(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} | {elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_f1_pos": val_metrics["f1_pos"],
            "elapsed_s": elapsed,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "args": vars(args),
            }, ckpt_path)
            print(f"  [OK] 新最优模型 -> {ckpt_path}  (val_f1={val_metrics['f1']:.4f})")

    log_path = out_log / "train_crossencoder.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"日志 -> {log_path}")
    print(f"评估：python evaluate.py --dataset {args.dataset} --model_type crossencoder --split test")


if __name__ == "__main__":
    main()
