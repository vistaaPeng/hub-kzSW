"""BERT Fine-tune 训练（+ early stopping）"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from common.config import DATA_DIR, OUTPUT_DIR, CKPT_DIR, BERT_MODEL_NAME, BERT_DEFAULTS
from common.utils import get_device
from dataset import build_dataloaders
from model import build_model


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            preds = logits.argmax(dim=-1).cpu()
            labels = batch["label"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.3f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="BERT Fine-tune")
    parser.add_argument("--num_train", type=int, default=BERT_DEFAULTS["num_train"])
    parser.add_argument("--epochs", type=int, default=BERT_DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=BERT_DEFAULTS["batch_size"])
    parser.add_argument("--max_length", type=int, default=BERT_DEFAULTS["max_length"])
    parser.add_argument("--lr", type=float, default=BERT_DEFAULTS["lr"])
    parser.add_argument("--pool", type=str, default=BERT_DEFAULTS["pool"], choices=["cls", "mean", "max"])
    parser.add_argument("--patience", type=int, default=BERT_DEFAULTS["early_stopping_patience"])
    args = parser.parse_args()

    device = get_device()
    print(f"设备: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_loader, val_loader, _ = build_dataloaders(
        DATA_DIR, tokenizer, max_length=args.max_length,
        batch_size=args.batch_size, num_train=args.num_train,
        val_subset=BERT_DEFAULTS["val_subset"],
    )

    # Model
    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        num_labels = json.load(f)["num_labels"]
    model = build_model(BERT_MODEL_NAME, num_labels, pool=args.pool).to(device)

    # Optimizer: 分层学习率
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * BERT_DEFAULTS["head_lr_mult"]},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * BERT_DEFAULTS["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop + early stopping
    best_acc, no_improve = 0.0, 0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            ckpt_path = CKPT_DIR / f"best_{args.pool}.pt"
            torch.save({"epoch": epoch, "pool": args.pool, "state_dict": model.state_dict(), "val_acc": val_acc, "args": vars(args)}, ckpt_path)
            print(f"  ✓ 最优模型 → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stopping: val_acc 连续 {args.patience} epoch 未提升")
                break

    log_path = OUTPUT_DIR / "train_log_cls.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"训练完成。best_val_acc={best_acc:.4f} | 日志 → {log_path}")


if __name__ == "__main__":
    main()
