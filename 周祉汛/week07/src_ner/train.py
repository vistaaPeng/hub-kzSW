"""
BERT NER 训练脚本，支持分层学习率、梯度累积、自动保存最优模型。
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from evaluate import evaluate_entity

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "ner"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs_ner"
CKPT_DIR = OUTPUT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="BERT NER 训练")
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=str)
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--lr", default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", default=5.0, type=float,
                        help="分类头学习率倍数")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)

        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 处理剩余梯度
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(loader)
    return avg_loss


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载标签映射
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = label_map["label2id"]

    # Tokenizer & DataLoader
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader = build_dataloaders(
        data_dir, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # 模型
    model = build_model(args.bert_path, num_labels,
                        use_crf=args.use_crf,
                        dropout=args.dropout)
    model = model.to(device)

    # 分层学习率
    bert_params = list(model.bert.parameters())
    classifier_params = list(model.classifier.parameters())
    if args.use_crf:
        crf_params = list(model.crf.parameters())
        head_params = classifier_params + crf_params
    else:
        head_params = classifier_params

    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"总训练步数: {total_steps}, warmup: {warmup_steps}")

    # 训练循环
    best_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_f1, _ = evaluate_entity(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_entity_f1={val_f1:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            suffix = "crf" if args.use_crf else "linear"
            ckpt_path = ckpt_dir / f"best_{suffix}.pt"
            torch.save({
                "epoch": epoch,
                "use_crf": args.use_crf,
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "label2id": label2id,
                "id2label": id2label,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (f1={val_f1:.4f})")

    # 保存训练日志
    suffix = "crf" if args.use_crf else "linear"
    log_path = output_dir / f"train_log_{suffix}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优实体 F1={best_f1:.4f}")
    print(f"训练日志 → {log_path}")


if __name__ == "__main__":
    main()