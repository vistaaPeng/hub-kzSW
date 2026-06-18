"""
在人民日报 NER（peoples_daily）上训练 BERT 序列标注模型

使用方式：
  cd work7
  python train.py                  # BERT + Linear
  python train.py --use_crf        # BERT + CRF
  python train.py --epochs 5       # 自定义轮数
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_label_schema, build_dataloaders, DATA_DIR
from model import build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = Path(__file__).parent / "outputs" / "checkpoints"
LOG_DIR = Path(__file__).parent / "outputs" / "logs"


def evaluate_epoch(model, loader, id2label, device, use_crf):
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                _, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()
            labels_np = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                for j, gold_id in enumerate(labels_np[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pid = pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0
                    else:
                        pid = pred_ids_list[i][j]
                    pred_seq.append(id2label.get(pid, "O"))
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return total_loss / len(loader), seqeval_f1(all_golds, all_preds)


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

    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    print(f"数据集：peoples_daily（{DATA_DIR}）")

    labels, label2id, id2label = build_label_schema()
    print(f"BIO 标签数：{len(labels)} → {labels}")

    tokenizer = BertTokenizerFast.from_pretrained(str(args.bert_path))
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=len(labels),
        dropout=args.dropout,
    ).to(device)

    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters())
        + list(model.dropout.parameters())
        + (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum,
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python evaluate.py {'--use_crf' if args.use_crf else ''} --split test")


def parse_args():
    parser = argparse.ArgumentParser(description="人民日报 NER 训练")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
