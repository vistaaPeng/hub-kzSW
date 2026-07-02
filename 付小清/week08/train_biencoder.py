"""
在 LCQMC / BQ Corpus 上训练 BiEncoder

使用方式：
  cd work8
  python train_biencoder.py --dataset bq_corpus --loss cosine
  python train_biencoder.py --dataset lcqmc --loss triplet
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

from dataset import build_pair_loaders, build_triplet_loader, resolve_data_dir
from evaluate import eval_biencoder, ckpt_dir, log_dir
from model import build_biencoder

PROJECT_ROOT = Path(__file__).parent.parent
BERT_PATH = PROJECT_ROOT.parent / "pretrain_models" / "bert-base-chinese"


def train_one_epoch_cosine(model, loader, optimizer, scheduler, device,
                           epoch, total_epochs, margin, grad_accum):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Cosine]", leave=False)
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        labels = batch["label"].to(device)
        emb_a, emb_b = model(batch_a, batch_b)
        cos_target = labels.float() * 2 - 1
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=margin)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def train_one_epoch_triplet(model, loader, optimizer, scheduler, device,
                            epoch, total_epochs, margin, grad_accum):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Triplet]", leave=False)
    for step, batch in enumerate(pbar):
        emb_a = model.encode(
            batch["input_ids_a"].to(device),
            batch["attention_mask_a"].to(device),
            batch["token_type_ids_a"].to(device),
        )
        emb_p = model.encode(
            batch["input_ids_p"].to(device),
            batch["attention_mask_p"].to(device),
            batch["token_type_ids_p"].to(device),
        )
        emb_n = model.encode(
            batch["input_ids_n"].to(device),
            batch["attention_mask_n"].to(device),
            batch["token_type_ids_n"].to(device),
        )
        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=margin)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        bs = emb_a.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def parse_args():
    parser = argparse.ArgumentParser(description="work8 BiEncoder 训练")
    parser.add_argument("--dataset", required=True, choices=["lcqmc", "bq_corpus"])
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--loss", default="cosine", choices=["cosine", "triplet"])
    parser.add_argument("--pool", default="mean", choices=["cls", "mean", "max"])
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--head_lr_mult", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--margin", default=0.3, type=float)
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
    print(f"数据集: {args.dataset}  Loss: {args.loss}  层数: {args.num_hidden_layers}  Epochs: {args.epochs}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")
    if args.loss == "cosine":
        train_loader, val_loader, _ = build_pair_loaders(
            data_dir, tokenizer, args.max_length, args.batch_size,
        )
    else:
        train_loader, val_loader = build_triplet_loader(
            data_dir, tokenizer, args.max_length, args.batch_size,
        )

    print("\n构建模型...")
    model = build_biencoder(
        bert_path=args.bert_path,
        pool=args.pool,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": args.lr},
        {"params": model.dropout.parameters(), "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"总训练步数: {total_steps}  Warmup: {warmup_steps}")

    ckpt_path = out_ckpt / f"biencoder_{args.loss}_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.loss == "cosine":
            train_loss = train_one_epoch_cosine(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )
        else:
            train_loss = train_one_epoch_triplet(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )

        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"threshold={val_metrics['threshold']:.2f} | {elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_f1_pos": val_metrics["f1_pos"],
            "threshold": val_metrics["threshold"],
            "elapsed_s": elapsed,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "threshold": val_metrics["threshold"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "args": vars(args),
            }, ckpt_path)
            print(f"  [OK] 新最优模型 -> {ckpt_path}  (val_f1={val_metrics['f1']:.4f})")

    log_path = out_log / f"train_biencoder_{args.loss}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"日志 -> {log_path}")
    print(f"评估：python evaluate.py --dataset {args.dataset} --model_type biencoder --loss {args.loss} --split test")


if __name__ == "__main__":
    main()
