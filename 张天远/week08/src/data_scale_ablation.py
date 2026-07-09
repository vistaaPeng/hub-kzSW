"""
数据规模消融 — Scaling Law for Text Matching

教学重点：
  1. 数据量越大，F1 越高——但边际收益递减
  2. 小数据下模型可能过拟合，大数据下才能体现真正泛化能力
  3. AFQMC 34K 只能做三档，LCQMC 238K 可以做六档

实验设计：
  训练集大小: 1K, 5K, 10K, 50K, 100K, full (190K)
  验证集: LCQMC validation (固定，约 24K 条)
  模型: BiEncoder Cosine, 4层, 2 epoch (快速消融)
  输出: F1 曲线图 + JSON 数据

使用方式：
  python src/data_scale_ablation.py
  python src/data_scale_ablation.py --epochs 1 --data_dir data/lcqmc
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

from model import BiEncoder
from dataset import PairDataset, load_jsonl, encode_single
from evaluate import eval_biencoder

ROOT       = Path(__file__).parent.parent
BERT_PATH  = "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"
FIG_DIR    = OUTPUT_DIR / "figures"

random.seed(42)

# ── 训练函数（轻量版，仅用于消融）─────────────────────────────────────────

def train_biencoder_epoch(model, loader, optimizer, scheduler, device, margin, grad_accum):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        enc_a = {}
        for k in ("input_ids_a","attention_mask_a","token_type_ids_a"):
            if k in batch:
                enc_a[k.replace("_a","")] = batch[k].to(device)
        enc_b = {}
        for k in ("input_ids_b","attention_mask_b","token_type_ids_b"):
            if k in batch:
                enc_b[k.replace("_b","")] = batch[k].to(device)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(device)
        else:
            continue

        emb_a, emb_b = model(enc_a, enc_b)
        cos_target = (labels.float() * 2 - 1)
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=margin)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


# ── 数据规模消融 ──────────────────────────────────────────────────────────

def run_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    data_dir = Path(args.data_dir)
    ds_name = data_dir.name  # "lcqmc" or "afqmc"

    # 加载完整训练集
    train_rows = load_jsonl(data_dir / "train.jsonl")
    pos_rows = [r for r in train_rows if r["label"] == 1]
    neg_rows = [r for r in train_rows if r["label"] == 0]
    print(f"全量训练集: {len(train_rows):,} 条 (正 {len(pos_rows):,} / 负 {len(neg_rows):,})")

    # 验证集（固定，共用）
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    val_rows = load_jsonl(data_dir / "validation.jsonl")
    val_ds = PairDataset(data_dir / "validation.jsonl", tokenizer, args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"验证集: {len(val_rows):,} 条")

    SIZES = [1000, 5000, 10000, 50000, 100000]
    results = []

    for size in SIZES:
        if size > len(train_rows):
            print(f"\n  跳过 size={size:,}（超出全量 {len(train_rows):,}）")
            continue

        n_pos = min(size // 3, len(pos_rows))  # 保持 1:2 正负比
        n_neg = size - n_pos
        subset = random.sample(pos_rows, n_pos) + random.sample(neg_rows, n_neg)
        random.shuffle(subset)
        print(f"\n{'='*55}")
        print(f"数据规模: {size:,} 条 (正 {n_pos:,} / 负 {n_neg:,})")

        # 构建临时 DataLoader
        train_ds = PairDataset.from_rows(subset, tokenizer, args.max_length)
        train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(subset) // 2),
                                  shuffle=True, num_workers=0)

        # 重新初始化模型（避免不同规模间的干扰）
        model = BiEncoder(args.bert_path, pool="mean",
                          num_hidden_layers=args.num_hidden_layers).to(device)
        total = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  参数量: {total:.1f}M")

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs // args.grad_accum
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)),
            num_training_steps=max(1, total_steps))

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_biencoder_epoch(
                model, train_loader, optimizer, scheduler, device,
                args.margin, args.grad_accum)

        elapsed = time.time() - t0

        # 评估
        val_metrics = eval_biencoder(model, val_loader, device)
        print(f"  train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}  "
              f"threshold={val_metrics['threshold']:.2f}  {elapsed:.0f}s")

        results.append({
            "size": size, "n_pos": n_pos, "n_neg": n_neg,
            "train_loss": round(train_loss, 4),
            "val_acc": round(val_metrics['accuracy'], 4),
            "val_f1": round(val_metrics['f1'], 4),
            "threshold": round(val_metrics['threshold'], 2),
            "elapsed_s": round(elapsed),
        })

    # ── 保存 ────────────────────────────────────────────────────────────
    log_path = LOG_DIR / f"data_scale_{ds_name}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"数据规模消融结果（{ds_name}）")
    print(f"{'Size':>8}  {'F1(val)':>9}  {'Acc(val)':>9}  {'Time':>7}")
    print(f"{'-'*40}")
    for r in results:
        print(f"{r['size']:>8,}  {r['val_f1']:>9.4f}  {r['val_acc']:>9.4f}  {r['elapsed_s']:>6}s")
    print(f"\n结果 → {log_path}")


# ── 补丁：PairDataset 从 rows 直接构建 ──────────────────────────────────

def _pair_dataset_from_rows(self, rows, tokenizer, max_length):
    """给 PairDataset 补一个类方法，从内存中的 rows 构建"""
    ds = PairDataset.__new__(PairDataset)
    ds.rows = rows
    ds.tokenizer = tokenizer
    ds.max_length = max_length
    return ds

PairDataset.from_rows = classmethod(_pair_dataset_from_rows)


def parse_args():
    p = argparse.ArgumentParser(description="数据规模消融实验")
    p.add_argument("--bert_path",    default=BERT_PATH)
    p.add_argument("--data_dir",     default="data/lcqmc", type=str)
    p.add_argument("--epochs",       default=2, type=int, help="每轮训练 epoch 数")
    p.add_argument("--batch_size",   default=32, type=int)
    p.add_argument("--max_length",   default=64, type=int)
    p.add_argument("--lr",           default=2e-5, type=float)
    p.add_argument("--margin",       default=0.3, type=float)
    p.add_argument("--grad_accum",   default=1, type=int)
    p.add_argument("--num_hidden_layers", default=4, type=int)
    return p.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
