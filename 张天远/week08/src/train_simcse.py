"""
SimCSE 无监督对比学习 — 文本匹配训练（第三种训练范式）

教学重点：
  1. SimCSE 无需任何标注数据——正例由同一句子的两次 Dropout 构造
  2. In-batch negatives：batch 内其他句子自动作为负例
  3. InfoNCE loss：对比学习标准损失函数
  4. 与 CosineEmbeddingLoss / TripletLoss 形成完整的三范式对比

原理（SimCSE unsupervised）：
  h_i = BERT(x_i, dropout_mask_1)    ← 第一次前向
  h_i' = BERT(x_i, dropout_mask_2)   ← 第二次前向（不同 dropout）
  → (h_i, h_i') 为正例对，loss 让两者在余弦空间中更近
  → 同 batch 的其他句子为负例

使用方式：
  python src/train_simcse.py
  python src/train_simcse.py --epochs 1 --temperature 0.05
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model import BiEncoder
from evaluate import eval_biencoder
from dataset import PairDataset, load_jsonl

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"  # default, overridable via --data_dir
BERT_PATH  = "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"


# ── SimCSE Dataset ────────────────────────────────────────────────────────

class SimCSEDataset(Dataset):
    """
    无监督 SimCSE：每个样本只是一句话（不需要标签）。

    从 AFQMC 训练集中抽取所有出现的句子（去重后约 20K 条唯一句子）。
    每次 __getitem__ 返回同一句话的 tokenization——模型会跑两次 dropout 得到正例对。
    """
    def __init__(self, data_path, tokenizer, max_length=64):
        rows = load_jsonl(data_path)
        # 收集所有唯一句子
        sentences = set()
        for r in rows:
            sentences.add(r["sentence1"])
            sentences.add(r["sentence2"])
        self.sentences = list(sentences)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
        }


# ── InfoNCE Loss ──────────────────────────────────────────────────────────

def infonce_loss(z1, z2, temperature=0.05):
    """
    z1, z2: [B, H] — 同一 batch 的两组嵌入（同一数据、不同 dropout）

    计算 InfoNCE：每个样本 i 的正例是 z2[i]，负例是 z2 中所有 j≠i。
    """
    B = z1.size(0)
    # 余弦相似度矩阵 [B, B]
    sim = torch.mm(z1, z2.T) / temperature
    # 对角线上是正例对
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ── 训练一个 epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    epoch, total_epochs, temperature, grad_accum):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [SimCSE]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        # 两次前向 = 不同 dropout → 正例对
        # 关键：需要 model.train() 让 dropout 生效
        z1 = model.encode(input_ids, attention_mask, token_type_ids)  # [B, H]
        z2 = model.encode(input_ids, attention_mask, token_type_ids)  # [B, H] (不同 dropout)

        loss = infonce_loss(z1, z2, temperature)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        bs = input_ids.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"SimCSE 无监督对比学习  pool={args.pool}  layers={args.num_hidden_layers}  temp={args.temperature}")

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\n构建 SimCSE 数据集...")
    data_dir = Path(args.data_dir)
    train_ds = SimCSEDataset(data_dir / "train.jsonl", tokenizer, args.max_length)
    print(f"  唯一句子: {len(train_ds):,} 条")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    # 验证集仍用 PairDataset + 阈值搜索（与其他方法统一口径）
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("\n构建 BiEncoder...")
    model = BiEncoder(args.bert_path, pool=args.pool,
                      num_hidden_layers=args.num_hidden_layers).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {total:.1f}M")

    # ── 优化器 ────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"总步数: {total_steps}  warmup: {warmup_steps}")

    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_path   = CKPT_DIR / "biencoder_simcse_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.temperature, args.grad_accum)

        # 每个 epoch 末评估（用余弦相似度 + 阈值搜索）
        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["f1"]
        val_thr = val_metrics["threshold"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} threshold={val_thr:.2f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_acc": val_acc, "val_f1": val_f1,
            "threshold": val_thr, "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "threshold": val_thr, "val_acc": val_acc, "val_f1": val_f1,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优 → {ckpt_path}  (val_f1={val_f1:.4f})")

    # ── 保存日志 ──────────────────────────────────────────────────────────
    log_path = LOG_DIR / "biencoder_simcse_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"日志 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="SimCSE 无监督对比学习")
    p.add_argument("--bert_path",         default=BERT_PATH)
    p.add_argument("--data_dir",          default=str(DATA_DIR), type=str,
                   help="数据目录（默认: data/afqmc）")
    p.add_argument("--pool",              default="mean", choices=["cls","mean","max"])
    p.add_argument("--num_hidden_layers", default=4, type=int)
    p.add_argument("--epochs",            default=3, type=int)
    p.add_argument("--batch_size",        default=64, type=int,
                   help="SimCSE 依赖大 batch 提供多样负例（推荐 ≥32）")
    p.add_argument("--max_length",        default=64, type=int)
    p.add_argument("--lr",                default=3e-5, type=float,
                   help="对比学习通常用稍高学习率")
    p.add_argument("--temperature",       default=0.05, type=float,
                   help="InfoNCE 温度参数（越小对困难负例越敏感，默认 0.05）")
    p.add_argument("--warmup_ratio",      default=0.1, type=float)
    p.add_argument("--grad_accum",        default=1, type=int)
    return p.parse_args()


if __name__ == "__main__":
    main()
