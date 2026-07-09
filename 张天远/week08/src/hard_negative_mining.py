"""
Hard Negative Mining — 难负例挖掘 + TripletLoss 重训练

教学重点：
  1. 随机负例太"容易"——模型看几个词就能区分，学不到精细判别
  2. 难负例：模型预测高分但标注为"不相似"的样本对
  3. 流程：基模训练 → 编码全库 → 挖掘高分负例 → 重训 TripletLoss

流程：
  Step 1: 用已训练的 BiEncoder Cosine 编码所有句子
  Step 2: 对每个 anchor，找余弦相似度最高但标签为 0 的负例
  Step 3: 构建 (anchor, positive, hard_negative) 三元组
  Step 4: 用 TripletLoss 重训练，对比随机负例 vs 难负例效果

使用方式：
  python src/hard_negative_mining.py
  python src/hard_negative_mining.py --top_k 5 --epochs 1
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from model import BiEncoder
from evaluate import eval_biencoder
from dataset import PairDataset, load_jsonl, encode_single

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"
BERT_PATH  = "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"

random.seed(42)


# ── Step 1: 编码所有句子 ──────────────────────────────────────────────────

@torch.no_grad()
def encode_all_sentences(model, tokenizer, sentences, device, max_length=64):
    """批量编码句子列表 → 句向量矩阵 [N, H]"""
    model.eval()
    vectors = []
    for text in tqdm(sentences, desc="编码句子"):
        enc = encode_single(tokenizer, text, max_length)
        inp = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        vec = model.encode(**inp).cpu().numpy()
        vectors.append(vec)
    return np.concatenate(vectors, axis=0)  # [N, H]


# ── Step 2: 挖掘难负例 ────────────────────────────────────────────────────

def mine_hard_negatives(model, tokenizer, train_rows, device, top_k=5):
    """
    对每条正例对，从全集中找 cos 相似度最高但标签为 0 的句子作为难负例。

    返回: list of (anchor, positive, hard_negative)
    """
    # 收集所有唯一句子
    sent_to_idx = {}
    idx_to_sent = []
    for r in train_rows:
        for s in (r["sentence1"], r["sentence2"]):
            if s not in sent_to_idx:
                sent_to_idx[s] = len(idx_to_sent)
                idx_to_sent.append(s)

    print(f"唯一句子: {len(idx_to_sent):,}")

    # 编码
    vecs = encode_all_sentences(model, tokenizer, idx_to_sent, device)
    # L2 归一化
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    # 为每个句子建立"已知正例集"（用于排除）
    pos_neighbors = defaultdict(set)
    for r in train_rows:
        if r["label"] == 1:
            pos_neighbors[r["sentence1"]].add(r["sentence2"])
            pos_neighbors[r["sentence2"]].add(r["sentence1"])

    # 挖掘难负例
    positives = [r for r in train_rows if r["label"] == 1]
    print(f"正例对: {len(positives):,}")

    triplets = []
    for r in tqdm(positives[:5000], desc="挖掘难负例"):  # 取 5K 正例，控制规模
        anchor = r["sentence1"]
        pos    = r["sentence2"]

        anchor_idx = sent_to_idx[anchor]
        anchor_vec = vecs[anchor_idx:anchor_idx+1]  # [1, H]

        # 计算与所有句子的余弦相似度
        sims = (anchor_vec @ vecs.T)[0]  # [N]
        # 排除自身和已知正例
        exclude = {anchor_idx} | {sent_to_idx[s] for s in pos_neighbors.get(anchor, set())}
        mask = np.ones(len(idx_to_sent), dtype=float)
        for ex in exclude:
            mask[ex] = -np.inf
        sims = sims * mask + (1 - mask) * (-2.0)

        # 取 Top-K 相似但非正例的作为难负例候选
        hard_indices = np.argsort(sims)[-top_k:][::-1]
        hard_indices = [i for i in hard_indices if mask[i] > -0.5]  # 过滤掉排除的

        if not hard_indices:
            continue

        # 随机选一个作为难负例
        hard_neg = idx_to_sent[random.choice(hard_indices)]
        triplets.append((anchor, pos, hard_neg))

    print(f"  挖掘难负例三元组: {len(triplets):,}")
    return triplets


# ── Triplet Dataset with Hard Negatives ────────────────────────────────────

class HardTripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=64, mix_ratio=1.0, random_pool=None):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mix_ratio = mix_ratio        # 1.0=全HN, 0.5=半HN半随机, 0.0=全随机
        self.random_pool = random_pool or []  # 随机负例候选句列表

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, hn = self.triplets[idx]
        # 混合采样：以 mix_ratio 概率用 HN，否则随机选
        if self.random_pool and random.random() > self.mix_ratio:
            n = random.choice(self.random_pool)
        else:
            n = hn
        return {
            **{"input_ids_a": encode_single(self.tokenizer, a, self.max_length)["input_ids"],
               "attention_mask_a": encode_single(self.tokenizer, a, self.max_length)["attention_mask"],
               "token_type_ids_a": encode_single(self.tokenizer, a, self.max_length)["token_type_ids"]},
            **{"input_ids_p": encode_single(self.tokenizer, p, self.max_length)["input_ids"],
               "attention_mask_p": encode_single(self.tokenizer, p, self.max_length)["attention_mask"],
               "token_type_ids_p": encode_single(self.tokenizer, p, self.max_length)["token_type_ids"]},
            **{"input_ids_n": encode_single(self.tokenizer, n, self.max_length)["input_ids"],
               "attention_mask_n": encode_single(self.tokenizer, n, self.max_length)["attention_mask"],
               "token_type_ids_n": encode_single(self.tokenizer, n, self.max_length)["token_type_ids"]},
        }


# ── TripletLoss 训练 ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    epoch, total_epochs, margin, grad_accum):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Triplet+Hard]", leave=False)
    for step, batch in enumerate(pbar):
        enc_a = {"input_ids": batch["input_ids_a"].to(device),
                 "attention_mask": batch["attention_mask_a"].to(device),
                 "token_type_ids": batch["token_type_ids_a"].to(device)}
        enc_p = {"input_ids": batch["input_ids_p"].to(device),
                 "attention_mask": batch["attention_mask_p"].to(device),
                 "token_type_ids": batch["token_type_ids_p"].to(device)}
        enc_n = {"input_ids": batch["input_ids_n"].to(device),
                 "attention_mask": batch["attention_mask_n"].to(device),
                 "token_type_ids": batch["token_type_ids_n"].to(device)}

        emb_a = model.encode(**enc_a)
        emb_p = model.encode(**enc_p)
        emb_n = model.encode(**enc_n)

        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=margin)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        bs = emb_a.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ── 加载已训练的 BiEncoder Cosine ─────────────────────────────────────
    base_ckpt = CKPT_DIR / "biencoder_cosine_best.pt"
    if not base_ckpt.exists():
        base_ckpt = CKPT_DIR / "biencoder_cosine_best_core.pt"
    if not base_ckpt.exists():
        print(f"❌ 未找到 BiEncoder Cosine checkpoint")
        print("  请先运行: python src/train_biencoder.py --loss cosine")
        return

    print(f"加载基模: {base_ckpt}")
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    model = BiEncoder(args.bert_path, pool="mean",
                      num_hidden_layers=ckpt.get("args", {}).get("num_hidden_layers", 4)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── 挖掘难负例 ────────────────────────────────────────────────────────
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    print(f"\n[Step 1] 挖掘难负例（Top-{args.top_k}）...")
    triplets = mine_hard_negatives(model, tokenizer, train_rows, device, args.top_k)

    if len(triplets) < 100:
        print(f"❌ 难负例不足 ({len(triplets)} 个)，退出")
        return

    # ── 重训练（TripletLoss + 难负例，可选混合采样）──────────────────────
    mix_label = f"mix={args.mix_ratio}" if args.mix_ratio < 1.0 else "纯HN"
    print(f"\n[Step 2] 重训练 TripletLoss（margin={args.margin}, {mix_label}）...")

    # 构建随机负例候选池（用于混合采样）
    random_pool = []
    if args.mix_ratio < 1.0:
        random_pool = list(set(r["sentence1"] for r in train_rows) | set(r["sentence2"] for r in train_rows))
        print(f"  随机候选池: {len(random_pool):,} 句")

    train_ds = HardTripletDataset(triplets, tokenizer, args.max_length,
                                  mix_ratio=args.mix_ratio, random_pool=random_pool)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_ds   = PairDataset(DATA_DIR / "validation.jsonl", tokenizer, args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps)

    mix_suffix = f"_mix{int(args.mix_ratio*100)}" if args.mix_ratio < 1.0 else ""
    ckpt_path = CKPT_DIR / f"biencoder_triplet_hardneg{mix_suffix}_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.margin, args.grad_accum)

        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"threshold={val_metrics['threshold']:.2f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_acc": val_metrics['accuracy'], "val_f1": val_metrics['f1'],
            "threshold": val_metrics['threshold'], "elapsed_s": elapsed,
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "threshold": val_metrics['threshold'],
                "val_acc": val_metrics['accuracy'], "val_f1": val_metrics['f1'],
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优 → {ckpt_path} (val_f1={val_metrics['f1']:.4f})")

    # ── 对比基线 ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("难负例挖掘 效果对比")
    # 读取随机 TripletLoss 基线（如有）
    baseline_log = LOG_DIR / "biencoder_triplet_log.json"
    if baseline_log.exists():
        with open(baseline_log) as f:
            bl = json.load(f)
        bl_f1 = bl[-1]["val_f1"]
        delta = best_val_f1 - bl_f1
        print(f"  TripletLoss（随机负例）: val_f1={bl_f1:.4f}")
        print(f"  TripletLoss（难负例）  : val_f1={best_val_f1:.4f}")
        print(f"  提升 (Δ)              : {delta:+.4f}")
    else:
        print(f"  TripletLoss（难负例）  : val_f1={best_val_f1:.4f}")
        print("  （需运行 TripletLoss 基线以对比）")

    # ── 保存日志 ──────────────────────────────────────────────────────────
    log_path = LOG_DIR / f"biencoder_triplet_hardneg{mix_suffix}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n日志 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Hard Negative Mining + TripletLoss 重训练")
    p.add_argument("--bert_path",    default=BERT_PATH)
    p.add_argument("--top_k",        default=5, type=int,
                   help="每 anchor 取 Top-K 高分负例作为难负例候选")
    p.add_argument("--epochs",       default=1, type=int)
    p.add_argument("--batch_size",   default=32, type=int)
    p.add_argument("--max_length",   default=64, type=int)
    p.add_argument("--lr",           default=2e-5, type=float)
    p.add_argument("--margin",       default=0.3, type=float)
    p.add_argument("--grad_accum",   default=1, type=int)
    p.add_argument("--mix_ratio",    default=1.0, type=float,
                   help="HN 比例：1.0=全HN, 0.5=半HN半随机, 0.0=全随机")
    return p.parse_args()


if __name__ == "__main__":
    main()
