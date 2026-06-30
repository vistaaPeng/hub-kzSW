"""
跨数据集离线 Hard Negative Mining
===================================
用 AFQMC 训练的模型去 LCQMC 训练集上挖难负例 → 带回 AFQMC 训练 TripletLoss

为什么跨数据集？
  同数据集挖掘：模型自挖自训 → 确认偏差 → 坍缩 (F1=0.563)
  跨数据集挖掘：挖负例的"盲区"来自不同分布 → 切断确认偏差链条

流程：
  Step 1: 加载 AFQMC 训练的基模
  Step 2: 编码 LCQMC 训练集所有句子为向量库
  Step 3: 对每个 AFQMC 正例 anchor，从 LCQMC 向量库找 Top-K 最相似句作为难负例
  Step 4: 构建 (AFQMC_s1, AFQMC_s2, LCQMC_hard_neg) 三元组
  Step 5: TripletLoss 重训练 AFQMC 模型

用法：
  python src/cross_dataset_hard_neg.py
  python src/cross_dataset_hard_neg.py --top_k 10 --epochs 3
"""

import argparse, json, random, time
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
DATA_AFQMC = ROOT / "data" / "afqmc"
DATA_LCQMC = ROOT / "data" / "lcqmc"
BERT_PATH  = "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"

random.seed(42)


# ── 编码 LCQMC 句子库 ─────────────────────────────────────────────────────

@torch.no_grad()
def encode_corpus(model, tokenizer, sentences, device, max_length=64):
    """编码句子列表 → L2 归一化向量矩阵 [N, H]"""
    model.eval()
    vectors = []
    for text in tqdm(sentences, desc="编码 LCQMC 句子库"):
        enc = encode_single(tokenizer, text, max_length)
        inp = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        vec = model.encode(**inp).cpu().numpy()
        vectors.append(vec)
    vecs = np.concatenate(vectors, axis=0)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


# ── 跨数据集挖掘 ──────────────────────────────────────────────────────────

def mine_cross_dataset(model, tokenizer, train_rows, mine_rows, device, top_k=10):
    """
    对每个训练集正例 anchor，从挖掘集句子库中找最难负例。

    Args:
      train_rows: 训练数据 (含 label，anchor+positive 来源)
      mine_rows:  挖掘数据 (作为负例候选池)

    Returns:
      triplets: list of (anchor, positive, hard_neg_from_mine)
    """
    # 收集挖掘集所有唯一句子
    mine_sents = list(set(r["sentence1"] for r in mine_rows) | set(r["sentence2"] for r in mine_rows))
    print(f"挖掘集候选句库: {len(mine_sents):,}")

    # 编码挖掘集句子
    mine_vecs = encode_corpus(model, tokenizer, mine_sents, device)

    # 取训练集正例
    positives = [r for r in train_rows if r["label"] == 1]
    print(f"训练集正例对: {len(positives):,}")

    # 为 AFQMC 句子建"已知正例集"，用于过滤（可选，跨数据集其实不需要）
    # 但保留以排除 AFQMC 自身标注的正例在 LCQMC 中恰好出现

    triplets = []
    limit = min(len(positives), 5000)  # 最多 5K 三元组
    for r in tqdm(positives[:limit], desc="跨数据集挖掘难负例"):
        anchor = r["sentence1"]
        pos    = r["sentence2"]

        # 编码 anchor
        enc = encode_single(tokenizer, anchor, 64)
        inp = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        anchor_vec = model.encode(**inp).detach().cpu().numpy()
        anchor_vec = anchor_vec / (np.linalg.norm(anchor_vec, axis=1, keepdims=True) + 1e-9)

        # 计算 anchor 与挖掘集所有句子的余弦相似度
        sims = (anchor_vec @ mine_vecs.T)[0]  # [N]

        # 排除与 anchor 或 pos 完全相同的句子
        for i, s in enumerate(mine_sents):
            if s == anchor or s == pos:
                sims[i] = -2.0

        # Top-K 作为难负例候选
        top_indices = np.argsort(sims)[-top_k:][::-1]
        hard_idx = random.choice(top_indices)
        hard_neg = mine_sents[hard_idx]

        triplets.append((anchor, pos, hard_neg))

    print(f"  产出三元组: {len(triplets):,}")
    return triplets


# ── Triplet Dataset ───────────────────────────────────────────────────────

class CrossTripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=64, mix_ratio=1.0, random_pool=None):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mix_ratio = mix_ratio
        self.random_pool = random_pool or []

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, hn = self.triplets[idx]
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


# ── 训练 ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    epoch, total_epochs, margin, grad_accum, loss_type="triplet"):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{loss_type}]", leave=False)
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

        if loss_type == "cosine":
            # CosineEmbeddingLoss: 正例拉近(y=1)，负例推到 margin 外(y=-1)
            loss_pos = F.cosine_embedding_loss(emb_a, emb_p,
                                               torch.ones(emb_a.size(0)).to(device),
                                               margin=0.0)
            loss_neg = F.cosine_embedding_loss(emb_a, emb_n,
                                               -torch.ones(emb_a.size(0)).to(device),
                                               margin=margin)
            loss = loss_pos + loss_neg
        else:
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

    # ── 加载基模 ──────────────────────────────────────────────────────────
    base_ckpt = CKPT_DIR / "biencoder_cosine_best.pt"
    if not base_ckpt.exists():
        print(f"❌ 未找到: {base_ckpt}")
        return

    print(f"加载基模: {base_ckpt}")
    ckpt_data = torch.load(base_ckpt, map_location=device, weights_only=False)
    model = BiEncoder(args.bert_path, pool="mean",
                      num_hidden_layers=ckpt_data.get("args", {}).get("num_hidden_layers", 4)).to(device)
    model.load_state_dict(ckpt_data["state_dict"])
    model.eval()

    # ── 加载数据 ──────────────────────────────────────────────────────────
    train_dir = ROOT / "data" / args.train_on
    mine_dir  = ROOT / "data" / args.mine_from
    train_rows = load_jsonl(train_dir / "train.jsonl")
    mine_rows  = load_jsonl(mine_dir / "train.jsonl")
    print(f"{args.train_on} 训练: {len(train_rows):,}  |  {args.mine_from} 挖负例: {len(mine_rows):,}")

    # ── 跨数据集挖掘 ──────────────────────────────────────────────────────
    print(f"\n[Step 1] 跨数据集挖掘（{args.train_on} → {args.mine_from}, Top-{args.top_k}）...")
    triplets = mine_cross_dataset(model, tokenizer, train_rows, mine_rows, device, args.top_k)

    if len(triplets) < 100:
        print(f"❌ 三元组不足 ({len(triplets)})")
        return

    # ── 重训练 ────────────────────────────────────────────────────────────
    # 构建随机负例候选池（用于混合采样）
    random_pool = []
    if args.mix_ratio < 1.0:
        random_pool = list(set(r["sentence1"] for r in train_rows) | set(r["sentence2"] for r in train_rows))
        print(f"  随机候选池: {len(random_pool):,} 句")

    mix_label = f"mix={args.mix_ratio}" if args.mix_ratio < 1.0 else "纯HN"
    print(f"\n[Step 2] TripletLoss 重训练（margin={args.margin}, {mix_label}, {args.epochs}ep）...")
    train_ds = CrossTripletDataset(triplets, tokenizer, args.max_length,
                                   mix_ratio=args.mix_ratio, random_pool=random_pool)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ds = PairDataset(train_dir / "validation.jsonl", tokenizer, args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=max(total_steps, 1))

    loss_tag = f"_{args.loss}" if args.loss != "triplet" else ""
    marg_tag = f"_m{int(args.margin*100)}" if args.margin != 0.3 else ""
    mix_tag  = f"_mix{int(args.mix_ratio*100)}" if args.mix_ratio < 1.0 else ""
    ckpt_path = CKPT_DIR / f"biencoder_triplet_xhard_{args.train_on}_{args.mine_from}{loss_tag}{marg_tag}{mix_tag}_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                                     epoch, args.epochs, args.margin, args.grad_accum, args.loss)

        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_f1  = val_metrics["f1"]
        val_thr = val_metrics["threshold"]
        f1_pos  = val_metrics["f1_pos"]
        f1_neg  = val_metrics["f1_neg"]
        sim_gap = val_metrics["sim_gap"]
        sim_std = val_metrics["sim_std"]

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_f1:.4f} thr={val_thr:.2f} | {elapsed:.0f}s")
        print(f"  [诊断] F1_pos={f1_pos:.4f} F1_neg={f1_neg:.4f} "
              f"pos_mean={val_metrics['pos_mean_sim']:.2f} neg_mean={val_metrics['neg_mean_sim']:.2f} "
              f"gap={sim_gap:.3f} std={sim_std:.3f}")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_acc": val_metrics["accuracy"], "val_f1": val_f1,
            "f1_pos": f1_pos, "f1_neg": f1_neg,
            "sim_gap": round(sim_gap, 4), "sim_std": round(sim_std, 4),
            "threshold": val_thr, "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "val_acc": val_metrics["accuracy"], "val_f1": val_f1,
                "threshold": val_thr, "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优 → {ckpt_path}")

    # ── 对比基线 ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("跨数据集 Hard Neg 效果对比")
    baseline_log = LOG_DIR / "biencoder_triplet_log.json"
    if baseline_log.exists():
        with open(baseline_log) as f:
            bl = json.load(f)
        bl_best = max(ep["val_f1"] for ep in bl)
        print(f"  TripletLoss（随机负例）  : val_f1={bl_best:.4f}")
        print(f"  TripletLoss（跨数据集HN） : val_f1={best_val_f1:.4f}")
        print(f"  Δ                        : {best_val_f1 - bl_best:+.4f}")
    else:
        print(f"  TripletLoss（跨数据集HN） : val_f1={best_val_f1:.4f}")

    # 对比离线同数据集 HN（如有）
    offline_log = LOG_DIR / "biencoder_triplet_hardneg_log.json"
    if offline_log.exists():
        with open(offline_log) as f:
            ol = json.load(f)
        ol_best = max(ep["val_f1"] for ep in ol)
        print(f"  TripletLoss（同数据集HN） : val_f1={ol_best:.4f}")

    log_path = LOG_DIR / f"biencoder_triplet_xhard_{args.train_on}_{args.mine_from}{loss_tag}{marg_tag}{mix_tag}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n日志 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="跨数据集离线 Hard Negative Mining")
    p.add_argument("--bert_path",   default=BERT_PATH)
    p.add_argument("--train_on",    default="afqmc", choices=["afqmc", "bq_corpus", "lcqmc"],
                   help="训练数据集（anchor + positive 来源）")
    p.add_argument("--mine_from",   default="lcqmc", choices=["afqmc", "bq_corpus", "lcqmc"],
                   help="挖掘难负例的数据集")
    p.add_argument("--top_k",       default=10, type=int, help="每 anchor 取 Top-K 难负例候选")
    p.add_argument("--epochs",      default=3, type=int)
    p.add_argument("--batch_size",  default=32, type=int)
    p.add_argument("--max_length",  default=64, type=int)
    p.add_argument("--lr",          default=2e-5, type=float)
    p.add_argument("--margin",      default=0.3, type=float)
    p.add_argument("--grad_accum",  default=1, type=int)
    p.add_argument("--mix_ratio",   default=1.0, type=float,
                   help="HN 比例：1.0=全HN, 0.5=半HN半随机")
    p.add_argument("--loss",        default="triplet", choices=["triplet", "cosine"],
                   help="TripletLoss 或 CosineEmbeddingLoss")
    return p.parse_args()


if __name__ == "__main__":
    main()
