"""
CrossEncoder 作 Hard Negative 挖掘器
=====================================
用 BiEncoder 粗筛 Top-K 候选 → CrossEncoder 精排 → 选出真正的难负例

为什么用 CrossEncoder 挖？
  BiEncoder 挖的"难负例"可能是它的 FP（确认偏差），CrossEncoder 有全层交互，
  判别力更强——用它挖出来的负例更接近"真正的难负例"而非模型盲区的噪声。

如果 CE 挖完训练后仍然坍缩 → 容量瓶颈确凿（4 层无解）
如果 CE 挖完训练后不坍缩   → BiEncoder 自挖自训的确认偏差才是主因

用法：
  python src/ce_hard_neg_mining.py
  python src/ce_hard_neg_mining.py --top_k 50 --train_on afqmc --mine_from afqmc
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

from model import BiEncoder, CrossEncoder
from evaluate import eval_biencoder
from dataset import PairDataset, load_jsonl, encode_single

ROOT       = Path(__file__).parent.parent
BERT_PATH  = "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"

random.seed(42)


# ── BiEncoder 编码（粗筛用）─────────────────────────────────────────────

@torch.no_grad()
def bi_encode_corpus(model, tokenizer, sentences, device, max_length=64):
    model.eval()
    vectors = []
    for text in tqdm(sentences, desc="BiEncoder 编码"):
        enc = encode_single(tokenizer, text, max_length)
        inp = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        vec = model.encode(**inp).detach().cpu().numpy()
        vectors.append(vec)
    vecs = np.concatenate(vectors, axis=0)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


# ── CrossEncoder 精排 ───────────────────────────────────────────────────

@torch.no_grad()
def ce_rerank_scores(model, tokenizer, anchor, candidates, device, max_length=128, batch_size=64):
    """CrossEncoder 对候选池打分，返回每个候选的"正类"概率"""
    model.eval()
    probs = []
    for i in range(0, len(candidates), batch_size):
        batch_cand = candidates[i:i + batch_size]
        enc = tokenizer([anchor] * len(batch_cand), batch_cand,
                        max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc)  # [B, 2]
        p = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        probs.extend(p)
    return probs


# ── 联合挖掘 ─────────────────────────────────────────────────────────────

def mine_with_ce(bi_model, ce_model, tokenizer, train_rows, mine_sents,
                 device, bi_top_k=50, max_anchors=5000):
    """
    BiEncoder 粗筛 Top-K → CrossEncoder 精排选最难负例

    Returns: triplets list of (anchor, positive, hard_negative)
    """
    # 编码挖掘集
    mine_vecs = bi_encode_corpus(bi_model, tokenizer, mine_sents, device)

    positives = [r for r in train_rows if r["label"] == 1]
    limit = min(len(positives), max_anchors)
    print(f"锚点: {limit:,}  |  候选库: {len(mine_sents):,}  |  Bi Top-K={bi_top_k}")

    # 建立 anchor 的"已知正例集"用于排除
    pos_neighbors = {}
    for r in train_rows:
        if r["label"] == 1:
            s = r["sentence1"]
            pos_neighbors.setdefault(s, set()).add(r["sentence2"])

    triplets = []
    for r in tqdm(positives[:limit], desc="CE 挖掘难负例"):
        anchor = r["sentence1"]
        pos    = r["sentence2"]

        # BiEncoder 粗筛
        enc = encode_single(tokenizer, anchor, 64)
        inp = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        anchor_vec = bi_model.encode(**inp).detach().cpu().numpy()
        anchor_vec = anchor_vec / (np.linalg.norm(anchor_vec, axis=1, keepdims=True) + 1e-9)
        sims = (anchor_vec @ mine_vecs.T)[0]

        # 排除 anchor 自身 + 已知正例
        exclude_indices = set()
        for i, s in enumerate(mine_sents):
            if s == anchor or s == pos or s in pos_neighbors.get(anchor, set()):
                exclude_indices.add(i)
        for ei in exclude_indices:
            sims[ei] = -2.0

        # Bi 粗筛 Top-K
        bi_top_indices = np.argsort(sims)[-bi_top_k:][::-1]
        bi_candidates = [mine_sents[i] for i in bi_top_indices if sims[i] > -1.0]

        if len(bi_candidates) < 2:
            continue

        # CrossEncoder 精排
        ce_scores = ce_rerank_scores(ce_model, tokenizer, anchor, bi_candidates, device)
        best_idx = int(np.argmax(ce_scores))
        hard_neg = bi_candidates[best_idx]

        triplets.append((anchor, pos, hard_neg))

    print(f"  产出三元组: {len(triplets):,}")
    return triplets


# ── Dataset & Training (same as cross_dataset_hard_neg.py) ──────────────

class HardTripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=64):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.triplets)
    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
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


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs, margin, grad_accum):
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [CE-HN]", leave=False)
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ── 加载模型 ──────────────────────────────────────────────────────────
    bi_ckpt_path = CKPT_DIR / "biencoder_cosine_best.pt"
    ce_ckpt_path = CKPT_DIR / "crossencoder_best.pt"
    if not bi_ckpt_path.exists() or not ce_ckpt_path.exists():
        print(f"❌ 缺 checkpoint: bi={bi_ckpt_path.exists()} ce={ce_ckpt_path.exists()}")
        return

    print(f"BiEncoder: {bi_ckpt_path}")
    bi_ckpt = torch.load(bi_ckpt_path, map_location=device, weights_only=False)
    bi_model = BiEncoder(args.bert_path, pool="mean",
                         num_hidden_layers=bi_ckpt.get("args",{}).get("num_hidden_layers",4)).to(device)
    bi_model.load_state_dict(bi_ckpt["state_dict"])

    print(f"CrossEncoder: {ce_ckpt_path}")
    ce_ckpt = torch.load(ce_ckpt_path, map_location=device, weights_only=False)
    ce_model = CrossEncoder(args.bert_path,
                            num_hidden_layers=ce_ckpt.get("args",{}).get("num_hidden_layers",4)).to(device)
    ce_model.load_state_dict(ce_ckpt["state_dict"])

    # ── 加载数据 ──────────────────────────────────────────────────────────
    train_dir = ROOT / "data" / args.train_on
    mine_dir  = ROOT / "data" / args.mine_from
    train_rows = load_jsonl(train_dir / "train.jsonl")
    mine_rows  = load_jsonl(mine_dir / "train.jsonl")
    mine_sents = list(set(r["sentence1"] for r in mine_rows) | set(r["sentence2"] for r in mine_rows))
    print(f"训练集: {args.train_on} ({len(train_rows):,})  |  挖掘集: {args.mine_from} ({len(mine_sents):,} 句)")

    # ── 挖掘 ──────────────────────────────────────────────────────────────
    print(f"\n[Step 1] CrossEncoder 挖掘（Bi Top-{args.bi_top_k} → CE 精排）...")
    triplets = mine_with_ce(bi_model, ce_model, tokenizer, train_rows, mine_sents,
                            device, bi_top_k=args.bi_top_k)

    if len(triplets) < 100:
        print(f"❌ 三元组不足 ({len(triplets)})"); return

    # ── 训练 ──────────────────────────────────────────────────────────────
    # 使用 BiEncoder 训练（同一模型，公平对比）
    print(f"\n[Step 2] TripletLoss 重训练（margin={args.margin}, {args.epochs}ep）...")
    train_ds = HardTripletDataset(triplets, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ds = PairDataset(train_dir / "validation.jsonl", tokenizer, args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(bi_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // max(args.grad_accum, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=max(total_steps, 1))

    ckpt_path = CKPT_DIR / f"biencoder_triplet_cehn_{args.train_on}_{args.mine_from}_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(bi_model, train_loader, optimizer, scheduler, device,
                                     epoch, args.epochs, args.margin, args.grad_accum)
        val_metrics = eval_biencoder(bi_model, val_loader, device)
        elapsed = time.time() - t0

        val_f1  = val_metrics["f1"]
        val_thr = val_metrics["threshold"]
        f1_pos  = val_metrics.get("f1_pos", 0)
        f1_neg  = val_metrics.get("f1_neg", 0)
        sim_gap = val_metrics.get("sim_gap", 0)
        sim_std = val_metrics.get("sim_std", 0)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_f1={val_f1:.4f} thr={val_thr:.2f} | {elapsed:.0f}s")
        if "pos_mean_sim" in val_metrics:
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
            torch.save({"epoch": epoch, "state_dict": bi_model.state_dict(),
                        "val_acc": val_metrics["accuracy"], "val_f1": val_f1,
                        "threshold": val_thr, "args": vars(args)}, ckpt_path)

    # ── 保存日志 ──────────────────────────────────────────────────────────
    log_path = LOG_DIR / f"biencoder_triplet_cehn_{args.train_on}_{args.mine_from}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    # ── 对比 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("CrossEncoder 挖掘 效果对比")
    for label, log_name in [("随机负例", "biencoder_triplet_log.json"),
                             ("Bi-HN (同域)", "biencoder_triplet_hardneg_log.json"),
                             ("Bi-HN (LCQMC)", "biencoder_triplet_xhard_afqmc_lcqmc_log.json"),
                             ("CE-HN", f"biencoder_triplet_cehn_{args.train_on}_{args.mine_from}_log.json")]:
        log_path = LOG_DIR / log_name
        if log_path.exists():
            with open(log_path) as f:
                data = json.load(f)
            best = max(ep["val_f1"] for ep in data)
            print(f"  {label:<20s}: val_f1={best:.4f}")
    print(f"\n日志 → {LOG_DIR / f'biencoder_triplet_cehn_{args.train_on}_{args.mine_from}_log.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="CrossEncoder Hard Negative Mining")
    p.add_argument("--bert_path",   default=BERT_PATH)
    p.add_argument("--train_on",    default="afqmc", choices=["afqmc", "bq_corpus", "lcqmc"])
    p.add_argument("--mine_from",   default="afqmc", choices=["afqmc", "bq_corpus", "lcqmc"])
    p.add_argument("--bi_top_k",    default=50, type=int, help="BiEncoder 粗筛候选数")
    p.add_argument("--epochs",      default=3, type=int)
    p.add_argument("--batch_size",  default=32, type=int)
    p.add_argument("--max_length",  default=64, type=int)
    p.add_argument("--lr",          default=2e-5, type=float)
    p.add_argument("--margin",      default=0.3, type=float)
    p.add_argument("--grad_accum",  default=1, type=int)
    return p.parse_args()


if __name__ == "__main__":
    main()
