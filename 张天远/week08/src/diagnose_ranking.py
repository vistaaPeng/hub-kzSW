"""
诊断：两阶段检索的评测协议不一致问题
对比 BiEncoder 在 "threshold" vs "rank#1" 两种协议下的 F1
"""
import json, time, argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score

from model import build_biencoder
from dataset import load_jsonl

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
BERT_PATH = "bert-base-chinese"

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/bq_corpus")
    p.add_argument("--k", default=100, type=int)
    p.add_argument("--ckpt_name", default="biencoder_cosine_best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  |  K={args.k}  |  data={args.data_dir}")

    # 加载 checkpoint
    data_dir = ROOT / args.data_dir
    ckpt_path = CKPT_DIR / args.ckpt_name
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_biencoder(BERT_PATH,
                            pool=ckpt.get("args", {}).get("pool", "mean"),
                            num_hidden_layers=ckpt.get("args", {}).get("num_hidden_layers")).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    # 加载数据
    rows = load_jsonl(data_dir / "validation.jsonl")
    all_s2 = list(set(r["sentence2"] for r in rows))
    all_s1 = [r["sentence1"] for r in rows]
    labels  = [r["label"]     for r in rows]
    print(f"文档库: {len(all_s2)}  查询: {len(all_s1)}")

    # 编码文档库
    corpus_vecs = []
    for text in tqdm(all_s2, desc="编码文档库"):
        enc = tokenizer(text, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        vec = model.encode(**enc).cpu().numpy()
        corpus_vecs.append(vec)
    corpus_vecs = np.concatenate(corpus_vecs, axis=0)
    corpus_vecs = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-9)
    s2_to_idx = {s2: i for i, s2 in enumerate(all_s2)}

    # 检索并统计排名
    sims_list = []     # gold_s2 直接余弦相似度
    ranks_list = []    # gold_s2 在 Top-K 中的排名 (1-indexed, 0=未进Top-K)
    recall_in_k = 0

    for s1, gold_s2 in tqdm(zip(all_s1, [r["sentence2"] for r in rows]), total=len(all_s1), desc="检索"):
        enc = tokenizer(s1, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        q_vec = model.encode(**enc).detach().cpu().numpy()
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
        dot_scores = (q_vec @ corpus_vecs.T)[0]
        gold_idx = s2_to_idx.get(gold_s2)
        gold_sim = float(dot_scores[gold_idx]) if gold_idx is not None else 0.0
        sims_list.append(gold_sim)

        # Top-K 索引 (降序)
        top_k_idx = np.argsort(dot_scores)[-args.k:][::-1]
        if gold_idx in top_k_idx:
            recall_in_k += 1
            rank = np.where(top_k_idx == gold_idx)[0][0] + 1  # 1-indexed
        else:
            rank = 0  # 未进 Top-K
        ranks_list.append(rank)

    sims = np.array(sims_list)
    labels_np = np.array(labels)

    # === 协议 A: Threshold-based (原 BiEncoder 评测) ===
    best_f1, best_thr = -1, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels_np, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    preds_a = (sims >= best_thr).astype(int)
    acc_a = accuracy_score(labels_np, preds_a)
    f1_a = f1_score(labels_np, preds_a, average="weighted", zero_division=0)

    # === 协议 B: Rank#1 (与两阶段相同的协议) ===
    ranks = np.array(ranks_list)
    preds_b = (ranks == 1).astype(int)  # rank==1 → predict match
    acc_b = accuracy_score(labels_np, preds_b)
    f1_b = f1_score(labels_np, preds_b, average="weighted", zero_division=0)

    # === 统计排名分布 ===
    rank_dist = {}
    for r in ranks_list:
        bucket = r if r <= 5 else (6 if r <= 10 else (11 if r <= 50 else (51 if r <= 100 else 0)))
        rank_dist[bucket] = rank_dist.get(bucket, 0) + 1

    print(f"\n{'='*60}")
    print(f"评测协议对比")
    print(f"{'='*60}")
    print(f"  协议 A (threshold) : Acc={acc_a:.4f}  F1={f1_a:.4f}  thr={best_thr:.2f}")
    print(f"  协议 B (rank#1)    : Acc={acc_b:.4f}  F1={f1_b:.4f}")
    print(f"  Δ (B - A)          : Acc={acc_b-acc_a:+.4f}  F1={f1_b-f1_a:+.4f}")
    print(f"\n  Recall@{args.k} = {recall_in_k/len(all_s1):.4f} ({recall_in_k}/{len(all_s1)})")
    print(f"\n  gold_s2 在 Top-K 中的排名分布:")
    for bucket in sorted(rank_dist.keys()):
        count = rank_dist[bucket]
        pct = count / len(all_s1) * 100
        if bucket == 0:
            label = f"  未进Top-{args.k}"
        elif bucket <= 5:
            label = f"  排名 {bucket}"
        elif bucket == 6:
            label = "  排名 6-10"
        elif bucket == 11:
            label = "  排名 11-50"
        elif bucket == 51:
            label = f"  排名 51-{args.k}"
        print(f"    {label:20s}: {count:6d} ({pct:5.1f}%)")

    # 保存
    out = {
        "data": args.data_dir,
        "k": args.k,
        "protocol_a_threshold": {"acc": round(float(acc_a), 4), "f1": round(float(f1_a), 4), "threshold": round(float(best_thr), 4)},
        "protocol_b_rank1":    {"acc": round(float(acc_b), 4), "f1": round(float(f1_b), 4)},
        "recall_at_k": round(recall_in_k / len(all_s1), 4),
        "rank_distribution": {str(k): v for k, v in sorted(rank_dist.items())},
    }
    out_path = ROOT / "outputs" / "logs" / f"diagnose_{data_dir.name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果 → {out_path}")

if __name__ == "__main__":
    main()
