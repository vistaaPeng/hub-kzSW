"""
两阶段检索演示：BiEncoder 召回 → CrossEncoder 精排

教学重点：
  1. 工业标准范式：第一阶段（BiEncoder ANN）快速，第二阶段（CrossEncoder）精准
  2. 用 val 集模拟检索场景——把每条 s1 当作查询，s2 池当作文档库
  3. 两套评估：(A) pair classification 直接对比标准 eval；
     (B) 检索指标（MRR / Recall@K）衡量精排的真实提升

前提：已训练好 BiEncoder Cosine 和 CrossEncoder checkpoint

使用方式：
  python src/two_stage_retrieval.py
  python src/two_stage_retrieval.py --k 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import PairDataset, CrossEncoderDataset, load_jsonl
from model import build_biencoder, build_crossencoder

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "afqmc"  # default, overridable via --data_dir
BERT_PATH = "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
LOG_DIR   = ROOT / "outputs" / "logs"
FIG_DIR   = ROOT / "outputs" / "figures"


@torch.no_grad()
def encode_corpus(model, texts, tokenizer, device, max_length=64):
    """批量编码文本池 → 句向量矩阵 [N, H]"""
    vectors = []
    for text in tqdm(texts, desc="编码文档库"):
        enc = tokenizer(text, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        vec = model.encode(**enc)  # [1, H]
        vectors.append(vec.cpu().numpy())
    return np.concatenate(vectors, axis=0)  # [N, H]


@torch.no_grad()
def rerank_with_crossencoder(model, tokenizer, query, candidates, device, max_length=128):
    """用 CrossEncoder 批量精排——一次前向处理全部候选"""
    s1_list = [query] * len(candidates)
    enc = tokenizer(s1_list, candidates, max_length=max_length, truncation=True,
                    padding="max_length", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc)          # [N, 2]
    probs = F.softmax(logits, dim=-1)[:, 1].tolist()  # [N]
    return probs


@torch.no_grad()
def classify_pairs_with_crossencoder(model, tokenizer, s1_list, s2_list, device,
                                     max_length=128, batch_size=64):
    """CrossEncoder 直接对 (s1, s2) 做 pair classification，返回预测标签。"""
    all_preds = []
    for i in range(0, len(s1_list), batch_size):
        batch_s1 = s1_list[i:i + batch_size]
        batch_s2 = s2_list[i:i + batch_size]
        enc = tokenizer(batch_s1, batch_s2, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
    return all_preds


def compute_retrieval_metrics(ranks, k_values=(1, 5, 10, 20, 50, 100)):
    """从 gold_s2 的 rank 列表计算检索指标（1-indexed rank，rank=1 表示命中）。"""
    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    recall_at_k = {}
    for k in k_values:
        recall_at_k[k] = float(np.mean(ranks <= k))
    return {"mrr": mrr, "recall_at_k": recall_at_k}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── 加载 checkpoint ─────────────────────────────────────────────────
    bi_ckpt_path = CKPT_DIR / "biencoder_cosine_best.pt"
    ce_ckpt_path = CKPT_DIR / "crossencoder_best.pt"
    if not bi_ckpt_path.exists():
        print(f"❌ BiEncoder checkpoint 不存在: {bi_ckpt_path}")
        print("  请先运行: python src/train_biencoder.py --loss cosine")
        return
    if not ce_ckpt_path.exists():
        print(f"❌ CrossEncoder checkpoint 不存在: {ce_ckpt_path}")
        print("  请先运行: python src/train_crossencoder.py")
        return

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    bi_ckpt = torch.load(bi_ckpt_path, map_location=device, weights_only=False)
    bi_model = build_biencoder(BERT_PATH, pool=bi_ckpt.get("args", {}).get("pool", "mean"),
                               num_hidden_layers=bi_ckpt.get("args", {}).get("num_hidden_layers")).to(device)
    bi_model.load_state_dict(bi_ckpt["state_dict"])
    bi_model.eval()

    ce_ckpt = torch.load(ce_ckpt_path, map_location=device, weights_only=False)
    ce_model = build_crossencoder(BERT_PATH,
                                  num_hidden_layers=ce_ckpt.get("args", {}).get("num_hidden_layers")).to(device)
    ce_model.load_state_dict(ce_ckpt["state_dict"])
    ce_model.eval()

    # ── 加载 val 数据 ───────────────────────────────────────────────────
    data_dir  = Path(args.data_dir) if args.data_dir else DATA_DIR
    val_path = data_dir / "validation.jsonl"
    rows = load_jsonl(val_path)
    print(f"Val 数据: {len(rows):,} 对")

    # 把 val 的所有 s2 当作"文档库"，所有 s1 当作"查询"
    all_s2 = list(set(r["sentence2"] for r in rows))
    all_s1 = [r["sentence1"] for r in rows]
    labels  = [r["label"]     for r in rows]
    gold_s2_list = [r["sentence2"] for r in rows]
    s2_to_idx = {s2: i for i, s2 in enumerate(all_s2)}
    print(f"文档库: {len(all_s2):,} 条唯一句子  查询: {len(all_s1):,} 条")

    # ── 阶段 1: BiEncoder 编码 + 检索 ──────────────────────────────────
    print("\n[阶段 1] BiEncoder 编码文档库...")
    t0 = time.time()
    corpus_vecs = encode_corpus(bi_model, all_s2, tokenizer, device)
    corpus_vecs = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-9)
    print(f"  编码耗时: {time.time()-t0:.1f}s  shape: {corpus_vecs.shape}")

    print("\n[阶段 1] BiEncoder 检索...")
    bi_predictions = []   # (top_k_s2, top_k_scores) per query
    bi_sims = []          # gold_s2 cosine similarity per query
    for s1, gold_s2 in tqdm(zip(all_s1, gold_s2_list), total=len(rows), desc="检索"):
        enc = tokenizer(s1, max_length=64, truncation=True,
                        padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        q_vec = bi_model.encode(**enc).detach().cpu().numpy()  # [1, H]
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)

        dot_scores = (q_vec @ corpus_vecs.T)[0]  # [N]
        top_k_idx = np.argsort(dot_scores)[-args.k:][::-1]
        top_k_s2 = [all_s2[j] for j in top_k_idx]
        top_k_scores = dot_scores[top_k_idx]

        gold_idx = s2_to_idx.get(gold_s2)
        gold_sim = float(dot_scores[gold_idx]) if gold_idx is not None else 0.0
        bi_sims.append(gold_sim)
        bi_predictions.append((top_k_s2, top_k_scores))

    # ── 评估 A: BiEncoder pair classification ──────────────────────────
    bi_sims_np = np.array(bi_sims)
    labels_np = np.array(labels)
    best_bi_f1, best_bi_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (bi_sims_np >= t).astype(int)
        f1 = f1_score(labels_np, preds, average="weighted", zero_division=0)
        if f1 > best_bi_f1:
            best_bi_f1 = f1
            best_bi_thresh = t
    bi_preds = (bi_sims_np >= best_bi_thresh).astype(int)
    bi_acc = accuracy_score(labels_np, bi_preds)
    bi_f1  = f1_score(labels_np, bi_preds, average="weighted", zero_division=0)

    # ── 评估 B: CrossEncoder 直接 pair classification ─────────────────
    print(f"\n[评估 B] CrossEncoder 直接 pair classification（与标准 eval 可比）...")
    t0 = time.time()
    ce_direct_preds = classify_pairs_with_crossencoder(
        ce_model, tokenizer, all_s1, gold_s2_list, device)
    ce_direct_acc = accuracy_score(labels_np, ce_direct_preds)
    ce_direct_f1  = f1_score(labels_np, ce_direct_preds, average="weighted", zero_division=0)
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"  CrossEncoder 直接分类: Acc={ce_direct_acc:.4f}  F1={ce_direct_f1:.4f}")

    # ── 评估 C: 检索指标（仅正样本，精排效果的正确衡量）───────────────
    print(f"\n[评估 C] 检索指标 — CrossEncoder 精排 Top-{args.k}（仅正样本）...")

    pos_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    print(f"  正样本查询: {len(pos_indices):,} 条")

    bi_ranks = []   # BiEncoder ranking 中 gold_s2 的 rank
    ce_ranks = []   # CrossEncoder reranking 中 gold_s2 的 rank

    for i in tqdm(pos_indices, desc="精排评估"):
        s1, gold_s2 = all_s1[i], gold_s2_list[i]
        top_k_s2, top_k_scores = bi_predictions[i]

        # 构建候选池：top-K + gold_s2（若不在 top-K 中）
        pool_s2 = list(top_k_s2)
        pool_bi_scores = list(top_k_scores)
        if gold_s2 not in pool_s2:
            pool_s2.append(gold_s2)
            # gold_s2 不在 top-K 中，补算它的 BiEncoder 分数
            gold_idx = s2_to_idx[gold_s2]
            enc = tokenizer(s1, max_length=64, truncation=True,
                            padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            q_vec = bi_model.encode(**enc).detach().cpu().numpy()
            q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
            gold_vec = corpus_vecs[gold_idx:gold_idx + 1]
            gold_bi_score = (q_vec @ gold_vec.T).item()
            pool_bi_scores.append(gold_bi_score)

        # BiEncoder rank: 在候选池中按 BiEncoder 分数排序，找 gold_s2 的排名
        bi_order = np.argsort(pool_bi_scores)[::-1]
        bi_rank = int(np.where(bi_order == pool_s2.index(gold_s2))[0][0]) + 1  # 1-indexed
        bi_ranks.append(bi_rank)

        # CrossEncoder rank: 在候选池中按 CrossEncoder 分数排序
        ce_scores = rerank_with_crossencoder(ce_model, tokenizer, s1, pool_s2, device)
        ce_order = np.argsort(ce_scores)[::-1]
        ce_rank = int(np.where(ce_order == pool_s2.index(gold_s2))[0][0]) + 1
        ce_ranks.append(ce_rank)

    bi_metrics = compute_retrieval_metrics(bi_ranks)
    ce_metrics = compute_retrieval_metrics(ce_ranks)

    # ── 汇总输出 ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  两阶段检索 — 完整评估（K = {args.k}）")
    print(f"{'='*60}")

    print(f"\n  ── A. Pair Classification（与标准 eval 可比）──")
    print(f"  {'':<30} {'Accuracy':>10} {'F1(weighted)':>13}")
    print(f"  {'BiEncoder (cosine + threshold)':<30} {bi_acc:>10.4f} {bi_f1:>13.4f}")
    print(f"  {'CrossEncoder (直接分类)':<30} {ce_direct_acc:>10.4f} {ce_direct_f1:>13.4f}")
    delta_cls_f1 = ce_direct_f1 - bi_f1
    print(f"  {'Δ (CE − Bi)':<30} {'':>10} {delta_cls_f1:>+13.4f}")

    print(f"\n  ── B. Retrieval 指标（{len(pos_indices):,} 正样本查询）──")
    print(f"  {'指标':<18} {'BiEncoder':>12} {'Bi+CE 精排':>14} {'提升':>10}")
    print(f"  {'─'*18} {'─'*12} {'─'*14} {'─'*10}")
    print(f"  {'MRR':<18} {bi_metrics['mrr']:>12.4f} {ce_metrics['mrr']:>14.4f}"
          f"  {ce_metrics['mrr'] - bi_metrics['mrr']:>+10.4f}")
    for k in [1, 5, 10, 20, 50, 100]:
        bi_r = bi_metrics["recall_at_k"].get(k, 0)
        ce_r = ce_metrics["recall_at_k"].get(k, 0)
        print(f"  {f'Recall@{k}':<18} {bi_r:>12.4f} {ce_r:>14.4f}"
              f"  {ce_r - bi_r:>+10.4f}")

    # ── 保存 ───────────────────────────────────────────────────────────
    log_path = LOG_DIR / f"two_stage_{data_dir.name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "k": args.k,
            "n_val": len(rows),
            "n_pos_queries": len(pos_indices),
            "pair_classification": {
                "biencoder": {
                    "accuracy": round(bi_acc, 4),
                    "f1_weighted": round(bi_f1, 4),
                    "threshold": round(float(best_bi_thresh), 4),
                },
                "crossencoder": {
                    "accuracy": round(ce_direct_acc, 4),
                    "f1_weighted": round(ce_direct_f1, 4),
                },
            },
            "retrieval": {
                "biencoder": {
                    "mrr": round(bi_metrics["mrr"], 4),
                    "recall_at_k": {str(k): round(v, 4) for k, v in bi_metrics["recall_at_k"].items()},
                },
                "crossencoder_rerank": {
                    "mrr": round(ce_metrics["mrr"], 4),
                    "recall_at_k": {str(k): round(v, 4) for k, v in ce_metrics["recall_at_k"].items()},
                },
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="BiEncoder 召回 + CrossEncoder 精排")
    p.add_argument("--k", default=100, type=int, help="BiEncoder 召回数量（默认 100）")
    p.add_argument("--data_dir", default=None, type=str,
                   help="数据目录（默认: data/afqmc）")
    return p.parse_args()


if __name__ == "__main__":
    main()
