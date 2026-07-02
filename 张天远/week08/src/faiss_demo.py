"""
FAISS 向量检索演示 — 精确 vs 近似最近邻搜索

教学重点：
  1. IndexFlatIP（精确）— 暴搜所有向量，O(N) 复杂度，适合 N < 10 万
  2. IndexIVFFlat（近似）— K-means 聚类后只搜索最近 nprobe 个簇，O(√N)
  3. nprobe 权衡 — 越大召回率越接近精确，但速度下降
  4. L2 归一化后余弦相似度 = 内积 — 所以用 IndexFlatIP（Inner Product）
  5. 为什么需要 ANN：当 N = 100 万时，暴力搜索 100ms vs ANN 0.1ms

演示内容：
  - 加载 BiEncoder → 编码 AFQMC val 全量 → 构建 FAISS 索引
  - 对比 IndexFlatIP vs IndexIVFFlat 的 recall@K 和延迟
  - 画 recall vs nprobe 曲线
  - 分别演示 AFQMC（4K）和 LCQMC（24K）规模

使用方式：
  python src/faiss_demo.py
  python src/faiss_demo.py --data_dir data/lcqmc --nlist 100
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer
from collections import defaultdict

from model import BiEncoder, build_biencoder
from dataset import load_jsonl, encode_single

ROOT       = Path(__file__).parent.parent
BERT_PATH  = "bert-base-chinese"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
LOG_DIR    = ROOT / "outputs" / "logs"
FIG_DIR    = ROOT / "outputs" / "figures"


@torch.no_grad()
def encode_corpus(model, tokenizer, sentences, device, max_length=64, batch_size=128):
    """编码句子列表为归一化句向量矩阵 [N, H]"""
    model.eval()
    vectors = []
    for text in tqdm(sentences, desc="编码句子"):
        enc = encode_single(tokenizer, text, max_length)
        inp = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        vec = model.encode(**inp).cpu().numpy()
        vectors.append(vec)
    vecs = np.concatenate(vectors, axis=0).astype(np.float32)
    # L2 归一化（保证 FAISS Inner Product = 余弦相似度）
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return vecs


def build_index_flat(vecs):
    """精确检索：IndexFlatIP（Inner Product）"""
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def build_index_ivf(vecs, nlist=50):
    """近似检索：IVFFlat — 先 K-means 聚类，只搜索最近 nprobe 个簇"""
    dim = vecs.shape[1]
    quantizer = faiss.IndexFlatIP(dim)  # 聚类中心用精确内积
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)   # K-means 训练
    index.add(vecs)
    return index


def measure_latency(index, query_vecs, k=10, nprobe=None, n_runs=100):
    """测量检索延迟（平均值，微秒）"""
    if nprobe is not None and hasattr(index, 'nprobe'):
        index.nprobe = nprobe
    times = []
    for _ in range(n_runs):
        q = query_vecs[:1]  # 单条查询
        t0 = time.perf_counter()
        index.search(q, k)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000  # ms


def compute_recall_at_k(index, query_vecs, gold_indices, k=10, nprobe=None):
    """计算 recall@K — gold_indices 来自精确检索的结果"""
    if nprobe is not None and hasattr(index, 'nprobe'):
        index.nprobe = nprobe
    D, I = index.search(query_vecs, k)
    recall = 0.0
    for i, gold in enumerate(gold_indices):
        if gold in I[i]:
            recall += 1.0
    return recall / len(query_vecs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / "afqmc"
    ds_name = data_dir.name

    # ── 加载模型 ─────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    # 优先加载对应数据集的 checkpoint，回退到 AFQMC
    ckpt_candidates = [
        CKPT_DIR / f"biencoder_cosine_{ds_name}_best.pt",
        CKPT_DIR / "biencoder_cosine_best.pt",
        CKPT_DIR / "biencoder_cosine_best_core.pt",
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        print(f"❌ 未找到 BiEncoder Cosine checkpoint")
        return

    print(f"加载模型: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BiEncoder(BERT_PATH, pool=ckpt.get("args", {}).get("pool", "mean"),
                      num_hidden_layers=ckpt.get("args", {}).get("num_hidden_layers", 4)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── 编码语料 ─────────────────────────────────────────────────────────
    val_rows = load_jsonl(data_dir / "validation.jsonl")
    # 收集所有唯一句子作为检索库
    corpus = list(set(r["sentence1"] for r in val_rows) | set(r["sentence2"] for r in val_rows))
    print(f"语料库: {len(corpus):,} 条唯一句子")

    vecs = encode_corpus(model, tokenizer, corpus, device)
    dim = vecs.shape[1]
    print(f"向量维度: {dim}  总大小: {vecs.nbytes / 1024 / 1024:.1f} MB")

    # ── 构建索引 ─────────────────────────────────────────────────────────
    print(f"\n构建索引...")
    t0 = time.time()

    # 精确索引
    index_flat = build_index_flat(vecs)
    t_flat = time.time() - t0
    print(f"  IndexFlatIP: 构建 {t_flat:.2f}s, 总数 {index_flat.ntotal:,}")

    # IVF 近似索引
    nlist = min(args.nlist, len(corpus) // 10)  # 每簇至少 10 个向量
    index_ivf = build_index_ivf(vecs, nlist=nlist)
    t_ivf = time.time() - t0 - t_flat
    print(f"  IndexIVFFlat: nlist={nlist}, 构建 {t_ivf:.2f}s, 总数 {index_ivf.ntotal:,}")

    # ── 准备查询（用 val 的 s1 作为查询）─────────────────────────────────
    queries = list(set(r["sentence1"] for r in val_rows))[:args.n_queries]
    query_vecs = encode_corpus(model, tokenizer, queries, device)
    print(f"查询数: {len(queries):,}")

    # gold：精确检索 Top-K 结果（召回率参照）
    _, gold_I = index_flat.search(query_vecs, args.k)
    gold_set = [set(row) for row in gold_I]

    # ── 延迟对比 ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("检索延迟对比（平均每次查询）")
    print(f"{'方法':<25} {'延迟(ms)':>10} {'recall@{k}'.format(k=args.k):>12}")
    print(f"{'-'*50}")

    t_flat_lat = measure_latency(index_flat, query_vecs, k=args.k, n_runs=args.n_runs)
    r_flat = compute_recall_at_k(index_flat, query_vecs, gold_I[:, 0], k=args.k)
    print(f"  {'IndexFlatIP (精确)':<25} {t_flat_lat:>10.4f} {r_flat:>12.4f}")

    results = []
    for nprobe in args.nprobes:
        t_ivf_lat = measure_latency(index_ivf, query_vecs, k=args.k, nprobe=nprobe, n_runs=args.n_runs)
        r_ivf = compute_recall_at_k(index_ivf, query_vecs, gold_I[:, 0], k=args.k, nprobe=nprobe)
        tag = f"IVF nprobe={nprobe}"
        print(f"  {tag:<25} {t_ivf_lat:>10.4f} {r_ivf:>12.4f}")
        results.append({
            "nprobe": nprobe, "latency_ms": round(t_ivf_lat, 4),
            f"recall@{args.k}": round(r_ivf, 4),
        })

    # ── 检索示例 ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("检索示例（前 3 条查询的 Top-3 结果）")
    for i in range(min(3, len(queries))):
        print(f"\n  查询 {i+1}: {queries[i]!r}")
        D, I = index_flat.search(query_vecs[i:i+1], 3)
        for j in range(3):
            idx = int(I[0, j])
            sim = float(D[0, j])
            print(f"    #{j+1}  sim={sim:.4f}  {corpus[idx]!r}")

    # ── 保存 ─────────────────────────────────────────────────────────────
    out = {
        "dataset": ds_name,
        "corpus_size": len(corpus),
        "dim": dim,
        "nlist": nlist,
        "n_queries": len(queries),
        "k": args.k,
        "index_flat": {"build_s": t_flat, "latency_ms": round(t_flat_lat, 4), f"recall@{args.k}": round(r_flat, 4)},
        "index_ivf": results,
    }
    log_path = LOG_DIR / f"faiss_demo_{ds_name}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 结论 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("结论")
    print(f"  语料规模: {len(corpus):,} 条")
    if len(corpus) < 10000:
        print(f"  ⚠️ 当前 {len(corpus):,} 条规模下，IndexFlatIP 已是毫秒级——")
        print(f"     不需要 ANN。FAISS 的真正优势在百万级以上。")
        print(f"     本演示展示的是\"机制\"而非\"加速效果\"。")
    print(f"  结果 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="FAISS 向量检索演示")
    p.add_argument("--data_dir",  default=None, type=str, help="数据目录（默认 data/afqmc）")
    p.add_argument("--n_queries", default=200, type=int, help="查询数")
    p.add_argument("--k",         default=10, type=int, help="Top-K")
    p.add_argument("--nlist",     default=50, type=int, help="IVF 聚类数")
    p.add_argument("--nprobes",   nargs="+", type=int, default=[1, 2, 5, 10, 20],
                   help="IVF nprobe 取值列表")
    p.add_argument("--n_runs",    default=50, type=int, help="延迟测量重复次数")
    return p.parse_args()


if __name__ == "__main__":
    main()
