"""
诊断 m26 (函数指针) 检索召回差的根因

用法: python tests/diagnose_m26_retrieval.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrievers.vector_store import VectorStore
from src.retrievers.bm25_store import BM25Store
from src.chunkers.narrative import Chunk

INDEX_DIR = Path("vectorstore")
Q = "Rust 中什么是函数指针？fn 类型和 Fn trait 有什么区别？"

# 加载
vs = VectorStore()
vs.load(str(INDEX_DIR / "faiss.index"), str(INDEX_DIR / "children.json"))
bm25 = BM25Store()
bm25.load(str(INDEX_DIR / "bm25.pkl"))

# BM25 检索
bm25_results = vs if False else None
bm25_results = bm25.search(Q, top_k=20) or []
print("=== BM25 Top-20 ===")
for i, (c, s) in enumerate(bm25_results, 1):
    url = c.metadata.get("source_url", "")[-50:]
    head = c.metadata.get("headings", "")[:50]
    print(f"  BM25[{i}] score={s:.4f} | {head}")

# 向量检索
vec_results = vs.search(Q, top_k=20) or []
print("\n=== Vector Top-20 ===")
for i, (c, s) in enumerate(vec_results, 1):
    url = c.metadata.get("source_url", "")[-50:]
    head = c.metadata.get("headings", "")[:50]
    print(f"  VEC[{i}] score={s:.4f} | {head}")

# 找 "函数指针" 关键词在两边分别出现的位置
print("\n=== '函数指针' 命中分析 ===")
ql = Q.lower()
for label, results in [("BM25", bm25_results), ("Vector", vec_results)]:
    for rank, (c, s) in enumerate(results, 1):
        if "函数指针" in c.text or "fn" in c.metadata.get("headings","").lower():
            src = c.metadata.get("source_url","")[-60:]
            print(f"  {label}[{rank}] '{c.metadata.get('headings','')[:60]}' | {src}")

# RRF 模拟
K = 60
rrf_scores = {}
for rank, (c, _) in enumerate(bm25_results, 1):
    url = c.metadata.get("source_url", c.chunk_id)
    rrf_scores[url] = rrf_scores.get(url, 0) + 1.0 / (K + rank)
for rank, (c, _) in enumerate(vec_results, 1):
    url = c.metadata.get("source_url", c.chunk_id)
    rrf_scores[url] = rrf_scores.get(url, 0) + 1.0 / (K + rank)

sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n=== RRF Top-5 ===")
for url, score in sorted_rrf:
    print(f"  {score:.4f} | {url[-60:]}")
