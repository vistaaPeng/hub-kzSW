"""
单独测试 m24 (Clone vs Copy) 的 Faithfulness

用法: python tests/test_single_m24.py
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrievers.vector_store import VectorStore
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid_retriever import HybridRetriever
from src.llm.generator import RAGGenerator
from src.llm.query_rewriter import QueryRewriter
from src.chunkers.narrative import Chunk
from evaluation.evaluator import RAGEvaluator

INDEX_DIR = Path("vectorstore")
Q = "Rust 中 Clone 和 Copy trait 的区别是什么？哪些类型实现了 Copy？"

# 加载
vs = VectorStore()
vs.load(str(INDEX_DIR / "faiss.index"), str(INDEX_DIR / "children.json"))
bm25 = BM25Store()
bm25.load(str(INDEX_DIR / "bm25.pkl"))
retriever = HybridRetriever(vs, bm25)
all_raw = json.loads((INDEX_DIR / "all_chunks.json").read_text("utf-8"))
all_chunks = [Chunk(d["chunk_id"], d["text"], d["metadata"], d.get("parent_chunk_id"), d.get("is_parent", False)) for d in all_raw]
generator = RAGGenerator()
rewriter = QueryRewriter()
evaluator = RAGEvaluator(retriever, generator, top_k=10)

# 查询重写
queries = rewriter.rewrite(Q, n_variants=3)
print(f"重写: {queries}\n")

# 多路检索
seen = {}
for q in queries:
    results = retriever.search(q, top_k=10)
    for chunk, score in results:
        url = chunk.metadata.get("source_url", chunk.chunk_id)
        if url not in seen or score > seen[url][1]:
            seen[url] = (chunk, score)
merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:10]
parent_results = retriever.expand_to_parents(merged, all_chunks)

print("检索结果 (top 5):")
for i, (c, s) in enumerate(parent_results[:5]):
    print(f"  [{i+1}] score={s:.4f} | {c.metadata.get('headings','')[:60]}")
    print(f"       {c.metadata.get('source_url','')}")

# 生成
t0 = time.time()
parent_chunks = [c for c, _ in parent_results[:8]]
answer = generator.generate(Q, parent_chunks, max_chunks=8)
elapsed = time.time() - t0
print(f"\n生成 ({elapsed:.1f}s):")
print(answer[:500])

# 评估 Faithfulness
print("\n--- Faithfulness 评估 ---")
qtype = evaluator._classify_question_type(Q)
print(f"问题类型: {qtype}")
ff_score, ff_detail = evaluator.compute_faithfulness(Q, answer, parent_chunks[:5])
print(f"Faithfulness: {ff_score}")
if ff_detail.get("claims"):
    for c in ff_detail["claims"][:5]:
        print(f"  {'✅' if c['supported'] else '❌'} {c['statement'][:80]}")
