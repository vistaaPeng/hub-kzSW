#!/usr/bin/env python
"""RAG latency diagnosis - standalone"""
import sys, json, time, os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
PROJECT = Path(r"E:\npl\workspaces\npl_tran\rag_scratch")
sys.path.insert(0, str(PROJECT))
from src.retrievers.vector_store import VectorStore
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid_retriever import HybridRetriever
from src.llm.generator import RAGGenerator
from src.llm.query_rewriter import QueryRewriter
from src.glossary import search_glossary
from evaluation.evaluator import EvalQuestion
INDEX_DIR = PROJECT / "vectorstore"
idx_path = INDEX_DIR / "glossary" / "glossary_faiss.index"
has_glossary_idx = idx_path.exists()

qs = [
    ("m01", "Rust 中的所有权（ownership）是什么？有哪些核心规则？"),
    ("m11", "Rust 中 String 和 &str 的区别是什么？何时使用哪个？"),
    ("m24", "Rust 中 Clone 和 Copy trait 的区别是什么？哪些类型实现了 Copy？"),
    ("m26", "Rust 中什么是函数指针？fn 类型和 Fn trait 有什么区别？"),
    ("m36", "Rust 中什么是孤儿规则（orphan rule）？它如何限制 trait 实现？"),
]

print("Loading...")
vs = VectorStore(); vs.load(str(INDEX_DIR/"faiss.index"), str(INDEX_DIR/"children.json"))
bm25 = BM25Store(); bm25.load(str(INDEX_DIR/"bm25.pkl"))
retriever = HybridRetriever(vs, bm25)
all_chunks = json.loads((INDEX_DIR/"all_chunks.json").read_text(encoding="utf-8"))
from src.chunkers.narrative import Chunk
all_chunks = [Chunk(d["chunk_id"], d["text"], d["metadata"], d.get("parent_chunk_id"), d.get("is_parent", False)) for d in all_chunks]
generator = RAGGenerator()
rewriter = QueryRewriter()

stage_times = {"query_rewrite":[],"vector_search":[],"bm25_search":[],"parent_expansion":[],"glossary_search":[],"llm_generation":[]}
samples = []

for qid, qtext in qs:
    print(f"\n{qid}...")
    t0=time.time()
    queries = rewriter.rewrite(qtext, n_variants=3)
    t1=time.time()
    stage_times["query_rewrite"].append(t1-t0)
    
    ts=time.time()
    vs.search(qtext, top_k=60)
    te=time.time()
    stage_times["vector_search"].append(te-ts)
    
    ts=time.time()
    bm25.search(qtext, top_k=60)
    te=time.time()
    stage_times["bm25_search"].append(te-ts)
    
    seen={}
    for rq in queries:
        for chunk, score in retriever.search(rq, top_k=10):
            url=chunk.metadata.get("source_url",chunk.chunk_id)
            if url not in seen or score>seen[url][1]: seen[url]=(chunk,score)
    merged=sorted(seen.values(),key=lambda x:x[1],reverse=True)[:10]
    
    ts=time.time()
    parent_results = retriever.expand_to_parents(merged, all_chunks)
    te=time.time()
    stage_times["parent_expansion"].append(te-ts)
    parent_chunks=[c for c,_ in parent_results[:8]]
    
    if has_glossary_idx:
        ts=time.time()
        # Workaround: glossary.py search_glossary missing Path import
        import src.glossary as _gl
        if not hasattr(_gl, 'Path'):
            from pathlib import Path; _gl.Path = Path
        gt = _gl.search_glossary(qtext, top_k=10)
        te=time.time()
        stage_times["glossary_search"].append(te-ts)
    else:
        gt = []
        stage_times["glossary_search"].append(0)
    
    ts=time.time()
    answer = generator.generate(qtext, parent_chunks, max_chunks=8, glossary_terms=gt)
    te=time.time()
    stage_times["llm_generation"].append(te-ts)
    
    total = time.time()-t0
    entry={"question_id":qid,"total_s":round(total,3),
           "stages":{"query_rewrite_s":round(t1-t0,3),"vector_search_s":round(stage_times["vector_search"][-1],4),
                     "bm25_search_s":round(stage_times["bm25_search"][-1],4),
                     "parent_expansion_s":round(stage_times["parent_expansion"][-1],4),
                     "glossary_search_s":round(stage_times["glossary_search"][-1],4),
                     "llm_generation_s":round(stage_times["llm_generation"][-1],3)}}
    samples.append(entry)
    bottleneck = max(entry["stages"], key=entry["stages"].get)
    print(f"  total={total:.2f}s  bottleneck={bottleneck}={entry['stages'][bottleneck]:.3f}s")

from statistics import median
def pct(data, p):
    sd=sorted(data); k=(len(sd)-1)*p; f=int(k); c=k-f
    return sd[f]*(1-c)+sd[f+1]*c if f+1<len(sd) else sd[f]

summary={}
for stage, times in stage_times.items():
    if times and any(t>0 for t in times):
        summary[stage]={"p50_s":round(median(times),4),"p95_s":round(pct(times,0.95),4),"min_s":round(min(times),4),"max_s":round(max(times),4)}

out={"meta":{"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"n_samples":len(qs)},"p50_p95_by_stage":summary,"samples":samples}
out_path=PROJECT/"tests"/"diagnose_results"/"diagnose_latency.json"
out_path.parent.mkdir(parents=True,exist_ok=True)
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nWritten to {out_path}")
for stage,v in sorted(summary.items(),key=lambda x:-x[1]["p95_s"]):
    print(f"  {stage:20s}  P50={v['p50_s']:.4f}s  P95={v['p95_s']:.4f}s  min={v['min_s']:.4f}s  max={v['max_s']:.4f}s")
