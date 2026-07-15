#!/usr/bin/env python
"""
交互式 RAG 查询 —— LLM 查询重写 + 多路检索合并 + 父子块扩展 + LLM 生成
---
用法: python scripts/query.py [--rerank] [--no-rewrite]
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.generator import RAGGenerator
from src.reranker.reranker import ReRanker
from src.pipeline import RAGPipeline

INDEX_DIR = Path("vectorstore")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG 交互式查询")
    parser.add_argument("--rerank", action="store_true", help="启用 Cross-Encoder 重排")
    parser.add_argument("--no-rewrite", action="store_true", help="禁用查询重写")
    args = parser.parse_args()

    if not (INDEX_DIR / "faiss.index").exists():
        print("❌ 索引未构建。请先运行: python scripts/build_index.py")
        return

    print("🔧 加载索引...")
    generator = RAGGenerator()
    reranker = ReRanker(enabled=args.rerank) if args.rerank else None
    pipeline = RAGPipeline.from_index(
        INDEX_DIR,
        generator=generator,
        rewrite=not args.no_rewrite,
        reranker=reranker,
    )

    rewrite_status = "✅ 已启用" if pipeline.rewriter else "❌ 未启用"
    rerank_status = "✅ 已启用" if args.rerank else "❌ 未启用"
    print(f"✅ 就绪 | 查询重写: {rewrite_status} | 重排: {rerank_status} | chunks: {len(pipeline.all_chunks)}")
    print("输入问题开始查询，输入 q 退出\n")

    while True:
        try:
            query = input("🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            break

        result = pipeline.retrieve(query, top_k=10, rerank=args.rerank)
        if len(result.queries) > 1:
            print(f"🔄 重写: {result.queries}")

        if not result.child_results:
            print("❌ 未找到相关文档\n")
            continue

        print(f"\n📚 检索到 {len(result.parent_results)} 个相关文档:")
        for i, (chunk, score) in enumerate(result.parent_results, 1):
            src = chunk.metadata.get("source_name", "?")
            headings = chunk.metadata.get("headings", "")[:50]
            n_sib = chunk.metadata.get("sibling_count", 1)
            source = chunk.metadata.get("source_url", "")[-60:]
            sib_info = f"({n_sib}块拼接)" if n_sib > 1 else ""
            print(f"  [{i}] [{src}] {sib_info} score={score:.3f}")
            print(f"      📄 {source}")
            print(f"      📍 {headings}")

        # LLM 生成
        print("\n🤖 生成回答...")
        t0 = time.time()
        result = pipeline.generate_answer(result, max_chunks=8)
        answer = result.answer
        elapsed = time.time() - t0
        print(f"\n{answer}\n")

        # 结构化日志（JSONL，可追溯）
        log_entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "rewrites": result.rewrites,
            "child_count": len(result.child_results),
            "parent_count": len(result.parent_results),
            "retrieved": [
                {"rank": i+1, "chunk_id": c.chunk_id,
                 "score": round(s, 4),
                 "source": c.metadata.get("source_url", ""),
                 "headings": c.metadata.get("headings", ""),
                 "siblings": c.metadata.get("sibling_count", 1)}
                for i, (c, s) in enumerate(result.parent_results[:8])
            ],
            "answer": answer[:500],
            "latency_s": round(elapsed, 2),
        }
        log_path = Path("logs/queries.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
