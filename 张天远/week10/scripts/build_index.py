#!/usr/bin/env python
"""
构建 RAG 索引 —— 从 parsed JSON → 分块 → 向量/BM25 索引
---
前置: python scripts/parse.py（HTML → JSON）
用法: python scripts/build_index.py [--sources book,reference,...]
注意: 更改分块策略后只需重跑本脚本，不需要重新解析 HTML
"""

import sys
import json
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.narrative import NarrativeChunker
from src.retrievers.vector_store import VectorStore
from src.retrievers.bm25_store import BM25Store


def build_all(data_dir: str = "data", sources: list[str] | None = None):
    data = Path(data_dir)
    parsed_dir = data / "parsed"

    if not parsed_dir.exists() or not list(parsed_dir.iterdir()):
        print("❌ 没有 parsed JSON。请先运行: python scripts/parse.py")
        return

    all_chunks = []

    source_dirs = [d for d in parsed_dir.iterdir() if d.is_dir()]
    if sources:
        source_dirs = [d for d in source_dirs if d.name in sources]

    print(f"📂 从 JSON 加载 {len(source_dirs)} 个来源...")

    chunker = NarrativeChunker()

    for src_dir in sorted(source_dirs):
        source_name = src_dir.name
        json_files = sorted(src_dir.glob("*.json"))
        if not json_files:
            continue

        source_chunks = []
        for file_idx, jf in enumerate(json_files):
            try:
                data_obj = json.loads(jf.read_text(encoding="utf-8"))
                chunks = chunker.chunk_elements(
                    data_obj.get("elements", []),
                    data_obj.get("source_url", ""),
                    data_obj.get("source_name", source_name),
                    data_obj.get("title", ""),
                    file_index=file_idx,
                )
                source_chunks.extend(chunks)
            except Exception as e:
                print(f"  ⚠ {jf.name}: {e}")

        all_chunks.extend(source_chunks)
        parents = chunker.get_parents(source_chunks)
        children = chunker.get_children(source_chunks)
        parent_count = len(parents)
        # 更新父块的 child_count
        for p in parents:
            p.metadata["child_count"] = len([c for c in children if c.parent_chunk_id == p.chunk_id])
        print(f"  ✅ {source_name}: {len(json_files)} JSON → {parent_count} parents + {len(children)} children")

    print(f"\n📊 总计: {len(all_chunks)} chunks")

    if not all_chunks:
        print("❌ 无 chunk，请检查 parsed JSON 是否有效")
        return

    # 拆分父子：索引只包含子块（检索轻量）
    children = chunker.get_children(all_chunks)
    parents = chunker.get_parents(all_chunks)
    print(f"  子块 (索引用): {len(children)}  父块 (LLM用): {len(parents)}")

    if not children:
        print("❌ 无子块，请检查 parsed JSON 是否有效")
        return

    # 构建向量索引（只索引子块）
    print("\n🔧 构建向量索引 (bge-base-zh-v1.5)...")
    vs = VectorStore()
    vs.build_index(children)
    vs.save("vectorstore/faiss.index", "vectorstore/children.json")
    print(f"  ✅ 向量索引: {len(children)} children × {vs.dim}d")

    # 构建 BM25 索引（只索引子块）
    print("\n🔧 构建 BM25 索引...")
    bm25 = BM25Store()
    bm25.build_index(children)
    bm25.save("vectorstore/bm25.pkl")
    print(f"  ✅ BM25 索引: {len(children)} docs")

    # 保存完整 chunk 列表（含父块）供 expand_to_parents 使用
    all_data = []
    for c in all_chunks:
        all_data.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata,
            "parent_chunk_id": c.parent_chunk_id,
            "is_parent": c.is_parent,
        })
    import json as _json
    Path("vectorstore/all_chunks.json").write_text(
        _json.dumps(all_data, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✅ 全量 chunk: {len(all_chunks)} (parents + children)")

    # 构建 Glossary 术语索引
    print("\n🔧 构建术语表索引...")
    from src.glossary import build_glossary_index
    build_glossary_index("vectorstore/glossary", model=vs.model)

    print("\n🎉 索引构建完成！现在可以运行: python scripts/query.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON → Chunk → 索引")
    parser.add_argument("--sources", type=str, help="逗号分隔的来源，默认全部")
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None
    build_all(sources=sources)
