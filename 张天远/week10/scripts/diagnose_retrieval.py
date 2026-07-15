#!/usr/bin/env python
"""
Diagnose retrieval quality for one query.

The report shows BM25, vector, and RRF rankings side by side so recall misses
can be traced to a specific retrieval stage.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.narrative import Chunk
from src.pipeline import load_retriever
from src.retrievers.hybrid_retriever import DEFAULT_RRF_K


def _doc_key(chunk: Chunk) -> str:
    return chunk.metadata.get("source_url", chunk.chunk_id)


def _snippet(text: str, max_chars: int = 160) -> str:
    text = " ".join(text.split())
    return text[:max_chars]


def serialize_ranked_results(
    results: list[tuple[Chunk, float]],
    expected_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    expected_terms = [t.lower() for t in (expected_terms or []) if t]
    rows = []
    for rank, (chunk, score) in enumerate(results, 1):
        searchable = f"{chunk.text} {chunk.metadata.get('headings', '')}".lower()
        rows.append({
            "rank": rank,
            "score": round(score, 6),
            "chunk_id": chunk.chunk_id,
            "doc_key": _doc_key(chunk),
            "source_name": chunk.metadata.get("source_name", ""),
            "source_url": chunk.metadata.get("source_url", ""),
            "headings": chunk.metadata.get("headings", ""),
            "morphology": chunk.metadata.get("morphology", ""),
            "matched_expected_terms": [
                term for term in expected_terms if term in searchable
            ],
            "preview": _snippet(chunk.text),
        })
    return rows


def rrf_breakdown(
    vector_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    *,
    top_k: int = 10,
    k: int = DEFAULT_RRF_K,
    candidate_k: int | None = None,
    expected_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return document-level RRF score components."""
    candidate_k = candidate_k or max(top_k * 10, 20)
    default_rank = candidate_k + 1
    expected_terms = [t.lower() for t in (expected_terms or []) if t]

    doc_best_bm25: dict[str, int] = {}
    doc_best_vector: dict[str, int] = {}
    doc_chunks: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(bm25_results, 1):
        key = _doc_key(chunk)
        if key not in doc_best_bm25 or rank < doc_best_bm25[key]:
            doc_best_bm25[key] = rank
            doc_chunks[key] = chunk

    for rank, (chunk, _) in enumerate(vector_results, 1):
        key = _doc_key(chunk)
        if key not in doc_best_vector or rank < doc_best_vector[key]:
            doc_best_vector[key] = rank
            doc_chunks.setdefault(key, chunk)

    rows = []
    for key in set(doc_best_bm25) | set(doc_best_vector):
        bm25_rank = doc_best_bm25.get(key)
        vector_rank = doc_best_vector.get(key)
        bm25_component = 1.0 / (k + (bm25_rank or default_rank))
        vector_component = 1.0 / (k + (vector_rank or default_rank))
        chunk = doc_chunks[key]
        searchable = f"{chunk.text} {chunk.metadata.get('headings', '')}".lower()
        rows.append({
            "doc_key": key,
            "score": round(bm25_component + vector_component, 6),
            "bm25_rank": bm25_rank,
            "vector_rank": vector_rank,
            "bm25_component": round(bm25_component, 6),
            "vector_component": round(vector_component, 6),
            "representative_chunk_id": chunk.chunk_id,
            "source_name": chunk.metadata.get("source_name", ""),
            "source_url": chunk.metadata.get("source_url", ""),
            "headings": chunk.metadata.get("headings", ""),
            "morphology": chunk.metadata.get("morphology", ""),
            "matched_expected_terms": [
                term for term in expected_terms if term in searchable
            ],
            "preview": _snippet(chunk.text),
        })

    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]


def diagnose_query(
    retriever,
    query: str,
    *,
    top_k: int = 10,
    candidate_k: int | None = None,
    expected_terms: list[str] | None = None,
) -> dict[str, Any]:
    """Build a retrieval diagnostic report for a query."""
    candidate_k = candidate_k or max(top_k * 10, 20)
    vector_results = retriever.vector_store.search(query, candidate_k) or []
    bm25_results = retriever.bm25_store.search(query, candidate_k) or []
    rrf_rows = rrf_breakdown(
        vector_results,
        bm25_results,
        top_k=top_k,
        k=retriever.k,
        candidate_k=candidate_k,
        expected_terms=expected_terms,
    )

    return {
        "query": query,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "expected_terms": expected_terms or [],
        "bm25": serialize_ranked_results(bm25_results[:top_k], expected_terms),
        "vector": serialize_ranked_results(vector_results[:top_k], expected_terms),
        "rrf": rrf_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose BM25/vector/RRF retrieval")
    parser.add_argument("query")
    parser.add_argument("--index-dir", default="vectorstore")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        help="expected term to highlight; can be passed multiple times",
    )
    parser.add_argument("--output", help="write JSON report to this file")
    args = parser.parse_args()

    retriever = load_retriever(args.index_dir)
    report = diagnose_query(
        retriever,
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        expected_terms=args.expect,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
