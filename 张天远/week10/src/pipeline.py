"""
Shared RAG query pipeline.

This module keeps the end-to-end query behavior in one place so CLI, API, and
evaluation paths measure the same system.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

from src.chunkers.narrative import Chunk
from src.retrievers.vector_store import VectorStore
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid_retriever import HybridRetriever
from src.llm.generator import RAGGenerator
from src.llm.query_rewriter import QueryRewriter
from src.glossary import search_glossary
from src.reranker.reranker import ReRanker


INDEX_DIR = Path("vectorstore")


@dataclass
class PipelineResult:
    question: str
    queries: list[str]
    child_results: list[tuple[Chunk, float]] = field(default_factory=list)
    parent_results: list[tuple[Chunk, float]] = field(default_factory=list)
    glossary_terms: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""

    @property
    def rewrites(self) -> list[str]:
        return self.queries[1:] if len(self.queries) > 1 else []

    @property
    def parent_chunks(self) -> list[Chunk]:
        return [c for c, _ in self.parent_results]


def load_all_chunks(index_dir: Path | str = INDEX_DIR) -> list[Chunk]:
    """Load all chunks, preferring parent+child data when available."""
    index_dir = Path(index_dir)
    path = index_dir / "all_chunks.json"
    if not path.exists():
        path = index_dir / "children.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk(
            d["chunk_id"],
            d["text"],
            d["metadata"],
            d.get("parent_chunk_id"),
            d.get("is_parent", False),
        )
        for d in data
    ]


def load_retriever(index_dir: Path | str = INDEX_DIR) -> HybridRetriever:
    """Load the default hybrid retriever from disk."""
    index_dir = Path(index_dir)
    vs = VectorStore()
    vs.load(str(index_dir / "faiss.index"), str(index_dir / "children.json"))
    bm25 = BM25Store()
    bm25.load(str(index_dir / "bm25.pkl"))
    return HybridRetriever(vs, bm25)


class RAGPipeline:
    """End-to-end RAG orchestration shared by CLI, API, and evaluation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        all_chunks: list[Chunk],
        generator: RAGGenerator | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: ReRanker | None = None,
    ):
        self.retriever = retriever
        self.all_chunks = all_chunks
        self.generator = generator
        self.rewriter = rewriter
        self.reranker = reranker

    @classmethod
    def from_index(
        cls,
        index_dir: Path | str = INDEX_DIR,
        *,
        generator: RAGGenerator | None = None,
        rewrite: bool = True,
        reranker: ReRanker | None = None,
    ) -> "RAGPipeline":
        """Construct a pipeline using the persisted vectorstore files."""
        retriever = load_retriever(index_dir)
        all_chunks = load_all_chunks(index_dir)
        rewriter = QueryRewriter() if rewrite else None
        return cls(retriever, all_chunks, generator, rewriter, reranker)

    def rewrite_query(self, question: str, n_variants: int = 3) -> list[str]:
        """Rewrite the query, falling back to the original query on failure."""
        if self.rewriter is None:
            return [question]
        try:
            return self.rewriter.rewrite(question, n_variants=n_variants)
        except Exception:
            return [question]

    def multi_search(self, queries: list[str], top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Run multiple queries and merge by source_url, keeping the best score."""
        seen: dict[str, tuple[Chunk, float]] = {}
        for q in queries:
            try:
                results = self.retriever.search(q, top_k=top_k)
            except Exception:
                continue
            for chunk, score in results:
                url = chunk.metadata.get("source_url", chunk.chunk_id)
                if url not in seen or score > seen[url][1]:
                    seen[url] = (chunk, score)
        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 10,
        rewrite: bool = True,
        rewrite_variants: int = 3,
        expand_siblings: bool = True,
        rerank: bool = False,
        rerank_top_k: int = 5,
    ) -> PipelineResult:
        """Retrieve child chunks, expand them to parent context, and optionally rerank."""
        queries = self.rewrite_query(question, rewrite_variants) if rewrite else [question]
        child_results = self.multi_search(queries, top_k=top_k)
        parent_results = self.retriever.expand_to_parents(
            child_results,
            self.all_chunks,
            expand_siblings=expand_siblings,
        )

        if rerank and self.reranker is not None:
            self.reranker.enabled = True
            chunks = [c for c, _ in parent_results]
            parent_results = self.reranker.rerank(question, chunks, top_k=rerank_top_k)

        return PipelineResult(
            question=question,
            queries=queries,
            child_results=child_results,
            parent_results=parent_results,
        )

    def query(
        self,
        question: str,
        *,
        top_k: int = 10,
        max_chunks: int = 8,
        rewrite: bool = True,
        rewrite_variants: int = 3,
        rerank: bool = False,
        include_glossary: bool = True,
        generate: bool = True,
    ) -> PipelineResult:
        """Run the full RAG flow through answer generation."""
        result = self.retrieve(
            question,
            top_k=top_k,
            rewrite=rewrite,
            rewrite_variants=rewrite_variants,
            rerank=rerank,
        )

        if include_glossary:
            try:
                result.glossary_terms = search_glossary(question, top_k=10)
            except Exception:
                result.glossary_terms = []

        if generate:
            if self.generator is None:
                raise RuntimeError("RAGPipeline requires a generator when generate=True")
            chunks = result.parent_chunks[:max_chunks]
            result.answer = self.generator.generate(
                question,
                chunks,
                max_chunks=max_chunks,
                glossary_terms=result.glossary_terms,
            )

        return result

    def generate_answer(
        self,
        result: PipelineResult,
        *,
        max_chunks: int = 8,
        include_glossary: bool = True,
    ) -> PipelineResult:
        """Generate an answer from an existing retrieval result."""
        if self.generator is None:
            raise RuntimeError("RAGPipeline requires a generator to generate answers")

        if include_glossary and not result.glossary_terms:
            try:
                result.glossary_terms = search_glossary(result.question, top_k=10)
            except Exception:
                result.glossary_terms = []

        result.answer = self.generator.generate(
            result.question,
            result.parent_chunks[:max_chunks],
            max_chunks=max_chunks,
            glossary_terms=result.glossary_terms,
        )
        return result


def source_dicts(results: list[tuple[Chunk, float]], limit: int = 8) -> list[dict[str, Any]]:
    """Serialize retrieved parent chunks for API/UI consumers."""
    return [
        {
            "rank": i + 1,
            "score": round(score, 4),
            "chunk_id": chunk.chunk_id,
            "source_url": chunk.metadata.get("source_url", ""),
            "source_name": chunk.metadata.get("source_name", ""),
            "headings": chunk.metadata.get("headings", ""),
            "preview": chunk.text[:200],
            "text": chunk.text,
        }
        for i, (chunk, score) in enumerate(results[:limit])
    ]
