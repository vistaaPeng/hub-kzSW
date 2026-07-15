"""
Tests for retrieval diagnostics.
"""

from src.chunkers.narrative import Chunk
from scripts.diagnose_retrieval import diagnose_query, rrf_breakdown


class MockStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class MockRetriever:
    def __init__(self, vector_results, bm25_results, k=60):
        self.vector_store = MockStore(vector_results)
        self.bm25_store = MockStore(bm25_results)
        self.k = k


def chunk(chunk_id, text, source_url, headings="", morphology="narrative"):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source_url": source_url,
            "source_name": "book",
            "headings": headings,
            "morphology": morphology,
        },
    )


def test_rrf_breakdown_reports_rank_components_and_expected_terms():
    a = chunk("a1", "所有权规则", "url-a", headings="所有权")
    b = chunk("b1", "生命周期", "url-b")

    rows = rrf_breakdown(
        vector_results=[(a, 0.9), (b, 0.8)],
        bm25_results=[(b, 2.0), (a, 1.0)],
        top_k=2,
        candidate_k=20,
        expected_terms=["所有权"],
    )

    assert len(rows) == 2
    assert rows[0]["score"] == rows[1]["score"]
    by_doc = {row["doc_key"]: row for row in rows}
    assert by_doc["url-a"]["vector_rank"] == 1
    assert by_doc["url-a"]["bm25_rank"] == 2
    assert by_doc["url-a"]["matched_expected_terms"] == ["所有权"]


def test_diagnose_query_calls_both_stores_and_returns_sections():
    a = chunk("a1", "Rust 关键字列表", "url-a", morphology="structured")
    b = chunk("b1", "Python text", "url-b")
    retriever = MockRetriever(
        vector_results=[(a, 0.9)],
        bm25_results=[(b, 2.0), (a, 1.0)],
    )

    report = diagnose_query(
        retriever,
        "rust有哪些关键字",
        top_k=2,
        candidate_k=5,
        expected_terms=["关键字"],
    )

    assert report["query"] == "rust有哪些关键字"
    assert set(report) >= {"bm25", "vector", "rrf"}
    assert retriever.vector_store.calls == [("rust有哪些关键字", 5)]
    assert retriever.bm25_store.calls == [("rust有哪些关键字", 5)]
    assert report["vector"][0]["matched_expected_terms"] == ["关键字"]
    assert report["rrf"][0]["doc_key"] in {"url-a", "url-b"}
