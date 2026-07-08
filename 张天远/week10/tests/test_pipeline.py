"""
Tests for the shared RAG pipeline orchestration.
"""

from src.chunkers.narrative import Chunk
from src.pipeline import RAGPipeline, source_dicts


class MockRetriever:
    def __init__(self, results_by_query, expanded=None):
        self.results_by_query = results_by_query
        self.expanded = expanded
        self.expand_calls = []

    def search(self, query: str, top_k: int = 5):
        return self.results_by_query.get(query, [])[:top_k]

    def expand_to_parents(self, child_results, all_chunks, expand_siblings=True):
        self.expand_calls.append((child_results, all_chunks, expand_siblings))
        if self.expanded is not None:
            return self.expanded
        return child_results


class MockRewriter:
    def rewrite(self, query: str, n_variants: int = 3):
        return [query, "Rust 所有权", "ownership rules"][: n_variants + 1]


class MockGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, query, chunks, max_chunks=5, temperature=0.1, glossary_terms=None):
        self.calls.append((query, chunks, max_chunks, glossary_terms))
        return "generated answer"


def chunk(chunk_id, text, source_url):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_url": source_url, "source_name": "book", "headings": "H"},
    )


def test_multi_search_merges_by_source_url_with_best_score():
    a_low = chunk("a1", "low", "url-a")
    a_high = chunk("a2", "high", "url-a")
    b = chunk("b", "other", "url-b")
    retriever = MockRetriever({
        "q": [(a_low, 0.1), (b, 0.5)],
        "Rust 所有权": [(a_high, 0.9)],
    })
    pipeline = RAGPipeline(retriever, all_chunks=[], rewriter=MockRewriter())

    result = pipeline.retrieve("q", top_k=5)

    assert result.queries == ["q", "Rust 所有权", "ownership rules"]
    assert [c.chunk_id for c, _ in result.child_results] == ["a2", "b"]
    assert result.child_results[0][1] == 0.9


def test_retrieve_expands_to_parent_results():
    child = chunk("child", "child text", "url")
    parent = Chunk("parent", "parent text", {"source_url": "url"}, is_parent=True)
    retriever = MockRetriever({"q": [(child, 0.7)]}, expanded=[(parent, 0.7)])
    pipeline = RAGPipeline(retriever, all_chunks=[parent])

    result = pipeline.retrieve("q", rewrite=False)

    assert result.parent_chunks == [parent]
    assert retriever.expand_calls[0][1] == [parent]


def test_generate_answer_uses_existing_parent_context():
    parent = Chunk("parent", "parent text", {"source_url": "url"}, is_parent=True)
    retriever = MockRetriever({"q": []}, expanded=[(parent, 0.8)])
    generator = MockGenerator()
    pipeline = RAGPipeline(retriever, all_chunks=[parent], generator=generator)
    result = pipeline.retrieve("q", rewrite=False)

    pipeline.generate_answer(result, include_glossary=False)

    assert result.answer == "generated answer"
    assert generator.calls[0][1] == [parent]


def test_source_dicts_include_full_text_and_preview():
    parent = Chunk(
        "parent",
        "x" * 250,
        {"source_url": "url", "source_name": "book", "headings": "Heading"},
        is_parent=True,
    )

    data = source_dicts([(parent, 0.123456)], limit=1)

    assert data[0]["score"] == 0.1235
    assert data[0]["preview"] == "x" * 200
    assert data[0]["text"] == "x" * 250
