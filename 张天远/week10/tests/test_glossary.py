"""
Tests for glossary search helpers.
"""

import sys
import types

from src.chunkers.narrative import Chunk
from src import glossary


def test_search_glossary_reuses_cached_retrievers(monkeypatch, tmp_path):
    load_counts = {"vector": 0, "bm25": 0}

    class FakeVectorStore:
        def __init__(self):
            load_counts["vector"] += 1

        def load(self, index_path, chunks_path):
            self.index_path = index_path
            self.chunks_path = chunks_path

        def search(self, query, top_k=5):
            chunk = Chunk(
                "g1",
                "supertrait 父 trait",
                {"english": "supertrait", "chinese": "父 trait"},
            )
            return [(chunk, 0.9)]

    class FakeBM25Store:
        def __init__(self):
            load_counts["bm25"] += 1

        def load(self, path):
            self.path = path

        def search(self, query, top_k=5):
            chunk = Chunk(
                "g2",
                "associated type 关联类型",
                {"english": "associated type", "chinese": "关联类型"},
            )
            return [(chunk, 1.0)]

    vector_module = types.ModuleType("src.retrievers.vector_store")
    vector_module.VectorStore = FakeVectorStore
    bm25_module = types.ModuleType("src.retrievers.bm25_store")
    bm25_module.BM25Store = FakeBM25Store

    monkeypatch.setitem(sys.modules, "src.retrievers.vector_store", vector_module)
    monkeypatch.setitem(sys.modules, "src.retrievers.bm25_store", bm25_module)
    monkeypatch.setattr(glossary, "get_glossary", lambda: {
        "supertrait": "父 trait",
        "associated type": "关联类型",
    })
    glossary.clear_glossary_retriever_cache()

    first = glossary.search_glossary("trait", index_dir=str(tmp_path))
    second = glossary.search_glossary("trait", index_dir=str(tmp_path))

    assert [t["english"] for t in first] == ["supertrait", "associated type"]
    assert second == first
    assert load_counts == {"vector": 1, "bm25": 1}

    glossary.clear_glossary_retriever_cache()
