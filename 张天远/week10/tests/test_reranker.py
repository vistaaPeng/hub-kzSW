"""
测试 Cross-Encoder 重排器。
"""
import pytest
from src.chunkers.narrative import Chunk
from src.reranker.reranker import ReRanker


@pytest.fixture
def rust_chunks():
    """Rust 相关 chunk 用于测试重排"""
    return [
        Chunk("c0", "Rust 的所有权系统确保内存安全，每个值都有一个所有者。",
              {"source_name": "book"}),
        Chunk("c1", "Python 使用垃圾回收器管理内存，开发者不需要手动释放。",
              {"source_name": "reference"}),
        Chunk("c2", "借用允许在 Rust 中不转移所有权的情况下使用值的引用。",
              {"source_name": "book"}),
        Chunk("c3", "JavaScript 的 var 关键字有函数作用域，let 和 const 有块作用域。",
              {"source_name": "reference"}),
    ]


class TestReRanker:
    """重排器测试"""

    def test_rerank_top1_relevant(self, rust_chunks):
        """重排后 top-1 应该是最相关的 Rust chunk"""
        reranker = ReRanker()
        results = reranker.rerank("Rust 所有权", rust_chunks, top_k=3)
        assert len(results) <= 3
        # 最相关的不应该是 Python/JS chunk
        top_chunk, top_score = results[0]
        assert "Python" not in top_chunk.text
        assert "JavaScript" not in top_chunk.text

    def test_empty_chunks(self):
        """空 chunk 列表返回空"""
        reranker = ReRanker()
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_disabled_mode(self, rust_chunks):
        """enabled=False 时跳过重排"""
        reranker = ReRanker(enabled=False)
        results = reranker.rerank("Rust", rust_chunks, top_k=2)
        assert len(results) == 2
        # disabled 模式下分数为 0
        assert results[0][1] == 0.0

    def test_top_k_truncation(self, rust_chunks):
        """top_k 正确截断"""
        reranker = ReRanker()
        results = reranker.rerank("内存管理", rust_chunks, top_k=2)
        assert len(results) == 2

    def test_scores_descending(self, rust_chunks):
        """分数严格降序"""
        reranker = ReRanker()
        results = reranker.rerank("Rust 所有权", rust_chunks, top_k=4)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
