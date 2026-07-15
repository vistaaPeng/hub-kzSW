"""
测试 BM25 检索器 —— 分词 + BM25 索引 + 检索。
"""
import pytest
from src.chunkers.narrative import Chunk
from src.retrievers.bm25_store import BM25Store, load_jieba_terms


@pytest.fixture
def sample_chunks():
    """创建测试用的 chunk 列表。
    
    注意：关键词"所有权"仅在 test_0000 和 test_0004 中出现（2/7），
    确保 BM25 IDF > 0，避免因过小语料库导致分数全部为 0。
    """
    return [
        Chunk(
            chunk_id="test_0000",
            text="Rust 的所有权系统确保内存安全，无需垃圾回收器。",
            metadata={"source_url": "url1", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0001",
            text="借用（Borrowing）允许在不转移值的情况下使用引用。",
            metadata={"source_url": "url1", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0002",
            text="生命周期（Lifetime）确保引用始终有效。",
            metadata={"source_url": "url2", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0003",
            text="Python 的变量不需要声明类型，是动态类型语言。",
            metadata={"source_url": "url3", "source_name": "reference"},
        ),
        Chunk(
            chunk_id="test_0004",
            text="Rust 的所有权模型是语言的核心特性之一，所有权规则在编译期检查。",
            metadata={"source_url": "url1", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0005",
            text="结构体（Struct）可以包含多个字段，并通过 impl 块定义方法。",
            metadata={"source_url": "url2", "source_name": "book"},
        ),
        Chunk(
            chunk_id="test_0006",
            text="模式匹配（Pattern Matching）是 Rust 中强大的控制流工具。",
            metadata={"source_url": "url2", "source_name": "book"},
        ),
    ]


class TestBM25Store:
    """BM25 检索器测试"""

    def test_build_and_search(self, sample_chunks):
        """索引构建后能检索到结果。"""
        store = BM25Store()
        store.build_index(sample_chunks)
        results = store.search("所有权", top_k=3)
        assert len(results) > 0, "应返回至少一个结果"
        assert len(results) <= 3, "不应超过 top_k"

        # 验证返回格式
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    def test_keyword_match_ownership(self, sample_chunks):
        """查询"所有权"返回包含"所有权"的 chunk。"""
        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.search("所有权", top_k=5)
        assert len(results) > 0

        # 第一个结果应包含"所有权"
        top_chunk, top_score = results[0]
        assert "所有权" in top_chunk.text, (
            f"第一个结果应包含'所有权'，实际文本: {top_chunk.text[:80]}"
        )
        # 分数应为正数
        assert top_score > 0

    def test_top_k_truncation(self, sample_chunks):
        """top_k 正确截断。"""
        store = BM25Store()
        store.build_index(sample_chunks)

        # 请求 2 个结果
        results = store.search("Rust", top_k=2)
        assert len(results) == 2

        # 请求超过总数
        results = store.search("Rust", top_k=100)
        assert len(results) == len(sample_chunks)

    def test_empty_query(self, sample_chunks):
        """空查询返回空列表。"""
        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.search("")
        assert results == []

        results = store.search("   ")
        assert results == []

    def test_empty_index(self):
        """空索引时 search 返回空列表。"""
        store = BM25Store()
        # 未调用 build_index
        results = store.search("所有权")
        assert results == []

        # build_index 传入空列表
        store.build_index([])
        results = store.search("所有权")
        assert results == []

    def test_scores_are_positive_for_matches(self, sample_chunks):
        """匹配的文档分数为正数。"""
        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.search("所有权", top_k=6)
        for chunk, score in results:
            if "所有权" in chunk.text:
                assert score > 0, (
                    f"包含'所有权'的 chunk 应有正分数，实际: {score}"
                )

    def test_irrelevant_query_returns_results(self, sample_chunks):
        """不相关的查询仍返回结果（分数可能为 0 或很低）。"""
        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.search("量子计算机", top_k=3)
        # 即使不相关，也应返回结果（与 VectorStore 行为一致）
        assert len(results) > 0
        # 分数应该很低（可能为 0）
        for _, score in results:
            assert score >= 0

    def test_rust_compound_terms_are_kept_together(self):
        """Rust 复合术语应作为整体 token，减少 BM25 查询漂移。"""
        load_jieba_terms()

        tokens = BM25Store._tokenize("associated type 和 父 trait 都很重要，函数指针也是。")

        assert "associated type" in tokens
        assert "父 trait" in tokens
        assert "函数指针" in tokens

    def test_glossary_terms_can_be_injected_into_jieba(self, monkeypatch):
        """Glossary 中的新增术语会注入 jieba 词典。"""
        import src.retrievers.bm25_store as bm25_module

        monkeypatch.setattr(bm25_module, "_JIEBA_TERMS_LOADED", False)
        monkeypatch.setattr(
            "src.glossary.get_glossary",
            lambda: {"custom rust term": "自定义术语"},
        )

        tokens = BM25Store._tokenize("custom rust term 对应 自定义术语")

        assert "custom rust term" in tokens
        assert "自定义术语" in tokens
