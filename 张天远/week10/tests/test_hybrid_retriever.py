"""
测试混合检索器 —— RRF 融合算法。
"""

import pytest
from unittest.mock import MagicMock
from src.chunkers.narrative import Chunk
from src.retrievers.hybrid_retriever import HybridRetriever, DEFAULT_RRF_K


# ── 测试 fixtures ──────────────────────────────────────────────

@pytest.fixture
def chunk_a():
    """Chunk A: 关于 Rust 所有权"""
    return Chunk(
        chunk_id="doc_0000",
        text="Rust 的所有权系统确保内存安全，无需垃圾回收器。",
        metadata={"source_url": "url1", "source_name": "rust_book"},
    )


@pytest.fixture
def chunk_b():
    """Chunk B: 关于 Rust 生命周期"""
    return Chunk(
        chunk_id="doc_0001",
        text="生命周期（Lifetime）确保引用始终有效。",
        metadata={"source_url": "url1", "source_name": "rust_book"},
    )


@pytest.fixture
def chunk_c():
    """Chunk C: 关于 Python"""
    return Chunk(
        chunk_id="doc_0002",
        text="Python 的变量不需要声明类型，是动态类型语言。",
        metadata={"source_url": "url2", "source_name": "python_book"},
    )


@pytest.fixture
def chunk_d():
    """Chunk D: 关于 Rust trait"""
    return Chunk(
        chunk_id="doc_0003",
        text="Trait 定义共享行为，类似于其他语言的接口。",
        metadata={"source_url": "url1", "source_name": "rust_book"},
    )


@pytest.fixture
def chunk_e():
    """Chunk E: 关于 Rust 宏"""
    return Chunk(
        chunk_id="doc_0004",
        text="Rust 的宏系统允许元编程，在编译期生成代码。",
        metadata={"source_url": "url1", "source_name": "rust_book"},
    )


# ── Mock 检索器工厂 ───────────────────────────────────────────

def make_mock_store(results_map):
    """
    创建 mock 检索器。

    Args:
        results_map: dict[query -> list[tuple[Chunk, float]]]
                     或用 callable 代替 dict 支持更灵活的匹配

    Returns:
        MagicMock 对象，search(query, top_k) 返回对应结果
    """
    mock = MagicMock()

    def search_side_effect(query, top_k=5):
        if callable(results_map):
            return results_map(query, top_k)
        return results_map.get(query, [])

    mock.search.side_effect = search_side_effect
    return mock


def with_source(chunk: Chunk, source_url: str) -> Chunk:
    """Copy a chunk with a specific source_url for document-level RRF tests."""
    return Chunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        metadata={**chunk.metadata, "source_url": source_url},
        parent_chunk_id=chunk.parent_chunk_id,
        is_parent=chunk.is_parent,
    )


# ── 测试类 ─────────────────────────────────────────────────────

class TestHybridRetriever:
    """混合检索器 RRF 融合测试"""

    def test_rrf_merges_both_sides(self, chunk_a, chunk_b, chunk_c, chunk_d):
        """RRF 融合：两边命中的 source_url 都进入最终结果"""
        query = "Rust 内存管理"

        # Vector: [A, B]，BM25: [C, D]
        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_b, 0.80)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_c, 2.5), (chunk_d, 1.8)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=4)

        urls = {chunk.metadata["source_url"] for chunk, _ in results}
        assert len(results) == 2
        # url1 来自 vector 和 BM25；url2 来自 BM25
        assert urls == {"url1", "url2"}
        assert results[0][0].metadata["source_url"] == "url1"

    def test_same_chunk_rank_boost(self, chunk_a, chunk_b, chunk_c):
        """同一 chunk 出现在两边时，RRF 分数更高，排名更靠前"""
        query = "Rust 所有权系统"

        # Vector 把 A 排第 1，BM25 也把 A 排第 1 → A 应获得最高 RRF 分数
        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_b, 0.70), (chunk_c, 0.30)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_a, 3.0), (chunk_b, 1.5), (chunk_c, 0.5)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=3)

        # chunk_a 应该是第一
        assert results[0][0].chunk_id == "doc_0000"
        # chunk_a 的 RRF 分数应该是 1/(k+1) + 1/(k+1) = 2/(k+1)
        expected_score = 2.0 / (DEFAULT_RRF_K + 1)
        assert results[0][1] == pytest.approx(expected_score)

    def test_k_parameter_affects_ordering(self, chunk_a, chunk_b, chunk_c):
        """k 参数影响排序：k 越小，排名差异影响越大"""
        query = "Rust"
        chunk_a = with_source(chunk_a, "url-a")
        chunk_b = with_source(chunk_b, "url-b")
        chunk_c = with_source(chunk_c, "url-c")

        # Vector: [A(rank1), B(rank2)]；BM25: [B(rank1), A(rank2)]
        # 即 A 和 B 在两个检索器中排名互换
        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_b, 0.70), (chunk_c, 0.30)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_b, 3.0), (chunk_a, 1.5), (chunk_c, 0.5)],
        })

        # k=60（默认）: A 和 B 的 RRF 分数相同（对称），因为 rank 互换
        # score(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639 + 0.01613 ≈ 0.03252
        # score(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61，相同的分数
        retriever_default = HybridRetriever(vector_store, bm25_store, k=60)
        results_default = retriever_default.search(query, top_k=2)

        # A 和 B 分数应该非常接近（对称），但由于 B 在两边名次交换后总分相同
        # 验证分数确实相等（允许浮点误差）
        assert results_default[0][1] == pytest.approx(results_default[1][1], rel=1e-9)

        # k=0: rank 差异被放大
        # score(A) = 1/(0+1) + 1/(0+2) = 1 + 0.5 = 1.5
        # score(B) = 1/(0+2) + 1/(0+1) = 0.5 + 1 = 1.5，依然相同
        # 但 A 和 C 的差距会变大
        retriever_k0 = HybridRetriever(vector_store, bm25_store, k=0)
        results_k0 = retriever_k0.search(query, top_k=3)

        # 第三个结果 C: score(C) = 1/(0+3) + 1/(0+3) = 1/3 + 1/3 ≈ 0.6667
        # 验证 C 在第三位
        assert len(results_k0) == 3
        assert results_k0[2][0].chunk_id == "doc_0002"

    def test_k_large_flattens_ranking(self, chunk_a, chunk_b, chunk_c):
        """k 很大时，排名差异几乎消失，所有分数趋近"""
        query = "Rust"
        chunk_a = with_source(chunk_a, "url-a")
        chunk_b = with_source(chunk_b, "url-b")
        chunk_c = with_source(chunk_c, "url-c")

        # A 在两个检索器中都排第一，C 排最后
        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_b, 0.70), (chunk_c, 0.30)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_a, 3.0), (chunk_b, 1.5), (chunk_c, 0.5)],
        })

        # k=10000: 所有分数都接近，A 和 C 的分数差距很小
        retriever = HybridRetriever(vector_store, bm25_store, k=10000)
        results = retriever.search(query, top_k=3)

        assert len(results) == 3
        # A 和 C 的分数差距应小于 0.0001
        score_diff = results[0][1] - results[2][1]
        assert score_diff < 0.0001

    def test_empty_results_one_store(self, chunk_a, chunk_b):
        """一个检索器返回空结果，另一个按 source_url 聚合后正常"""
        query = "Rust"

        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_b, 0.70)],
        })
        bm25_store = make_mock_store({})  # BM25 无结果

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=5)

        assert len(results) == 1
        # 两个 chunk 属于同一个 source_url，因此只返回一个文档结果
        chunk_ids = {chunk.chunk_id for chunk, _ in results}
        assert chunk_ids == {"doc_0000"}

    def test_empty_results_both_stores(self):
        """两个检索器都返回空结果"""
        query = "不存在的查询"
        vector_store = make_mock_store({})
        bm25_store = make_mock_store({})

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=5)

        assert results == []

    def test_top_k_smaller_than_available(self, chunk_a, chunk_b, chunk_c, chunk_d, chunk_e):
        """top_k 小于可用结果数时，只返回 top_k 个"""
        query = "Rust"
        chunk_a = with_source(chunk_a, "url-a")
        chunk_b = with_source(chunk_b, "url-b")
        chunk_c = with_source(chunk_c, "url-c")
        chunk_d = with_source(chunk_d, "url-d")
        chunk_e = with_source(chunk_e, "url-e")

        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_b, 0.70), (chunk_c, 0.30)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_d, 2.0), (chunk_e, 1.0)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=3)

        # 共 5 个不同的 chunk，但只返回 top_k=3
        assert len(results) == 3

    def test_duplicate_in_same_store_handled(self, chunk_a, chunk_b):
        """同一 source_url 在同一检索器中重复出现时只取最佳排名"""
        query = "Rust"
        chunk_a = with_source(chunk_a, "url-a")
        chunk_b = with_source(chunk_b, "url-b")

        # 同一 chunk 在同一个检索器中多次出现（边界情况）
        vector_store = make_mock_store({
            query: [(chunk_a, 0.95), (chunk_a, 0.80)],  # A 出现两次
        })
        bm25_store = make_mock_store({
            query: [(chunk_b, 2.0)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=5)

        # chunk_a 应该只出现一次，并只按最佳排名计分
        assert len(results) == 2  # A 和 B，各一个
        chunk_a_result = [s for c, s in results if c.chunk_id == "doc_0000"]
        assert len(chunk_a_result) == 1  # 只有一个 A

        # A 的分数 = vector 最佳 rank1 + BM25 缺失 default_rank51
        expected_a = 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 51)
        assert chunk_a_result[0] == pytest.approx(expected_a)

    def test_results_are_sorted_by_rrf_descending(self, chunk_a, chunk_b, chunk_c, chunk_d):
        """验证结果按 RRF 分数降序排列"""
        query = "Rust"
        chunk_a = with_source(chunk_a, "url-a")
        chunk_b = with_source(chunk_b, "url-b")
        chunk_c = with_source(chunk_c, "url-c")
        chunk_d = with_source(chunk_d, "url-d")

        # Vector: A(rank1) → 1/(k+1), B(rank2) → 1/(k+2)
        # BM25: D(rank1) → 1/(k+1), C(rank2) → 1/(k+2)
        # 预期顺序: A 和 D 并列最高，B 和 C 并列其次
        vector_store = make_mock_store({
            query: [(chunk_a, 0.9), (chunk_b, 0.7)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_d, 3.0), (chunk_c, 1.5)],
        })

        retriever = HybridRetriever(vector_store, bm25_store, k=60)
        results = retriever.search(query, top_k=4)

        assert len(results) == 4
        scores = [s for _, s in results]
        # 验证降序
        assert scores == sorted(scores, reverse=True)

        # A/D: rank1 in one retriever + default_rank41 in the other.
        # B/C: rank2 in one retriever + default_rank41 in the other.
        high = 1.0 / 61 + 1.0 / 101
        low = 1.0 / 62 + 1.0 / 101
        assert results[0][1] == pytest.approx(high)
        assert results[1][1] == pytest.approx(high)
        assert results[2][1] == pytest.approx(low)
        assert results[3][1] == pytest.approx(low)

    def test_list_intent_boosts_structured_chunks(self, chunk_c):
        """列表类查询优先 structured 形态的候选块。"""
        query = "rust有哪些关键字"
        structured = Chunk(
            chunk_id="doc_structured",
            text="fn let mut struct enum trait impl match async await",
            metadata={
                "source_url": "url-structured",
                "source_name": "rust_book",
                "morphology": "structured",
            },
        )
        narrative = Chunk(
            chunk_id=chunk_c.chunk_id,
            text=chunk_c.text,
            metadata={
                **chunk_c.metadata,
                "source_url": "url-narrative",
                "morphology": "narrative",
            },
        )

        vector_store = make_mock_store({
            query: [(narrative, 0.9), (structured, 0.8)],
        })
        bm25_store = make_mock_store({
            query: [(structured, 3.0), (narrative, 2.0)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=2)

        assert results[0][0].chunk_id == "doc_structured"
        assert results[0][1] == pytest.approx(results[1][1] + 0.002)

    def test_code_intent_boosts_code_unit_chunks(self, chunk_c):
        """代码类查询优先 code_unit 形态的候选块。"""
        query = "如何写Rust函数示例"
        code_chunk = Chunk(
            chunk_id="doc_code",
            text="fn add(a: i32, b: i32) -> i32 { a + b }",
            metadata={
                "source_url": "url-code",
                "source_name": "rust_book",
                "morphology": "code_unit",
            },
        )
        narrative = Chunk(
            chunk_id=chunk_c.chunk_id,
            text=chunk_c.text,
            metadata={
                **chunk_c.metadata,
                "source_url": "url-narrative",
                "morphology": "narrative",
            },
        )

        vector_store = make_mock_store({
            query: [(narrative, 0.9), (code_chunk, 0.8)],
        })
        bm25_store = make_mock_store({
            query: [(code_chunk, 3.0), (narrative, 2.0)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=2)

        assert results[0][0].chunk_id == "doc_code"
        assert results[0][1] == pytest.approx(results[1][1] + 0.002)

    def test_heading_match_boosts_close_candidates(self, chunk_c):
        """中文查询命中 headings 时，可以在基础 RRF 相同的候选中胜出。"""
        query = "rust有哪些关键字"
        heading_match = Chunk(
            chunk_id="doc_heading",
            text="Rust 关键字包括 fn、let、mut、struct。",
            metadata={
                "source_url": "url-heading",
                "source_name": "rust_book",
                "headings": "Rust 关键字",
                "morphology": "narrative",
            },
        )
        plain = Chunk(
            chunk_id=chunk_c.chunk_id,
            text=chunk_c.text,
            metadata={
                **chunk_c.metadata,
                "source_url": "url-plain",
                "headings": "Python 变量",
                "morphology": "narrative",
            },
        )

        vector_store = make_mock_store({
            query: [(plain, 0.9), (heading_match, 0.8)],
        })
        bm25_store = make_mock_store({
            query: [(heading_match, 3.0), (plain, 2.0)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=2)

        assert results[0][0].chunk_id == "doc_heading"
        assert results[0][1] == pytest.approx(results[1][1] + 0.003)

    def test_respects_top_k_parameter_in_search_calls(self, chunk_a):
        """验证传给子检索器的 top_k 参数合理（足够大以获取候选）"""
        query = "Rust"

        vector_store = make_mock_store({
            query: [(chunk_a, 0.95)],
        })
        bm25_store = make_mock_store({
            query: [(chunk_a, 2.0)],
        })

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search(query, top_k=1)

        assert len(results) == 1

        # 验证子检索器被调用了（top_k * 4 ≥ 20，所以至少是 20）
        vector_store.search.assert_called_once()
        bm25_store.search.assert_called_once()

        # 调用参数中的 top_k 至少为 20（作为第三个位置参数传入）
        call_args = vector_store.search.call_args[0]
        assert len(call_args) >= 2
        # call_args: (query, top_k) — top_k 是第二个位置参数
        assert call_args[1] >= 20

    def test_expand_to_parents_centers_on_hit_parent(self):
        """父块扩展以命中父块为中心，而不是从文档开头拼接。"""
        parents = [
            Chunk(
                chunk_id=f"book_f0000p000{i}",
                text=f"section-{i} " + ("x" * 40),
                metadata={"source_url": "url-book", "source_name": "book"},
                is_parent=True,
            )
            for i in range(4)
        ]
        child = Chunk(
            chunk_id="book_f0000c0002_00",
            text="hit child",
            metadata={"source_url": "url-book", "source_name": "book"},
            parent_chunk_id="book_f0000p0002",
        )

        results = HybridRetriever.expand_to_parents(
            [(child, 0.9)],
            parents + [child],
            max_chars=150,
        )

        assert len(results) == 1
        merged = results[0][0]
        assert "section-2" in merged.text
        assert "section-1" in merged.text
        assert "section-3" in merged.text
        assert "section-0" not in merged.text
        assert merged.metadata["sibling_chunk_ids"] == [
            "book_f0000p0001",
            "book_f0000p0002",
            "book_f0000p0003",
        ]
