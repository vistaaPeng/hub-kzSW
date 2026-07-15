"""
测试评估器 —— 使用 mock retriever，不依赖模型。
"""
import pytest
from dataclasses import dataclass

from src.chunkers.narrative import Chunk
from evaluation.evaluator import RAGEvaluator, EvalQuestion, EvalResult


# ---------------------------------------------------------------------------
# Mock Retriever
# ---------------------------------------------------------------------------

@dataclass
class MockChunk:
    """轻量 mock，不需要真实的 Chunk dataclass 也能测试。"""
    text: str


class MockRetriever:
    """可控的 mock 检索器，返回预设的 chunk 列表。"""

    def __init__(self, chunks: list[MockChunk] | None = None):
        self.chunks: list[MockChunk] = chunks or []

    def search(self, query: str, top_k: int = 5):
        return [(c, 0.9) for c in self.chunks[:top_k]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rust_keywords() -> list[str]:
    return ["所有权", "借用", "生命周期"]


@pytest.fixture
def rust_question() -> EvalQuestion:
    return EvalQuestion(
        question_id="q1",
        question="Rust 的所有权和借用规则是什么？",
        keywords=["所有权", "借用", "生命周期"],
    )


@pytest.fixture
def all_relevant_chunks(rust_keywords) -> list[MockChunk]:
    return [
        MockChunk(text=f"这是关于{k}的说明文档，包含详细解释。")
        for k in rust_keywords
    ]


@pytest.fixture
def mixed_chunks() -> list[MockChunk]:
    return [
        MockChunk(text="这是关于所有权的说明文档。"),          # relevant
        MockChunk(text="Python 的变量不需要声明类型。"),        # NOT relevant
        MockChunk(text="借用（Borrowing）允许在不转移所有权的情况下使用值。"),  # relevant
        MockChunk(text="JavaScript 是动态类型语言。"),          # NOT relevant
        MockChunk(text="生命周期确保引用始终有效。"),           # relevant
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrecision:
    """Context Precision 测试。"""

    def test_exact_match_all_relevant(self, rust_question, all_relevant_chunks):
        """检索结果覆盖所有关键词 → precision=1.0"""
        retriever = MockRetriever(all_relevant_chunks)
        evaluator = RAGEvaluator(retriever, top_k=3)

        result = evaluator.evaluate_one(rust_question)

        assert result.precision == 1.0
        assert result.num_retrieved == 3
        assert result.num_relevant == 3

    def test_partial_match(self, rust_question, mixed_chunks):
        """只覆盖部分关键词 → precision=0.x"""
        retriever = MockRetriever(mixed_chunks)
        evaluator = RAGEvaluator(retriever, top_k=5)

        result = evaluator.evaluate_one(rust_question)

        # 3 relevant out of 5 → precision = 0.6
        assert result.precision == 0.6
        assert result.num_retrieved == 5
        assert result.num_relevant == 3

    def test_empty_results(self, rust_question):
        """空结果 → precision=0.0"""
        retriever = MockRetriever([])
        evaluator = RAGEvaluator(retriever, top_k=5)

        result = evaluator.evaluate_one(rust_question)

        assert result.precision == 0.0
        assert result.num_retrieved == 0
        assert result.num_relevant == 0

    def test_case_insensitive_keyword_match(self):
        """关键词大小写不敏感。"""
        question = EvalQuestion(
            question_id="q2",
            question="What is Rust?",
            keywords=["RUST", "Ownership"],
        )
        chunks = [
            MockChunk(text="rust is a systems programming language."),
            MockChunk(text="Ownership is a key concept in rust."),
        ]
        retriever = MockRetriever(chunks)
        evaluator = RAGEvaluator(retriever, top_k=2)

        result = evaluator.evaluate_one(question)

        assert result.precision == 1.0
        assert result.num_relevant == 2

    def test_no_matching_keywords(self, rust_question):
        """所有 chunk 都不含关键词 → precision=0.0"""
        chunks = [
            MockChunk(text="Python 是一门动态语言。"),
            MockChunk(text="JavaScript 运行在浏览器中。"),
        ]
        retriever = MockRetriever(chunks)
        evaluator = RAGEvaluator(retriever, top_k=2)

        result = evaluator.evaluate_one(rust_question)

        assert result.precision == 0.0
        assert result.num_retrieved == 2
        assert result.num_relevant == 0


class TestEvaluateAll:
    """批量评估测试。"""

    def test_evaluate_all_returns_correct_count(self, rust_question, all_relevant_chunks):
        """evaluate_all 返回正确数量的结果。"""
        questions = [
            rust_question,
            EvalQuestion(
                question_id="q2",
                question="什么是 Python？",
                keywords=["动态类型", "解释型"],
            ),
            EvalQuestion(
                question_id="q3",
                question="什么是 JavaScript？",
                keywords=["浏览器", "事件循环"],
            ),
        ]
        retriever = MockRetriever(all_relevant_chunks)  # 不含 Python/JS 关键词
        evaluator = RAGEvaluator(retriever, top_k=3)

        results = evaluator.evaluate_all(questions)

        assert len(results) == 3


class TestSummary:
    """汇总统计测试。"""

    def test_summary_stats_correct(self):
        """汇总统计平均值正确。"""
        results = [
            EvalResult(question_id="q1", precision=1.0, num_retrieved=5, num_relevant=5),
            EvalResult(question_id="q2", precision=0.5, num_retrieved=4, num_relevant=2),
            EvalResult(question_id="q3", precision=0.0, num_retrieved=3, num_relevant=0),
        ]
        evaluator = RAGEvaluator(MockRetriever())

        summary = evaluator.summary(results)

        assert summary["total_questions"] == 3
        assert summary["avg_precision"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert summary["avg_num_retrieved"] == pytest.approx((5 + 4 + 3) / 3)
        assert summary["avg_num_relevant"] == pytest.approx((5 + 2 + 0) / 3)

    def test_summary_empty(self):
        """空结果列表 → 默认值。"""
        evaluator = RAGEvaluator(MockRetriever())

        summary = evaluator.summary([])

        assert summary["avg_precision"] == 0.0
        assert summary["total_questions"] == 0


class TestRecallInterface:
    """Recall 预留接口测试。"""

    def test_compute_recall_exists(self):
        """compute_recall 方法存在。"""
        evaluator = RAGEvaluator(MockRetriever())
        assert hasattr(evaluator, "compute_recall")
        assert callable(evaluator.compute_recall)

    def test_compute_recall_returns_default(self, rust_question):
        """compute_recall 返回默认值 -1.0。"""
        evaluator = RAGEvaluator(MockRetriever())

        result = evaluator.compute_recall(rust_question, [])

        assert result == -1.0

    def test_compute_recall_with_ground_truth(self, rust_question):
        """compute_recall 即使传入 ground_truth 也返回 -1.0。"""
        evaluator = RAGEvaluator(MockRetriever())
        ground_truth = [MockChunk(text="关于所有权的文档")]

        result = evaluator.compute_recall(rust_question, [], ground_truth)

        assert result == -1.0


class TestDataclasses:
    """EvalQuestion / EvalResult dataclass 测试。"""

    def test_eval_question_defaults(self):
        """expected_source 默认为 None。"""
        q = EvalQuestion(question_id="q1", question="test?", keywords=["a"])
        assert q.expected_source is None

    def test_eval_result_defaults(self):
        """recall 默认为 -1.0。"""
        r = EvalResult(question_id="q1", precision=0.8)
        assert r.recall == -1.0
        assert r.num_retrieved == 0
        assert r.num_relevant == 0

    def test_eval_result_with_real_chunk(self):
        """确保 MockRetriever 也兼容真实的 Chunk dataclass。"""
        evaluator = RAGEvaluator(MockRetriever())
        q = EvalQuestion(question_id="q1", question="Rust", keywords=["Rust"])
        retriever = MockRetriever([Chunk(chunk_id="c1", text="Rust is great.")])
        evaluator.retriever = retriever

        result = evaluator.evaluate_one(q)

        assert result.precision == 1.0
