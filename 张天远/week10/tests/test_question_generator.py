"""
测试题生成器测试 —— 基础功能 + API 集成（可选）。
"""
import pytest
from src.chunkers.narrative import Chunk
from evaluation.question_generator import (
    generate_questions,
    _extract_keywords,
    _generate_fallback_question,
)

# 模拟 chunks
SAMPLE_CHUNKS = [
    Chunk(
        chunk_id="book_0000",
        text=(
            "Rust 的所有权系统确保内存安全。每个值在 Rust 中都有一个所有者，"
            "同一时间只能有一个可变引用或多个不可变引用。"
        ),
        metadata={
            "source_url": "https://doc.rust-lang.org/book/ch04-01.html",
            "source_name": "book",
            "headings": "所有权 > 所有权规则",
        },
    ),
    Chunk(
        chunk_id="book_0001",
        text=(
            "借用允许在不转移所有权的情况下使用值的引用。"
            "借用分为不可变借用（&T）和可变借用（&mut T）。"
        ),
        metadata={
            "source_url": "https://doc.rust-lang.org/book/ch04-02.html",
            "source_name": "book",
            "headings": "所有权 > 引用与借用",
        },
    ),
    Chunk(
        chunk_id="book_0002",
        text=(
            "生命周期（lifetime）是 Rust 中用于确保引用始终有效的机制。"
            "生命周期标注使用撇号语法，如 'a。"
            "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str { ... }"
        ),
        metadata={
            "source_url": "https://doc.rust-lang.org/book/ch10-03.html",
            "source_name": "book",
            "headings": "泛型 > 生命周期",
        },
    ),
    Chunk(
        chunk_id="book_0003",
        text=(
            "trait 定义了共享的行为。类型通过 impl Trait for Type 语法来实现 trait。"
            "trait 可以有默认实现，也可以用作参数约束。"
            "例如：fn notify(item: &impl Summary) { ... }"
        ),
        metadata={
            "source_url": "https://doc.rust-lang.org/book/ch10-02.html",
            "source_name": "book",
            "headings": "泛型 > trait",
        },
    ),
    Chunk(
        chunk_id="book_0004",
        text=(
            "智能指针包括 Box<T>、Rc<T> 和 RefCell<T>。Box<T> 用于堆上分配，"
            "Rc<T> 用于多重所有权，RefCell<T> 提供内部可变性。"
        ),
        metadata={
            "source_url": "https://doc.rust-lang.org/book/ch15-00.html",
            "source_name": "book",
            "headings": "智能指针",
        },
    ),
]


class TestExtractKeywords:
    """测试关键词提取"""

    def test_extract_from_ownership_text(self):
        """从所有权文本中提取关键词"""
        keywords = _extract_keywords(SAMPLE_CHUNKS[0].text)
        assert len(keywords) > 0
        # 应该包含 Rust 相关术语
        assert any("Rust" in kw for kw in keywords)

    def test_extract_from_code_text(self):
        """从含代码的文本中提取关键词"""
        keywords = _extract_keywords(SAMPLE_CHUNKS[2].text)
        assert len(keywords) > 0

    def test_max_keywords(self):
        """关键词数量不超过上限"""
        keywords = _extract_keywords(SAMPLE_CHUNKS[2].text, max_keywords=3)
        assert len(keywords) <= 3

    def test_empty_text(self):
        """空文本返回空列表"""
        keywords = _extract_keywords("")
        assert keywords == []


class TestFallbackQuestion:
    """测试降级题目生成"""

    def test_generates_question(self):
        """降级生成返回有效题目结构"""
        q = _generate_fallback_question(SAMPLE_CHUNKS[0], "概念解释")
        assert "question" in q
        assert "keywords" in q
        assert "source_chunk_id" in q
        assert q["source_chunk_id"] == "book_0000"
        assert len(q["question"]) > 0
        assert isinstance(q["keywords"], list)

    def test_all_question_types(self):
        """所有题型都能生成"""
        for qtype in ["概念解释", "代码补全", "对比分析", "术语定义"]:
            q = _generate_fallback_question(SAMPLE_CHUNKS[0], qtype)
            assert isinstance(q["question"], str)
            assert len(q["question"]) > 0

    def test_code_completion_has_code(self):
        """代码补全题型包含代码块"""
        q = _generate_fallback_question(SAMPLE_CHUNKS[4], "代码补全")
        assert "fn" in q["question"] or "}" in q["question"]


class TestGenerateQuestions:
    """测试 generate_questions 主接口（使用降级模式）"""

    def test_generates_correct_count(self):
        """生成指定数量的题目"""
        questions = generate_questions(SAMPLE_CHUNKS, n=4)
        assert len(questions) == 4

    def test_each_question_has_required_fields(self):
        """每道题包含必需字段"""
        questions = generate_questions(SAMPLE_CHUNKS, n=3)
        for q in questions:
            assert "question" in q
            assert "keywords" in q
            assert "source_chunk_id" in q
            assert isinstance(q["question"], str)
            assert isinstance(q["keywords"], list)
            assert isinstance(q["source_chunk_id"], str)
            assert len(q["question"]) > 0

    def test_source_chunk_ids_are_valid(self):
        """题目引用的 chunk_id 来自输入"""
        valid_ids = {c.chunk_id for c in SAMPLE_CHUNKS}
        questions = generate_questions(SAMPLE_CHUNKS, n=6)
        for q in questions:
            assert q["source_chunk_id"] in valid_ids

    def test_empty_chunks(self):
        """空列表返回空结果"""
        questions = generate_questions([], n=5)
        assert questions == []

    def test_more_than_available(self):
        """n 大于可用 chunk 数时仍正常工作（有放回抽样）"""
        questions = generate_questions(SAMPLE_CHUNKS, n=8)
        assert len(questions) == 8

    def test_question_types_varied(self):
        """题型多样化"""
        questions = generate_questions(SAMPLE_CHUNKS, n=6)
        # 检查至少出现 2 种以上题型
        qtype_counts = {}
        for q in questions:
            for qtype in ["概念解释", "代码补全", "对比分析", "术语定义"]:
                # 通过问题内容特征大致判断题型
                if "解释" in q["question"]:
                    qtype_counts["概念解释"] = qtype_counts.get("概念解释", 0) + 1
                    break
                elif "fn " in q["question"] or "TODO" in q["question"]:
                    qtype_counts["代码补全"] = qtype_counts.get("代码补全", 0) + 1
                    break
                elif "对比" in q["question"]:
                    qtype_counts["对比分析"] = qtype_counts.get("对比分析", 0) + 1
                    break
                elif "定义" in q["question"]:
                    qtype_counts["术语定义"] = qtype_counts.get("术语定义", 0) + 1
                    break
        # 至少有 2 种题型
        assert len(qtype_counts) >= 2, f"只有 {len(qtype_counts)} 种题型"


class TestGenerateQuestionsIntegration:
    """集成测试（需要 API Key）"""

    @pytest.mark.integration
    def test_llm_generation(self):
        """使用真实 LLM 生成题目"""
        from evaluation.question_generator import _get_api_key
        if not _get_api_key():
            pytest.skip("DEEPSEEK_API_KEY 未设置，跳过集成测试")

        questions = generate_questions(SAMPLE_CHUNKS[:2], n=4)
        assert len(questions) == 4
        for q in questions:
            assert len(q["question"]) > 0
            assert len(q["keywords"]) > 0
            assert q["source_chunk_id"] in {"book_0000", "book_0001"}
