"""
测试 LLM 生成器 —— prompt 构建 + API 调用（可选）。
"""
import pytest
from src.chunkers.narrative import Chunk
from src.llm.generator import build_prompt, RAGGenerator

SAMPLE_CHUNKS = [
    Chunk(
        chunk_id="test_0000",
        text="Rust 的所有权系统确保内存安全。每个值在 Rust 中都有一个所有者。",
        metadata={"source_url": "url1", "source_name": "book",
                    "headings": "所有权 > 所有权规则"},
    ),
    Chunk(
        chunk_id="test_0001",
        text="借用允许在不转移所有权的情况下使用值的引用。",
        metadata={"source_url": "url1", "source_name": "book",
                    "headings": "所有权 > 引用与借用"},
    ),
]


class TestBuildPrompt:
    """测试 prompt 构建"""

    def test_prompt_contains_query(self):
        """prompt 包含用户问题"""
        prompt = build_prompt("什么是所有权？", SAMPLE_CHUNKS)
        assert "什么是所有权？" in prompt

    def test_prompt_contains_chunks(self):
        """prompt 包含检索到的 chunk 文本"""
        prompt = build_prompt("Rust 所有权", SAMPLE_CHUNKS)
        assert "所有权系统" in prompt
        assert "借用" in prompt

    def test_prompt_contains_system_instructions(self):
        """prompt 包含系统指令"""
        prompt = build_prompt("问题", SAMPLE_CHUNKS)
        assert "根据提供的文档内容回答" in prompt

    def test_prompt_max_chunks_limit(self):
        """max_chunks 限制使用的 chunk 数量"""
        prompt = build_prompt("Rust 所有权", SAMPLE_CHUNKS, max_chunks=1)
        # 只有一个 [文档 1]，没有 [文档 2]
        assert "[文档 1]" in prompt
        assert "[文档 2]" not in prompt

    def test_empty_chunks_produces_empty_context(self):
        """空 chunk 列表仍生成有效 prompt"""
        prompt = build_prompt("Rust 所有权", [])
        assert "Rust 所有权" in prompt
        # 上下文区应该没有文档
        assert "[文档 1]" not in prompt


class TestRAGGenerator:
    """集成测试（需要 API Key）"""

    @pytest.mark.integration
    def test_api_call(self):
        """真实 API 调用测试"""
        try:
            gen = RAGGenerator()
        except RuntimeError:
            pytest.skip("DEEPSEEK_API_KEY 未设置，跳过集成测试")

        answer = gen.generate("Rust 的所有权是什么？", SAMPLE_CHUNKS)
        assert answer is not None
        assert len(answer) > 0
