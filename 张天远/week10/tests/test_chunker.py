"""
测试分块器 —— 父子块模式。
新接口：chunk_elements 返回父块+子块混合列表。
"""
import pytest
from src.chunkers.narrative import NarrativeChunker, Chunk

SAMPLE_ELEMENTS = [
    {"type": "heading", "level": 1, "text": "所有权"},
    {"type": "paragraph", "text": "所有权是 Rust 最独特的特性。", "morphology": "narrative"},
    {"type": "heading", "level": 2, "text": "所有权规则", "morphology": "narrative"},
    {"type": "paragraph", "text": "Rust 中每一个值都有一个所有者。", "morphology": "narrative"},
    {"type": "list", "ordered": False, "morphology": "structured",
     "items": ["规则1: 同一时间只有一个所有者", "规则2: 离开作用域即丢弃"]},
    {"type": "blockquote", "text": "注意：所有权影响所有 Rust 代码。", "morphology": "note"},
    {"type": "heading", "level": 2, "text": "变量作用域", "morphology": "narrative"},
    {"type": "paragraph", "text": "作用域是一个变量在程序中有效的范围。", "morphology": "narrative"},
    {"type": "code_block", "language": "rust", "code": "{\n    let s = \"hello\";\n}", "morphology": "code_unit"},
]

SAMPLE_SOURCE_URL = "https://rustwiki.org/zh-CN/book/ch04-01.html"
SAMPLE_SOURCE_NAME = "book"
SAMPLE_TITLE = "所有权"


class TestParentChildChunks:
    """父子块模式测试"""

    def test_produces_parents_and_children(self):
        chunker = NarrativeChunker()
        all_chunks = chunker.chunk_elements(
            SAMPLE_ELEMENTS, SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, SAMPLE_TITLE
        )
        parents = chunker.get_parents(all_chunks)
        children = chunker.get_children(all_chunks)
        assert len(parents) >= 1  # 至少 1 个父块
        assert len(children) >= 2  # 子块数 >= 父块数
        assert all(p.is_parent for p in parents)
        assert all(not c.is_parent for c in children)

    def test_children_link_to_parent(self):
        chunker = NarrativeChunker()
        all_chunks = chunker.chunk_elements(
            SAMPLE_ELEMENTS, SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, SAMPLE_TITLE
        )
        parents = chunker.get_parents(all_chunks)
        children = chunker.get_children(all_chunks)
        parent_ids = {p.chunk_id for p in parents}
        for c in children:
            assert c.parent_chunk_id in parent_ids, f"child {c.chunk_id} parent not found"

    def test_parent_has_full_content(self):
        chunker = NarrativeChunker()
        all_chunks = chunker.chunk_elements(
            SAMPLE_ELEMENTS, SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, SAMPLE_TITLE
        )
        parents = chunker.get_parents(all_chunks)
        parent_texts = " ".join(p.text for p in parents)
        assert "所有权规则" in parent_texts
        assert "变量作用域" in parent_texts
        assert "let s = \"hello\"" in parent_texts

    def test_children_are_small_chunks(self):
        chunker = NarrativeChunker(child_min=80, child_max=300)
        all_chunks = chunker.chunk_elements(
            SAMPLE_ELEMENTS, SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, SAMPLE_TITLE
        )
        children = chunker.get_children(all_chunks)
        for c in children:
            assert len(c.text) <= 600  # 允许适度超出（不可拆分的内容如代码块）

    def test_code_unit_remains_independent(self):
        chunker = NarrativeChunker()
        all_chunks = chunker.chunk_elements(
            SAMPLE_ELEMENTS, SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, SAMPLE_TITLE
        )
        children = chunker.get_children(all_chunks)
        code_children = [c for c in children if c.metadata.get("morphology") == "code_unit"]
        assert len(code_children) >= 1
        for cc in code_children:
            assert "let s =" in cc.text or "fn main" in cc.text

    def test_empty_elements(self):
        chunker = NarrativeChunker()
        all_chunks = chunker.chunk_elements([], SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, "")
        assert all_chunks == []

    def test_struct_metadata(self):
        """metadata 包含必要字段"""
        chunker = NarrativeChunker()
        all_chunks = chunker.chunk_elements(
            [{"type": "paragraph", "text": "Hello Rust", "morphology": "narrative"}],
            SAMPLE_SOURCE_URL, SAMPLE_SOURCE_NAME, "Test"
        )
        parent = chunker.get_parents(all_chunks)[0]
        assert parent.metadata["source_url"] == SAMPLE_SOURCE_URL
        assert parent.metadata["source_name"] == SAMPLE_SOURCE_NAME
        assert parent.is_parent is True
