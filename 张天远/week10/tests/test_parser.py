"""
测试 HTML 解析器 —— HTML → 结构化 JSON 中间格式。
"""

import json
import pytest
from src.parser import parse_html_to_json, ParsedDocument


MOCK_MDBOOK_HTML = """<!DOCTYPE html>
<html>
<body>
<div id="content" class="content">
<main>
    <h1>什么是所有权？</h1>
    <p>Rust 的核心功能（之一）是<strong>所有权</strong>（ownership）。
    它使 Rust 无需垃圾回收器即可保证内存安全。</p>
    <pre><code class="language-rust">fn main() {
    let s = String::from("hello");
    println!("{}", s);
}</code></pre>
    <h2>所有权规则</h2>
    <ul>
        <li>Rust 中每一个值都有一个<strong>所有者</strong></li>
        <li>同一时间只能有一个所有者</li>
        <li>当所有者离开作用域，值被丢弃</li>
    </ul>
    <p>这里是一些内联代码：<code>let x = 5;</code> 是变量绑定。</p>
    <blockquote>
        <p><strong>注意：</strong>所有权是 Rust 最独特的特性。</p>
    </blockquote>
</main>
</div>
</body>
</html>"""

SOURCE_URL = "https://rustwiki.org/zh-CN/book/ch04-01.html"


class TestParser:
    """HTML → JSON 解析测试"""

    def test_parses_title(self):
        """提取页面标题 (h1)"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        assert doc.title == "什么是所有权？"
        assert doc.source_url == SOURCE_URL
        assert doc.source_name == "book"

    def test_parses_headings(self):
        """提取 h1/h2 标题层级"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        headings = [e for e in doc.elements if e["type"] == "heading"]
        assert len(headings) == 2
        assert headings[0] == {"type": "heading", "level": 1, "text": "什么是所有权？"}
        assert headings[1] == {"type": "heading", "level": 2, "text": "所有权规则"}

    def test_parses_paragraphs(self):
        """提取段落文本（保留内联格式）"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        paragraphs = [e for e in doc.elements if e["type"] == "paragraph"]
        assert len(paragraphs) >= 2
        # 段落应保留内联格式（粗体等）
        assert "所有权" in paragraphs[0]["text"]

    def test_parses_code_blocks(self):
        """提取代码块 + 语言标注"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        code_blocks = [e for e in doc.elements if e["type"] == "code_block"]
        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "rust"
        assert "fn main()" in code_blocks[0]["code"]

    def test_parses_lists(self):
        """提取无序/有序列表"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        lists = [e for e in doc.elements if e["type"] == "list"]
        assert len(lists) == 1
        assert lists[0]["ordered"] is False
        assert len(lists[0]["items"]) == 3
        assert "所有者" in lists[0]["items"][0]

    def test_parses_blockquote(self):
        """提取引用块（mdBook 的注释/警告）"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        quotes = [e for e in doc.elements if e["type"] == "blockquote"]
        assert len(quotes) == 1
        assert "所有权" in quotes[0]["text"]

    def test_elements_order_preserved(self):
        """验证元素顺序与 HTML 一致"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        types = [e["type"] for e in doc.elements]
        assert types == [
            "heading",    # h1
            "paragraph",  # 第一段
            "code_block", # 代码块
            "heading",    # h2
            "list",       # ul
            "paragraph",  # 内联代码段落
            "blockquote", # 引用块
        ]

    def test_empty_content_returns_empty_elements(self):
        """空内容返回空元素列表"""
        html = "<html><body><div id='content'><main></main></div></body></html>"
        doc = parse_html_to_json(html, SOURCE_URL, "book")
        assert doc.elements == []

    def test_paragraph_has_narrative_morphology(self):
        """普通段落 morphology 为 narrative"""
        html = '<html><body><div id="content"><main><p>Rust 是一种语言。</p></main></div></body></html>'
        doc = parse_html_to_json(html, SOURCE_URL, "book")
        para = doc.elements[0]
        assert para["morphology"] == "narrative"

    def test_code_block_has_code_unit_morphology(self):
        """代码块 morphology 为 code_unit"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        code = [e for e in doc.elements if e["type"] == "code_block"][0]
        assert code["morphology"] == "code_unit"

    def test_list_has_structured_morphology(self):
        """列表 morphology 为 structured"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        lst = [e for e in doc.elements if e["type"] == "list"][0]
        assert lst["morphology"] == "structured"

    def test_blockquote_has_note_morphology(self):
        """blockquote morphology 为 note"""
        doc = parse_html_to_json(MOCK_MDBOOK_HTML, SOURCE_URL, "book")
        bq = [e for e in doc.elements if e["type"] == "blockquote"][0]
        assert bq["morphology"] == "note"

    def test_inline_code_paragraph_is_mixed(self):
        """含内联代码的段落 morphology 为 mixed"""
        html = '<html><body><div id="content"><main><p>使用 <code>let x = 5;</code> 声明变量。</p></main></div></body></html>'
        doc = parse_html_to_json(html, SOURCE_URL, "book")
        para = doc.elements[0]
        assert para["morphology"] == "mixed"
