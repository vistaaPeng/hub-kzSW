"""
测试数据下载器。
"""

import tempfile
from pathlib import Path

import pytest

from src.downloader import extract_chapter_links

# Mock mdBook HTML 侧边栏
MOCK_MDBOOK_HTML = """<!DOCTYPE html>
<html>
<head><title>Test Book</title></head>
<body>
    <nav>
        <ol class="chapter">
            <li><a href="ch01.html">第一章</a></li>
            <li><a href="ch02.html">第二章</a>
                <ol>
                    <li><a href="ch02-01.html">2.1 节</a></li>
                    <li><a href="ch02-02.html">2.2 节</a></li>
                </ol>
            </li>
            <li><a href="ch03.html">第三章</a></li>
            <li><a href="https://other-site.com/external.html">外部链接</a></li>
            <li><a href="#section-4">锚点链接</a></li>
        </ol>
    </nav>
</body>
</html>"""

BASE_URL = "https://rustwiki.org/zh-CN/book/"


class TestExtractChapterLinks:
    """测试侧边栏链接提取"""

    def test_extracts_html_links(self):
        """提取 .html 链接"""
        links = extract_chapter_links(MOCK_MDBOOK_HTML, BASE_URL)
        expected = [
            "https://rustwiki.org/zh-CN/book/ch01.html",
            "https://rustwiki.org/zh-CN/book/ch02-01.html",
            "https://rustwiki.org/zh-CN/book/ch02-02.html",
            "https://rustwiki.org/zh-CN/book/ch02.html",
            "https://rustwiki.org/zh-CN/book/ch03.html",
        ]
        assert sorted(links) == sorted(expected)

    def test_resolves_relative_paths(self):
        """相对路径自动转为绝对 URL"""
        html = '<ol class="chapter"><li><a href="foo/bar.html">Test</a></li></ol>'
        links = extract_chapter_links(html, "https://example.com/docs/")
        assert links == ["https://example.com/docs/foo/bar.html"]

    def test_skips_non_html_links(self):
        """跳过锚点和非 HTML 链接"""
        html = '<ol class="chapter"><li><a href="#intro">Intro</a></li><li><a href="readme.md">MD</a></li></ol>'
        links = extract_chapter_links(html, BASE_URL)
        assert links == []

    def test_no_chapter_ol_returns_empty(self):
        """没有 chapter 类的 ol 时返回空列表"""
        html = "<html><body><p>No sidebar</p></body></html>"
        links = extract_chapter_links(html, BASE_URL)
        assert links == []

    def test_filters_external_domains(self):
        """过滤跨域链接"""
        html = (
            '<ol class="chapter">'
            '<li><a href="local.html">Local</a></li>'
            '<li><a href="https://evil.com/page.html">Evil</a></li>'
            "</ol>"
        )
        links = extract_chapter_links(html, BASE_URL)
        assert links == ["https://rustwiki.org/zh-CN/book/local.html"]
