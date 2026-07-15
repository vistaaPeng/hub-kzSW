"""
HTML 解析器 —— 将 mdBook 生成的 HTML 转换为结构化 JSON 中间格式。

JSON 中间格式用于后续分块器消费，解耦 HTML 解析和分块逻辑。
每个 element 标注 morphology（数据形态），分块器据此路由策略。
"""

from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag


@dataclass
class ParsedDocument:
    """HTML 解析后的结构化文档"""
    source_url: str
    source_name: str
    title: str
    elements: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        import json
        from pathlib import Path
        data = {
            "source_url": self.source_url,
            "source_name": self.source_name,
            "title": self.title,
            "elements": self.elements,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def parse_html_to_json(html: str, source_url: str, source_name: str) -> ParsedDocument:
    """解析 mdBook HTML → ParsedDocument，每个 element 带 morphology 标注。"""
    soup = BeautifulSoup(html, "html.parser")

    # 找内容区
    content_div = soup.find("div", id="content")
    if content_div is None:
        content_div = soup.find("main")
    if content_div is None:
        content_div = soup

    main = content_div.find("main") if content_div.name != "main" else content_div
    if main is None:
        main = content_div

    # 提取标题：优先从 main 的 h1，否则从整个页面找
    title = ""
    h1 = main.find("h1")
    if not h1:
        h1 = content_div.find("h1")
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    elements = []
    for child in main.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        element = _parse_element(child)
        if element:
            elements.append(element)

    return ParsedDocument(
        source_url=source_url,
        source_name=source_name,
        title=title,
        elements=elements,
    )


def _parse_element(tag: Tag) -> dict[str, Any] | None:
    """解析单个 HTML 元素，标注 morphology（数据形态）。"""
    name = tag.name

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return {
            "type": "heading",
            "level": int(name[1]),
            "text": _clean_text(tag.get_text()),
        }

    if name == "p":
        text = _clean_text(tag.get_text())
        if not text:
            return None
        # 检测段落中的内容特征来判定 morphology
        morph = _detect_paragraph_morphology(tag, text)
        return {"type": "paragraph", "morphology": morph, "text": text}

    if name == "pre":
        code_tag = tag.find("code")
        language = None
        if code_tag:
            classes = code_tag.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    language = cls.replace("language-", "")
                    break
        code_text = code_tag.get_text() if code_tag else tag.get_text()
        return {
            "type": "code_block",
            "morphology": "code_unit",
            "language": language,
            "code": code_text,
        }

    if name in ("ul", "ol"):
        items = [_clean_text(li.get_text()) for li in tag.find_all("li", recursive=False)]
        return {
            "type": "list",
            "morphology": "structured",
            "ordered": name == "ol",
            "items": items,
        } if items else None

    if name == "blockquote":
        text = _clean_text(tag.get_text())
        # mdBook 的 blockquote 通常用于警告/注释
        return {"type": "blockquote", "morphology": "note", "text": text} if text else None

    if name == "table":
        headers, rows = _parse_table(tag)
        return {
            "type": "table",
            "morphology": "structured",
            "headers": headers,
            "rows": rows,
        } if rows else None

    # 未识别元素
    text = _clean_text(tag.get_text())
    if text and len(text) > 10:
        return {"type": "paragraph", "morphology": "narrative", "text": text}
    return None


def _detect_paragraph_morphology(tag: Tag, text: str) -> str:
    """检测段落的数据形态：narrative / mixed / specification。"""
    # 内嵌代码（`code` 标签）→ mixed
    if tag.find("code"):
        return "mixed"

    # 检测规范条目特征：短句 + 冒号 + 定义模式
    # 如 "let 语句: ..." 或 "表达式: ..."
    if len(text) < 150 and (":" in text or "：" in text):
        if any(kw in text for kw in ["语法", "句法", "表达式", "语句", "类型"]):
            return "specification"

    return "narrative"


def _parse_table(tag: Tag) -> tuple[list[str], list[list[str]]]:
    """解析 <table> → (headers, rows)。"""
    headers, rows = [], []
    thead = tag.find("thead")
    if thead:
        headers = [_clean_text(th.get_text()) for th in thead.find_all("th")]
    tbody = tag.find("tbody") or tag
    for tr in tbody.find_all("tr"):
        if thead and tr.parent and tr.parent.name == "thead":
            continue
        row = [_clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
        if row:
            rows.append(row)
    return headers, rows


def _clean_text(text: str) -> str:
    import re
    return re.sub(r'\s+', ' ', text).strip()
