"""
全形态分块器 —— 生成父子两层 chunk。

父块：按 h2/h3 标题切分的 section，完整内容供 LLM 使用。
子块：父块按 200-400 字符切分（句子/段落边界），供检索使用。
子块通过 parent_chunk_id 映射到父块。

数据形态：
- narrative: 叙述段落（按句子边界切分）
- code_unit:  代码块 + 紧随说明，最小检索单元
- specification: 规范条目，类似 code_unit 独立成子块
- structured: 列表/表格，按条目拆分子块
- note: 警告/注释框，合并到前一个 narrative（不独立）
"""

import re
from dataclasses import dataclass, field
from typing import Any

CHILD_MIN_CHARS = 80    # 子块最小字符数（低于此值强制合并）
CHILD_TARGET_MIN = 200  # 子块目标最小字符
CHILD_TARGET_MAX = 400  # 子块目标最大字符
PARENT_MAX_CHARS = 3000 # 父块最大字符（超过则拆分为多个父块）


@dataclass
class Chunk:
    """检索的最小单元。支持父子块模式：子块检索，父块返回 LLM。"""
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: str | None = None
    is_parent: bool = False


class NarrativeChunker:
    """全形态分块器 —— 生成父子两层"""

    def __init__(self,
                 child_min: int = CHILD_TARGET_MIN,
                 child_max: int = CHILD_TARGET_MAX):
        self.child_min = child_min
        self.child_max = child_max

    # ── 公共接口 ──────────────────────────────────────────

    def chunk_elements(
        self, elements: list[dict[str, Any]],
        source_url: str, source_name: str, title: str,
        file_index: int = 0,
    ) -> list[Chunk]:
        """
        将元素列表切分为父子 chunk 列表（包含父块和子块）。
        
        Args:
            file_index: 文件序号（用于生成全局唯一 chunk_id，避免多文件 ID 冲突）
        """
        if not elements:
            return []

        # 第一步：按 h2/h3 切分为 section
        sections = self._split_by_headings(elements, title)

        # 第二步：P0 过滤 git 元数据
        sections = [s for s in sections if not _is_metadata_noise(s)]

        # 注入 source_url 到每个 section（子块 metadata 需要）
        for sec in sections:
            sec["_source_url"] = source_url

        all_chunks = []
        parent_idx = 0

        for sec in sections:
            # 构建父块
            parent = self._make_parent(sec, source_url, source_name, parent_idx, file_index)
            if not parent:
                continue
            all_chunks.append(parent)

            # 构建子块
            children = self._make_children(sec, parent.chunk_id, source_name)
            all_chunks.extend(children)
            parent_idx += 1

        return all_chunks

    def get_parents(self, all_chunks: list[Chunk]) -> list[Chunk]:
        """从混合列表中提取父块。"""
        return [c for c in all_chunks if c.is_parent]

    def get_children(self, all_chunks: list[Chunk]) -> list[Chunk]:
        """从混合列表中提取子块（检索用）。"""
        return [c for c in all_chunks if not c.is_parent]

    # ── Section 切分 ──────────────────────────────────────

    @staticmethod
    def _split_by_headings(elements: list[dict], title: str) -> list[dict]:
        """按 h2/h3 边界切分为 section。"""
        sections = []
        current = {"headings": [title], "elements": []}

        for el in elements:
            if el["type"] == "heading" and el["level"] >= 2:
                if current["elements"]:
                    sections.append(current)
                current = {"headings": [title, el["text"]], "elements": []}
            else:
                current["elements"].append(el)

        if current["elements"]:
            sections.append(current)
        return sections

    # ── 父块构建 ──────────────────────────────────────────

    def _make_parent(self, section: dict, source_url: str,
                     source_name: str, idx: int, file_index: int = 0) -> Chunk | None:
        """构建父块：合并 section 所有内容。超过上限则拆分。"""
        # 构建完整文本（含标题链）
        headings_chain = " > ".join(section["headings"])
        text_parts = []
        for i, h in enumerate(section["headings"]):
            prefix = "#" if i == 0 else "#" * (i + 1)
            text_parts.append(f"{prefix} {h}")
        text_parts.append(_build_rich_text(section["elements"]))
        text = "\n\n".join(text_parts)
        if not text.strip():
            return None

        # 超过上限则按段落边界拆分（简化：截断到最后一个完整段落）
        if len(text) > PARENT_MAX_CHARS:
            # 在 PARENT_MAX_CHARS 附近找最近的段落边界
            cutoff = text.rfind('\n\n', 0, PARENT_MAX_CHARS)
            if cutoff > PARENT_MAX_CHARS // 2:
                text = text[:cutoff]
            else:
                text = text[:PARENT_MAX_CHARS]

        headings_chain = " > ".join(section["headings"])
        morph = _detect_section_morphology(section["elements"])

        return Chunk(
            chunk_id=f"{source_name}_f{file_index:04d}p{idx:04d}",
            text=text,
            metadata={
                "source_url": source_url,
                "source_name": source_name,
                "headings": headings_chain,
                "morphology": morph,
                "char_count": len(text),
                "chunk_count": 0,  # 子块数，后续填充
            },
            is_parent=True,
        )

    # ── 子块构建 ──────────────────────────────────────────

    def _make_children(self, section: dict, parent_id: str,
                       source_name: str) -> list[Chunk]:
        """构建子块：按形态和大小切分。"""
        elements = section["elements"]
        children = []
        buffer = []
        buf_len = 0
        child_idx = 0
        # 从父块 ID 派生子块 ID 前缀（确保全局唯一）
        # parent_id 格式: book_f0000p0000 → prefix: book_f0000, section: 0000
        cid_prefix = parent_id.rsplit("p", 1)[0]  # e.g. "book_f0000"
        cid_section = parent_id.rsplit("p", 1)[1]  # e.g. "0000"
        # 子块文本前缀（保留标题上下文，避免检索时丢失关键词）
        heading_prefix = " > ".join(section["headings"]) + "\n"

        def flush():
            nonlocal buffer, buf_len, child_idx
            if not buffer:
                return
            text = " ".join(buffer)
            if len(text) >= CHILD_MIN_CHARS:
                morph = _detect_paragraph_morphology(buffer)
                children.append(Chunk(
                    chunk_id=f"{cid_prefix}c{cid_section}_{child_idx:02d}",
                    text=text,
                    metadata={
                        "source_url": section.get("_source_url", ""),
                        "source_name": source_name,
                        "headings": " > ".join(section["headings"]),
                        "morphology": morph,
                        "char_count": len(text),
                    },
                    parent_chunk_id=parent_id,
                ))
                child_idx += 1
            buffer = []
            buf_len = 0

        for el in elements:
            morph = el.get("morphology", "narrative")

            if morph == "code_unit":
                # 代码单元：独立成子块（不要拆分代码）
                flush()
                code_text = _element_text(el)
                children.append(Chunk(
                    chunk_id=f"{cid_prefix}c{cid_section}_{child_idx:02d}",
                    text=code_text,
                    metadata={
                        "source_url": section.get("_source_url", ""),
                        "source_name": source_name,
                        "headings": " > ".join(section["headings"]),
                        "morphology": "code_unit",
                        "char_count": len(code_text),
                    },
                    parent_chunk_id=parent_id,
                ))
                child_idx += 1
                continue

            if morph == "specification":
                # 规范条目：类似 code_unit，独立成子块
                flush()
                spec_text = _element_text(el)
                children.append(Chunk(
                    chunk_id=f"{cid_prefix}c{cid_section}_{child_idx:02d}",
                    text=spec_text,
                    metadata={
                        "source_url": section.get("_source_url", ""),
                        "source_name": source_name,
                        "headings": " > ".join(section["headings"]),
                        "morphology": "specification",
                        "char_count": len(spec_text),
                    },
                    parent_chunk_id=parent_id,
                ))
                child_idx += 1
                continue

            if morph == "note":
                # P2: note 合并到 buffer（不独立）
                buffer.append(_element_text(el))
                buf_len += len(buffer[-1])
                continue

            if morph == "structured":
                # 列表/表格：按条目拆分子块（每 3-5 个条目一个子块）
                flush()
                items = el.get("items", [])
                batch = []
                for i, item in enumerate(items):
                    batch.append(item)
                    if len(batch) >= 4:
                        children.append(Chunk(
                            chunk_id=f"{cid_prefix}c{cid_section}_{child_idx:02d}",
                            text=heading_prefix + "\n".join(batch),
                            metadata={
                                "source_url": section.get("_source_url", ""),
                                "source_name": source_name,
                                "headings": " > ".join(section["headings"]),
                                "morphology": "structured",
                                "char_count": sum(len(b) for b in batch),
                            },
                            parent_chunk_id=parent_id,
                        ))
                        child_idx += 1
                        batch = []
                if batch:
                    children.append(Chunk(
                        chunk_id=f"{cid_prefix}c{cid_section}_{child_idx:02d}",
                        text=heading_prefix + "\n".join(batch),
                        metadata={
                            "source_url": section.get("_source_url", ""),
                            "source_name": source_name,
                            "headings": " > ".join(section["headings"]),
                            "morphology": "structured",
                            "char_count": sum(len(b) for b in batch),
                        },
                        parent_chunk_id=parent_id,
                    ))
                    child_idx += 1
                continue

            # narrative / mixed → 累积到 buffer
            text = _element_text(el)
            buffer.append(text)
            buf_len += len(text)

            # P4: 超过上限时 flush
            if buf_len >= self.child_max:
                flush()

        flush()  # 收尾

        return children


# ── 辅助函数 ──────────────────────────────────────────

def _build_rich_text(elements: list[dict]) -> str:
    """构建富文本（父块用）。"""
    parts = []
    for el in elements:
        t = el["type"]
        if t == "heading":
            parts.append(f"{'#' * el['level']} {el['text']}")
        elif t == "paragraph":
            parts.append(el["text"])
        elif t == "code_block":
            lang = el.get("language") or ""
            parts.append(f"```{lang}\n{el['code']}\n```")
        elif t == "list":
            for item in el.get("items", []):
                parts.append(f"- {item}")
        elif t == "blockquote":
            parts.append(f"> {el['text']}")
        elif t == "table":
            parts.append(_table_to_text(el))
    return "\n\n".join(parts)


def _element_text(el: dict) -> str:
    """提取单个元素的纯文本。"""
    t = el["type"]
    if t in ("paragraph", "blockquote"):
        return el["text"]
    if t == "code_block":
        return el["code"]
    if t == "list":
        return " ".join(el.get("items", []))
    if t == "heading":
        return el["text"]
    if t == "table":
        return _table_to_text(el)
    return ""


def _table_to_text(el: dict) -> str:
    """表格转文本。"""
    lines = []
    if el.get("headers"):
        lines.append(" | ".join(el["headers"]))
        lines.append(" | ".join("---" for _ in el["headers"]))
    for row in el.get("rows", []):
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _is_metadata_noise(section: dict) -> bool:
    """P0: 检测是否为 git 元数据等噪音 section。"""
    for el in section.get("elements", []):
        text = _element_text(el)
        if "commit:" in text.lower() and len(text) < 200:
            return True
        if text.strip().startswith("keywords.md commit:"):
            return True
    return False


def _detect_section_morphology(elements: list[dict]) -> str:
    """从 section 推断 morphology。"""
    for el in elements:
        m = el.get("morphology", "")
        if m in ("code_unit", "specification", "structured", "note"):
            return m
    return "narrative"


def _detect_paragraph_morphology(buffer: list[str]) -> str:
    """从 buffer 推断 morphology（简化：取第一个非空元素）。"""
    # 简化处理：子块 morphology 从元素来，buffer 中混合可能丢失
    return "narrative"
