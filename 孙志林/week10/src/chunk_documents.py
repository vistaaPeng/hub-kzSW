"""
文档分块模块：对解析后的文本做分块处理

三种分块策略：
  A. fixed      - 固定大小分块（简单但会切断语义）
  B. semantic   - 语义分块（按段落/章节边界切，保留语义完整性）
  C. hierarchical - 层级分块（父子块，父块召回上下文，子块精确匹配）

每个chunk包含：
  - chunk_id    唯一标识
  - content     文本内容（供embedding）
  - metadata    元信息（stock_code, year, page_num, section等）
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterator[str]:
    """策略A：固定大小分块

    按字符数切块，相邻块有重叠。
    缺点：无视句子/段落边界，表格会被切断。
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


def chunk_semantic(blocks: list[dict], max_chunk_size: int = 800, min_chunk_size: int = 100) -> Iterator[dict]:
    """策略B：语义分块

    按解析结构分块：遇到标题强制切块，段落尽量合并到max_chunk_size以内。
    优点：保留语义完整性，章节边界清晰。
    """
    buffer_blocks = []
    buffer_len = 0

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        if len(content) < min_chunk_size:
            return None
        meta = {
            "page_num": buf[0]["page_num"],
            "section": " > ".join(buf[0]["section_path"]) if buf[0].get("section_path") else "",
            "block_types": list({b["block_type"] for b in buf}),
            "is_ocr": any(b.get("is_ocr", False) for b in buf),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen = len(block["content"])

        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result:
                    yield result
                buffer_blocks = []
            block_copy = dict(block)
            block_copy["metadata"] = {
                "page_num": block["page_num"],
                "section": " > ".join(block["section_path"]) if block.get("section_path") else "",
                "block_types": ["title"],
                "is_ocr": block.get("is_ocr", False),
            }
            yield {"content": block["content"], "metadata": block_copy["metadata"]}
            continue

        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result:
                    yield result
                buffer_blocks = []
            table_meta = {
                "page_num": block["page_num"],
                "section": " > ".join(block["section_path"]) if block.get("section_path") else "",
                "block_types": ["table"],
                "is_ocr": block.get("is_ocr", False),
            }
            yield {"content": block["content"], "metadata": table_meta}
            continue

        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result:
                yield result
            buffer_blocks = [block]
            buffer_len = blen
        else:
            buffer_blocks.append(block)
            buffer_len += blen

    if buffer_blocks:
        result = flush(buffer_blocks)
        if result:
            yield result


def chunk_hierarchical(blocks: list[dict], parent_size: int = 1500, child_size: int = 400) -> Iterator[dict]:
    """策略C：层级分块

    创建父子块结构：
    - 子块：小粒度，用于精确匹配
    - 父块：包含子块及其上下文，用于召回更丰富的信息

    优点：兼顾召回效果和精确匹配。
    """
    semantic_chunks = list(chunk_semantic(blocks, max_chunk_size=child_size))

    parent_buffer = []
    parent_len = 0

    for i, child in enumerate(semantic_chunks):
        child_len = len(child["content"])

        if parent_len + child_len > parent_size and parent_buffer:
            parent_content = "\n\n".join(c["content"] for c in parent_buffer)
            parent_meta = {
                "page_num": parent_buffer[0]["metadata"]["page_num"],
                "section": parent_buffer[0]["metadata"]["section"],
                "block_types": ["hierarchical"],
                "is_ocr": any(c["metadata"]["is_ocr"] for c in parent_buffer),
            }
            yield {
                "content": parent_content,
                "metadata": parent_meta,
                "is_parent": True,
                "child_indices": [j for j in range(len(semantic_chunks)) if semantic_chunks[j] in parent_buffer],
            }
            parent_buffer = [child]
            parent_len = child_len
        else:
            parent_buffer.append(child)
            parent_len += child_len

        child["metadata"]["is_parent"] = False
        child["metadata"]["has_parent"] = True
        yield child

    if parent_buffer:
        parent_content = "\n\n".join(c["content"] for c in parent_buffer)
        parent_meta = {
            "page_num": parent_buffer[0]["metadata"]["page_num"],
            "section": parent_buffer[0]["metadata"]["section"],
            "block_types": ["hierarchical"],
            "is_ocr": any(c["metadata"]["is_ocr"] for c in parent_buffer),
        }
        yield {"content": parent_content, "metadata": parent_meta, "is_parent": True}


def process_all_docs(strategy: str = "semantic"):
    """处理所有解析后的文档"""
    all_chunks = []
    parsed_files = list(PARSED_DIR.glob("*.json"))

    logger.info(f"Processing {len(parsed_files)} documents with strategy: {strategy}")

    for pf in parsed_files:
        with open(pf, "r", encoding="utf-8") as f:
            doc = json.load(f)

        stock_code = doc.get("stock_code", "")
        year = doc.get("year", "")
        blocks = doc.get("blocks", [])

        if strategy == "fixed":
            full_text = "\n\n".join(b["content"] for b in blocks)
            chunks = list(chunk_fixed(full_text))
            for content in chunks:
                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "content": content,
                    "stock_code": stock_code,
                    "year": year,
                    "page_num": -1,
                    "section": "",
                    "block_type": "text",
                    "is_ocr": False,
                    "strategy": strategy,
                })
        elif strategy == "semantic":
            for chunk in chunk_semantic(blocks):
                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "content": chunk["content"],
                    "stock_code": stock_code,
                    "year": year,
                    "page_num": chunk["metadata"]["page_num"],
                    "section": chunk["metadata"]["section"],
                    "block_type": ",".join(chunk["metadata"]["block_types"]),
                    "is_ocr": chunk["metadata"]["is_ocr"],
                    "strategy": strategy,
                })
        elif strategy == "hierarchical":
            for chunk in chunk_hierarchical(blocks):
                is_parent = chunk.get("is_parent", False)
                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "content": chunk["content"],
                    "stock_code": stock_code,
                    "year": year,
                    "page_num": chunk["metadata"]["page_num"],
                    "section": chunk["metadata"]["section"],
                    "block_type": ",".join(chunk["metadata"]["block_types"]),
                    "is_ocr": chunk["metadata"]["is_ocr"],
                    "strategy": strategy,
                    "is_parent": is_parent,
                })

    output_path = CHUNKS_DIR / f"all_{strategy}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Generated {len(all_chunks)} chunks, saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Document chunking")
    parser.add_argument("--strategy", type=str, default="semantic", choices=["fixed", "semantic", "hierarchical"])
    args = parser.parse_args()
    process_all_docs(strategy=args.strategy)