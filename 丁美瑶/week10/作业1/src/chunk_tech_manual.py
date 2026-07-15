import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed_tech"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks_tech"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


def chunk_semantic(blocks: list[dict], max_chunk_size: int = 800, min_chunk_size: int = 100) -> Iterator[dict]:
    buffer_blocks = []
    buffer_len    = 0

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        meta = {
            "page_num": buf[0]["page_num"],
            "block_types": list({b["block_type"] for b in buf}),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])

        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": block["page_num"],
                    "block_types": ["table"],
                }
            }
            continue

        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len    = 0

        buffer_blocks.append(block)
        buffer_len += blen

    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


STRATEGY = "semantic"


def build_chunk_id(filename: str, idx: int) -> str:
    safe_name = filename.replace(".", "_").replace(" ", "_")
    return f"tech_{safe_name}_{idx:05d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    blocks   = data.get("blocks", [])
    filename = data.get("filename", parsed_path.stem)

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content":  text_chunk,
                "metadata": {"block_types": ["text"]}
            })
    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)
    else:
        raise ValueError(f"未知策略: {strategy}")

    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(filename, idx)
        chunk["chunk_id"]              = chunk_id
        chunk["metadata"]["filename"]  = filename
        chunk["metadata"]["strategy"]  = strategy
        chunk["metadata"]["source_file"] = parsed_path.name
        result.append(chunk)

    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  → {len(result)} 个 chunk，已保存 {out_path.name}")
    return result


def main():
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error(f"没有找到解析结果，请先运行 parse_tech_manual.py")
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)

    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    logger.info(f"其中表格块: {table_count}")


if __name__ == "__main__":
    main()
