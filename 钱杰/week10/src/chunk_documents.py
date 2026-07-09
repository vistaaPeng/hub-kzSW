import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
PARSED_DIR = BASE_DIR / "data" / "parsed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "semantic"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEMANTIC_CHUNK_SIZE = 800

TARGET_DOCS = {"ai_basics", "climate_change", "python_programming", "space_exploration"}


def chunk_semantic(blocks: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    chunk_id = 0
    buffer = []
    buffer_length = 0
    
    for block in blocks:
        content = block.get("content", "")
        block_type = block.get("block_type", "text")
        
        if block_type == "title":
            if buffer:
                chunk_content = "\n".join([b["content"] for b in buffer])
                chunks.append({
                    "chunk_id": f"{meta.get('doc_type', '')}_{chunk_id:04d}",
                    "content": chunk_content.strip(),
                    "metadata": {
                        "doc_type": meta.get("doc_type", ""),
                        "title": meta.get("title", ""),
                        "page_num": buffer[0].get("page_num", 0),
                        "section": "",
                        "block_types": [b.get("block_type", "text") for b in buffer],
                        "is_ocr": any(b.get("is_ocr", False) for b in buffer),
                        "strategy": "semantic",
                        "source_file": meta.get("filename", ""),
                    },
                })
                chunk_id += 1
                buffer = []
                buffer_length = 0
            
            buffer.append(block)
            buffer_length += len(content)
        
        else:
            if buffer_length + len(content) > SEMANTIC_CHUNK_SIZE:
                chunk_content = "\n".join([b["content"] for b in buffer])
                chunks.append({
                    "chunk_id": f"{meta.get('doc_type', '')}_{chunk_id:04d}",
                    "content": chunk_content.strip(),
                    "metadata": {
                        "doc_type": meta.get("doc_type", ""),
                        "title": meta.get("title", ""),
                        "page_num": buffer[0].get("page_num", 0),
                        "section": "",
                        "block_types": [b.get("block_type", "text") for b in buffer],
                        "is_ocr": any(b.get("is_ocr", False) for b in buffer),
                        "strategy": "semantic",
                        "source_file": meta.get("filename", ""),
                    },
                })
                chunk_id += 1
                buffer = []
                buffer_length = 0
            
            buffer.append(block)
            buffer_length += len(content)
    
    if buffer:
        chunk_content = "\n".join([b["content"] for b in buffer])
        chunks.append({
            "chunk_id": f"{meta.get('doc_type', '')}_{chunk_id:04d}",
            "content": chunk_content.strip(),
            "metadata": {
                "doc_type": meta.get("doc_type", ""),
                "title": meta.get("title", ""),
                "page_num": buffer[0].get("page_num", 0),
                "section": "",
                "block_types": [b.get("block_type", "text") for b in buffer],
                "is_ocr": any(b.get("is_ocr", False) for b in buffer),
                "strategy": "semantic",
                "source_file": meta.get("filename", ""),
            },
        })
    
    return chunks


def process_parsed_files():
    all_chunks = []
    
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    
    target_files = [f for f in parsed_files if f.stem in TARGET_DOCS]
    
    if not target_files:
        logger.warning("未找到目标解析文件，请先运行 parse_pdf.py")
        return
    
    logger.info(f"找到 {len(target_files)} 个目标解析文件")
    
    for parsed_file in target_files:
        with open(parsed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blocks = data.get("blocks", [])
        meta = data.get("meta", {})
        
        logger.info(f"处理: {parsed_file.name}, {len(blocks)} blocks")
        
        chunks = chunk_semantic(blocks, meta)
        
        all_chunks.extend(chunks)
        logger.info(f"生成: {len(chunks)} chunks")
    
    output_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    avg_length = sum(len(c["content"]) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    logger.info(f"总 chunks: {len(all_chunks)}, 平均长度: {avg_length:.0f} 字符")
    logger.info(f"输出: {output_path.name}")


def main():
    logger.info(f"分块策略: {STRATEGY}")
    process_parsed_files()


if __name__ == "__main__":
    main()