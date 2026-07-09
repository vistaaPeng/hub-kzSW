"""
知识库索引构建

流程：加载 Markdown → 分块 → DashScope Embedding → FAISS 索引

用法：
  python src/build_index.py
  需要环境变量 DASHSCOPE_API_KEY
"""

import json
import logging
import os
import re
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from config import (
    BASE_DIR, DOCS_DIR, VECTORSTORE_DIR,
    INDEX_PATH, META_PATH,
    DASHSCOPE_URL, EMBED_MODEL, EMBED_DIM, EMBED_BATCH_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "获取地址: https://dashscope.console.aliyun.com/"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """批量调用 DashScope text-embedding-v3"""
    all_embeddings = []
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        logger.info(f"Embedding 进度: {batch_num}/{total_batches} 批")

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIM,
        )
        for item in sorted(resp.data, key=lambda x: x.index):
            all_embeddings.append(item.embedding)

    vectors = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def load_documents(docs_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "content": text})
        logger.info(f"加载文档: {path.name} ({len(text)} 字符)")
    return docs


def split_by_sections(text: str, source: str) -> list[dict]:
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks, chunk_idx = [], 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        title_match = re.match(r"^(#{1,3})\s+(.+)", section)
        title = title_match.group(2).strip() if title_match else "正文"

        if len(section) <= CHUNK_SIZE * 2:
            chunks.append({
                "chunk_id": f"{source}_{chunk_idx:03d}",
                "content": section,
                "metadata": {"source": source, "section": title, "chunk_index": chunk_idx},
            })
            chunk_idx += 1
        else:
            paragraphs = section.split("\n\n")
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) > CHUNK_SIZE and buffer:
                    chunks.append({
                        "chunk_id": f"{source}_{chunk_idx:03d}",
                        "content": buffer.strip(),
                        "metadata": {"source": source, "section": title, "chunk_index": chunk_idx},
                    })
                    chunk_idx += 1
                    buffer = (para[-CHUNK_OVERLAP:] + "\n\n" + para) if CHUNK_OVERLAP else para
                else:
                    buffer = buffer + "\n\n" + para if buffer else para
            if buffer.strip():
                chunks.append({
                    "chunk_id": f"{source}_{chunk_idx:03d}",
                    "content": buffer.strip(),
                    "metadata": {"source": source, "section": title, "chunk_index": chunk_idx},
                })
                chunk_idx += 1

    return chunks


def build_index():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()

    docs = load_documents(DOCS_DIR)
    if not docs:
        raise FileNotFoundError(f"未找到文档，请将 Markdown 放入 {DOCS_DIR}")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(split_by_sections(doc["content"], doc["source"]))
    logger.info(f"共生成 {len(all_chunks)} 个文本块")

    chunks_path = BASE_DIR / "data" / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    texts = [c["content"] for c in all_chunks]
    embeddings = embed_texts(client, texts)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS 索引已保存 → {INDEX_PATH} ({index.ntotal} 条)")

    meta_list = [
        {"chunk_id": c["chunk_id"], "content": c["content"], **c["metadata"]}
        for c in all_chunks
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    logger.info("索引构建完成！")


if __name__ == "__main__":
    build_index()
