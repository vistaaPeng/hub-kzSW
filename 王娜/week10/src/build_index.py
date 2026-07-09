"""
向量索引构建脚本（原生实现 — 本地模型版）

Embedding 方案：BAAI/bge-small-en-v1.5（本地模型）
  - 维度：512
  - 纯 CPU 运行，无需 GPU
  - 无需 API Key，无需网络

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

依赖：
  pip install faiss-cpu sentence-transformers numpy
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"          # 与 chunk_documents.py 保持一致
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

MODEL_DIR       = BASE_DIR / "models" / "bge-small-en-v1.5"
EMBED_DIM       = None                # 从加载的模型动态获取
BATCH_SIZE      = 32                  # CPU 上合理的 batch size


# ── 本地 Embedding 模型 ──────────────────────────────────────────────────────

def load_embedding_model():
    """加载本地 sentence-transformer 模型。"""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"找不到本地模型: {MODEL_DIR}\n"
            "请先下载模型或使用 API 版本:\n"
            "  1. 运行 src_langchain/download_model.py 下载\n"
            "  2. 或手动下载到 models/bge-small-en-v1.5/"
        )
    logger.info(f"加载本地模型: {MODEL_DIR}")
    model = SentenceTransformer(str(MODEL_DIR))
    global EMBED_DIM
    EMBED_DIM = model.get_embedding_dimension()
    logger.info(f"模型加载成功, 维度={EMBED_DIM}")
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    批量计算 embedding，返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。
    """
    logger.info(f"计算 {len(texts)} 条 embedding...")
    batch_size = min(BATCH_SIZE, len(texts))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # 自动 L2 归一化
    )

    return np.array(embeddings, dtype="float32")


# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], model: SentenceTransformer):
    """
    构建 FAISS 向量索引。

    FAISS 说明：
      IndexFlatIP = 暴力内积检索，精确但不近似。
      数据量 < 10 万时速度完全够用，是教学的首选。
      数据量更大时可换 IndexIVFFlat（需要 train）或 IndexHNSW。
    """
    import faiss

    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(model, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 持久化：索引文件 + 元数据（分开存，避免把大向量序列化进 JSON）
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    # FAISS 对中文路径支持不佳，先写入临时目录再拷贝
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = f.name
    faiss.write_index(index, tmp_path)
    shutil.move(tmp_path, index_path)
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":   c["chunk_id"],
            "content":    c["content"],
            "stock_code": c["metadata"].get("stock_code", ""),
            "year":       c["metadata"].get("year", ""),
            "page_num":   c["metadata"].get("page_num", -1),
            "section":    c["metadata"].get("section", ""),
            "block_types":c["metadata"].get("block_types", []),
            "is_ocr":     c["metadata"].get("is_ocr", False),
            "strategy":   c["metadata"].get("strategy", ""),
            "source_file":c["metadata"].get("source_file", ""),
            # 层级分块时保留父块内容供 LLM 读取
            "parent_content": c["metadata"].get("parent_content", ""),
            "parent_id":      c["metadata"].get("parent_id", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


# ── ChromaDB 索引构建（可选对比） ──────────────────────────────────────────────

def build_chroma_index(chunks: list[dict], model: SentenceTransformer):
    """
    ChromaDB 版本（可选）。
    优势：内置元数据过滤，可直接 where={"stock_code": "600519"} 过滤。
    劣势：写入较慢，不适合大批量。
    """
    try:
        import chromadb
    except ImportError:
        logger.error("请先安装 chromadb: pip install chromadb")
        return

    chroma_dir = VECTORSTORE_DIR / "chroma"
    client_db  = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client_db.get_or_create_collection(
        name="annual_reports",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"向 ChromaDB 写入 {len(chunks)} 条 chunk...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(model, texts)

    for i in range(0, len(chunks), 100):
        batch   = chunks[i:i+100]
        ids     = [c["chunk_id"] for c in batch]
        docs    = [c["content"] for c in batch]
        embs    = embeddings[i:i+100].tolist()
        metas   = []
        for c in batch:
            m = dict(c["metadata"])
            # ChromaDB 只支持 str/int/float/bool 类型的 metadata value
            m["block_types"] = ",".join(m.get("block_types") or [])
            m.pop("parent_content", None)   # 太长
            metas.append(m)
        collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        logger.info(f"  已写入 {min(i+100, len(chunks))}/{len(chunks)}")

    logger.info(f"ChromaDB 写入完成，共 {collection.count()} 条")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    model = load_embedding_model()

    # 默认用 FAISS
    build_faiss_index(chunks, model)

    # 若要对比 ChromaDB，取消下行注释：
    # build_chroma_index(chunks, model)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()