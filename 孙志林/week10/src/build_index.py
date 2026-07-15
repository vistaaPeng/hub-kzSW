"""
向量索引构建模块

Embedding方案：本地 BGE-small-zh-v1.5（开源中文向量化模型）
向量库：FAISS（IndexFlatIP，内积=归一化后的余弦相似度）

使用方式：
  python build_index.py
  python build_index.py --strategy semantic
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"

STRATEGY = "semantic"
CHUNKS_FILE = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
BATCH_SIZE = 32


def get_embeddings():
    """加载本地或下载BGE embedding模型"""
    from sentence_transformers import SentenceTransformer

    local_path = MODELS_DIR / "bge-small-zh-v1.5"
    model_path = str(local_path) if local_path.exists() else EMBED_MODEL

    logger.info(f"Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path, cache_folder=str(MODELS_DIR))
    return model


def embed_texts(model, texts: list[str], show_progress: bool = True) -> np.ndarray:
    """批量计算embedding，返回shape=(N, EMBED_DIM)的float32数组"""
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 50 == 0:
            logger.info(f"  Embedding progress: {batch_idx}/{total_batches} batches")

        for attempt in range(3):
            try:
                vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  Attempt {attempt+1} failed, retrying: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms

    return embeddings


def build_faiss_index(embeddings: np.ndarray):
    """构建FAISS索引（IndexFlatIP）"""
    import faiss

    embed_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors, dimension: {embed_dim}")
    return index


def main(strategy: str = STRATEGY):
    chunks_file = CHUNKS_DIR / f"all_{strategy}.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}\nPlease run chunk_documents.py first")

    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["content"] for chunk in chunks]
    logger.info(f"Total chunks: {len(chunks)}, total text length: {sum(len(t) for t in texts):,}")

    logger.info("Starting embedding...")
    model = get_embeddings()
    embeddings = embed_texts(model, texts)
    logger.info(f"Embedding completed, shape: {embeddings.shape}")

    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")

    meta_path = VECTORSTORE_DIR / "faiss_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build vector index")
    parser.add_argument("--strategy", type=str, default=STRATEGY, choices=["fixed", "semantic", "hierarchical"])
    args = parser.parse_args()
    main(strategy=args.strategy)