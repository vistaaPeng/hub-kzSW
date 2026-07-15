import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks_tech"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_tech"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-small-zh-v1.5"

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY    = "semantic"
CHUNKS_FILE = CHUNKS_DIR / f"all_{STRATEGY}.json"


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    if not BGE_MODEL_PATH.exists():
        logger.warning(f"本地模型不存在: {BGE_MODEL_PATH}，将从 HuggingFace 下载")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Embedding 模型加载完成: {model_path}")
    return embeddings


def embed_texts(embeddings, texts: list[str], batch_size: int = 10) -> np.ndarray:
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_idx = i // batch_size + 1

        if batch_idx % 10 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        vecs = embeddings.embed_documents(batch)
        all_embeddings.extend(vecs)

    return np.array(all_embeddings, dtype="float32")


def build_faiss_index(chunks: list[dict], embeddings):
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts      = [c["content"] for c in chunks]
    embeddings_array = embed_texts(embeddings, texts)

    logger.info(f"构建 FAISS 索引，维度={embeddings_array.shape[1]}...")
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "filename": c["metadata"].get("filename", ""),
            "page_num": c["metadata"].get("page_num", -1),
            "block_types": c["metadata"].get("block_types", []),
            "strategy": c["metadata"].get("strategy", ""),
            "source_file": c["metadata"].get("source_file", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_tech_manual.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    embeddings = get_embeddings()
    build_faiss_index(chunks, embeddings)

    logger.info("\n技术手册索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()
