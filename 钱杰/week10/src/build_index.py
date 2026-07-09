import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTORSTORE_DIR / "dense_index.npy"
META_PATH = VECTORSTORE_DIR / "dense_meta.json"

MODEL_NAME = str(BASE_DIR / "model_cache" / "bert-base-chinese")


def load_chunks() -> List[Dict[str, Any]]:
    chunks_file = CHUNKS_DIR / "all_semantic.json"
    
    if not chunks_file.exists():
        chunks_file = CHUNKS_DIR / "all_fixed.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"未找到分块文件，请先运行 chunk_documents.py")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"加载 {len(chunks)} 个 chunks")
    return chunks


def encode_texts(texts: List[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    logger.info(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    embeddings = []
    batch_size = 8
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def build_index():
    chunks = load_chunks()
    
    if not chunks:
        logger.error("没有 chunks 需要处理")
        return
    
    texts = [chunk["content"] for chunk in chunks]
    
    logger.info("开始构建密集向量索引...")
    
    embeddings = encode_texts(texts)
    
    logger.info(f"向量编码完成，{embeddings.shape[0]} × {embeddings.shape[1]}")
    
    np.save(str(INDEX_PATH), embeddings)
    
    logger.info(f"向量索引已保存: {INDEX_PATH.name}")
    
    meta_list = []
    for chunk in chunks:
        meta = {
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "doc_type": chunk["metadata"].get("doc_type", ""),
            "title": chunk["metadata"].get("title", ""),
            "page_num": chunk["metadata"].get("page_num", 0),
            "section": chunk["metadata"].get("section", ""),
            "is_ocr": chunk["metadata"].get("is_ocr", False),
            "source_file": chunk["metadata"].get("source_file", ""),
        }
        meta_list.append(meta)
    
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    
    meta_size = os.path.getsize(META_PATH) / (1024 * 1024)
    logger.info(f"元数据已保存: {META_PATH.name} ({meta_size:.1f} MB)")


def main():
    build_index()


if __name__ == "__main__":
    main()