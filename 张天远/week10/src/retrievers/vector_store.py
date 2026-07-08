"""
向量存储 —— 基于 FAISS IndexFlatIP 的语义检索。

使用 BAAI/bge-base-zh-v1.5（768 维）作为 Embedding 模型。
"""

import json
from pathlib import Path
from typing import Any
import os

from src.config import HF_CACHE_DIR, HF_ENDPOINT, HF_OFFLINE

# 在模块级别配置 HF 访问（必须在任何模型加载前设置）
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
if HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunkers.narrative import Chunk

DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
CACHE_DIR = HF_CACHE_DIR


class VectorStore:
    """
    FAISS 向量存储，支持索引构建、查询、持久化。

    使用内积相似度（IndexFlatIP），bge 模型已归一化所以等价于余弦相似度。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            model_name,
            cache_folder=CACHE_DIR,
            device=device,
            local_files_only=HF_OFFLINE,
        )
        self.dim = self.model.get_embedding_dimension()
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        将文本列表编码为向量。

        Args:
            texts: 文本列表

        Returns:
            (N, dim) 的 numpy 数组，已 L2 归一化
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化，使内积 = 余弦相似度
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        构建 FAISS 索引。

        Args:
            chunks: Chunk 列表
        """
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.encode(texts)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        查询 top-K 最相似的 chunk。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(chunk, score), ...] 按相似度降序排列
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        query_vec = self.encode([query])
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, index_path: str, chunks_path: str) -> None:
        """
        持久化索引和 chunk 数据。

        Args:
            index_path: FAISS 索引文件路径
            chunks_path: chunk 元数据 JSON 路径
        """
        if self.index is None:
            raise RuntimeError("索引未构建，无法保存")

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        # 保存 chunk 元数据
        chunks_data = []
        for c in self.chunks:
            chunks_data.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
                "parent_chunk_id": c.parent_chunk_id,
                "is_parent": c.is_parent,
            })
        Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chunks_path).write_text(
            json.dumps(chunks_data, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, index_path: str, chunks_path: str) -> None:
        """
        加载持久化的索引和 chunk 数据。

        Args:
            index_path: FAISS 索引文件路径
            chunks_path: chunk 元数据 JSON 路径
        """
        self.index = faiss.read_index(index_path)

        data = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunks = []
        for d in data:
            self.chunks.append(Chunk(
                chunk_id=d["chunk_id"],
                text=d["text"],
                metadata=d["metadata"],
                parent_chunk_id=d.get("parent_chunk_id"),
                is_parent=d.get("is_parent", False),
            ))
