"""
Cross-Encoder 重排器 —— 对粗排结果做精排。

使用 BAAI/bge-reranker-base，输入 (query, chunk_text) 对，
输出相关性分数。支持可选开关。
"""

import os
# 模块级别阻止 HF 在线访问（必须在任何模型加载前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "1"

from src.chunkers.narrative import Chunk
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "BAAI/bge-reranker-base"
CACHE_DIR = "M:/huggingface_cache"


class ReRanker:
    """Cross-Encoder 重排器，可配置开关"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model: CrossEncoder | None = None
        if enabled:
            self.model = CrossEncoder(
                model_name,
                cache_folder=CACHE_DIR,
            )

    def rerank(
        self, query: str, chunks: list[Chunk], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """对 chunk 列表重排。"""
        if not chunks:
            return []

        if not self.enabled or self.model is None:
            return [(c, 0.0) for c in chunks[:top_k]]

        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]
