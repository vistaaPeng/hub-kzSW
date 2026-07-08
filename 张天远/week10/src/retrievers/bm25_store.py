"""
BM25 检索器 —— 基于 rank-bm25 + jieba 分词的关键词检索。

与 VectorStore 接口一致（build_index / search 返回 [(Chunk, float)]），
便于后续 RRF 融合。
"""

import jieba
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

from src.chunkers.narrative import Chunk

RUST_TERMS = {
    "Rust", "Cargo", "crate", "Crate", "trait", "Trait", "impl", "enum", "struct",
    "async", "await", "match", "unsafe", "where", "Self", "self", "super",
    "所有权", "借用", "生命周期", "智能指针", "函数指针", "闭包", "泛型",
    "关联类型", "父 trait", "子 trait", "模式匹配", "错误处理", "可变引用",
    "不可变引用", "迭代器", "宏", "模块", "关键字", "保留字",
    "associated type", "function pointer", "supertrait", "sub-trait",
    "smart pointer", "lifetime", "ownership", "borrowing", "pattern matching",
}
_JIEBA_TERMS_LOADED = False
_BM25_TERMS: set[str] = set()


def load_jieba_terms() -> None:
    """Inject Rust and glossary terms into jieba once."""
    global _JIEBA_TERMS_LOADED
    global _BM25_TERMS
    if _JIEBA_TERMS_LOADED:
        return

    terms = set(RUST_TERMS)
    try:
        from src.glossary import get_glossary
        glossary = get_glossary()
        for english, chinese in glossary.items():
            if english:
                terms.add(english)
            if chinese:
                terms.add(chinese)
    except Exception:
        pass

    for term in terms:
        clean = term.strip()
        if clean:
            jieba.add_word(clean, freq=200000)

    _BM25_TERMS = terms
    _JIEBA_TERMS_LOADED = True


class BM25Store:
    """
    BM25 关键词检索器。

    使用 jieba 对中文文本分词，rank-bm25 的 BM25Okapi 实现。
    接口与 VectorStore 对齐，search 返回 [(Chunk, score), ...]。
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []   # 分词后的文档列表
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """用 jieba 对文本分词（精确模式）。"""
        load_jieba_terms()
        tokens = [t for t in jieba.cut(text) if t.strip()]
        lowered = text.lower()
        phrase_tokens = [
            term for term in _BM25_TERMS
            if " " in term and term.lower() in lowered
        ]
        return tokens + phrase_tokens

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        构建 BM25 索引。

        Args:
            chunks: Chunk 列表
        """
        self.chunks = list(chunks)
        self._corpus = [self._tokenize(c.text) for c in chunks]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        BM25 关键词检索。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(chunk, score), ...] 按 BM25 分数降序排列。
            空查询或无索引时返回空列表。
        """
        if self._bm25 is None:
            return []

        query = query.strip()
        if not query:
            return []

        tokenized = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized)

        # 按分数降序取 top_k（不限制最低分，与 VectorStore 行为一致）
        top_k = min(top_k, len(scores))
        if top_k <= 0:
            return []

        # argsort 降序
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save(self, path: str) -> None:
        """持久化 BM25 索引为 pickle 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata,
                 "parent_chunk_id": c.parent_chunk_id, "is_parent": c.is_parent}
                for c in self.chunks
            ],
            "corpus": self._corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """从 pickle 文件加载 BM25 索引。"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.chunks = [
            Chunk(d["chunk_id"], d["text"], d["metadata"],
                  parent_chunk_id=d.get("parent_chunk_id"),
                  is_parent=d.get("is_parent", False))
            for d in data["chunks"]
        ]
        self._corpus = data["corpus"]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
