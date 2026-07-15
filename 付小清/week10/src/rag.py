"""
RAG 核心流水线

流程：向量检索 + BM25 关键词检索 → RRF 融合 → 相关性过滤 → qwen-plus 生成
"""

import json
import logging
import os

import faiss
import numpy as np
from openai import OpenAI

from config import (
    INDEX_PATH, META_PATH,
    DASHSCOPE_URL, EMBED_MODEL, EMBED_DIM, LLM_MODEL,
    TOP_K, TOP_K_RETRIEVE, SCORE_THRESHOLD, RRF_K, SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class BM25Store:
    """基于 jieba + rank_bm25 的关键词检索，擅长精确匹配医学术语和数值。"""

    def __init__(self, meta_list: list[dict]):
        from rank_bm25 import BM25Okapi
        import jieba

        self.meta_list = meta_list
        self.jieba = jieba
        logger.info("构建 BM25 索引（分词中）...")
        tokenized = [list(jieba.cut(item["content"])) for item in meta_list]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 索引完成")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


def reciprocal_rank_fusion(
    vec_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """RRF 融合：score(d) = Σ 1/(k + rank_i(d))"""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        item["score"] = rrf_scores[cid]
        results.append(item)
    return results


class RAGPipeline:
    def __init__(self, use_bm25: bool = True):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"向量索引不存在: {INDEX_PATH}\n请先运行: python src/build_index.py"
            )

        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        self.use_bm25 = use_bm25
        self._client: OpenAI | None = None
        self.bm25_store = BM25Store(self.meta_list) if use_bm25 else None
        logger.info(f"加载 FAISS 索引: {self.index.ntotal} 条向量")

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "请设置环境变量 DASHSCOPE_API_KEY\n"
                "获取地址: https://dashscope.console.aliyun.com/"
            )
        self._client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)
        return self._client

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self._get_client().embeddings.create(
            model=EMBED_MODEL,
            input=[query],
            dimensions=EMBED_DIM,
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)
            item["score"] = float(score)
            results.append(item)
        return results

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        vec_results = self._vector_search(query, TOP_K_RETRIEVE)

        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(query, TOP_K_RETRIEVE)
            merged = reciprocal_rank_fusion(vec_results, bm25_results)
            return merged[:top_k]

        return vec_results[:top_k]

    def _build_context(self, retrieved: list[dict]) -> tuple[str, list[dict]]:
        parts, citations = [], []
        for i, item in enumerate(retrieved, 1):
            source = item.get("source", "")
            section = item.get("section", "")
            label = f"[{i}] {source}"
            if section:
                label += f" · {section}"
            parts.append(f"{label}\n{item['content']}")
            citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})
        return "\n\n---\n\n".join(parts), citations

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        vec_results = self._vector_search(question, TOP_K_RETRIEVE)

        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(question, TOP_K_RETRIEVE)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
        else:
            candidates = vec_results

        retrieved = candidates[:top_k]

        if not retrieved:
            return {"answer": "未检索到相关内容。", "citations": [], "retrieved": []}

        top_vec_score = max((c.get("vec_score", 0) for c in retrieved), default=0)
        if top_vec_score < SCORE_THRESHOLD:
            return {
                "answer": (
                    f"检索相关度较低（{top_vec_score:.2f}），知识库中可能没有相关信息。"
                    "如有不适请及时就医。"
                ),
                "citations": [],
                "retrieved": retrieved,
            }

        context, citations = self._build_context(retrieved)
        answer = self._call_llm(question, context)
        disclaimer = "\n\n【免责声明】以上回答仅供参考，不能替代医生诊断和治疗。如有不适请及时就医。"
        return {"answer": answer + disclaimer, "citations": citations, "retrieved": retrieved}

    def _call_llm(self, question: str, context: str) -> str:
        user_msg = (
            f"【参考资料】\n{context}\n\n"
            f"【问题】\n{question}\n\n"
            "请根据参考资料回答，引用处标注来源编号（如[1]）。"
        )
        resp = self._get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content
