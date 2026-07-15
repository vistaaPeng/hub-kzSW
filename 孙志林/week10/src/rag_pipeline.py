"""
RAG问答流水线（核心模块）

完整流程：
  查询改写（可选）
    ↓
  向量检索（FAISS）
    +
  BM25关键词检索（jieba + rank_bm25）
    ↓
  RRF融合排名
    ↓
  CrossEncoder重排（可选）
    ↓
  相关性阈值过滤
    ↓
  LLM生成 + 引用标注

使用方式：
  python rag_pipeline.py
  python rag_pipeline.py --query "贵州茅台2023年营收"
  python rag_pipeline.py --query "..." --stock 600519 --year 2023
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH = VECTORSTORE_DIR / "faiss_meta.json"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512
LLM_MODEL = "qwen-plus"
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TOP_K_RETRIEVE = 10
TOP_K_RERANK = 4
SCORE_THRESHOLD = 0.25

SYSTEM_PROMPT = """你是一个专业的财务分析助手，专门回答关于中国上市公司年度报告的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：营业收入为1476亿元[1]
4. 数字要精确，不得四舍五入或模糊表达
5. 回答简洁，重点突出，避免无关废话"""


def get_client():
    """获取DashScope客户端（兼容OpenAI接口）"""
    try:
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
        return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)
    except ImportError:
        logger.warning("openai库未安装，LLM生成功能不可用")
        return None


class VectorStore:
    """向量检索模块"""

    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        model_path = BASE_DIR / "models" / "bge-small-zh-v1.5"
        model_name = str(model_path) if model_path.exists() else EMBED_MODEL
        self.embed_model = SentenceTransformer(model_name, cache_folder=str(BASE_DIR / "models"))

        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode(query, normalize_embeddings=True)
        return np.array([vec], dtype="float32")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE, filter_meta: Optional[dict] = None) -> list[dict]:
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            if filter_meta:
                if not all(str(item.get(k, "")) == str(v) for k, v in filter_meta.items()):
                    continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results


class BM25Store:
    """BM25关键词检索模块"""

    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("Building BM25 index...")
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info("BM25 index built")

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


def reciprocal_rank_fusion(vec_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """RRF融合：合并向量检索和BM25的排名结果

    公式：score(d) = Σ 1/(k + rank_i(d))
    k=60为经验值，平衡不同检索方式的贡献
    """
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
        results.append(item)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """CrossEncoder重排：对候选集二次精排

    使用BAAI/bge-reranker-base模型，双向注意力比bi-encoder更准确
    """
    try:
        from sentence_transformers import CrossEncoder

        model_path = BASE_DIR / "models" / "bge-reranker-base"
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-base"
        reranker = CrossEncoder(model_name)

        pairs = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda x: -x.get("rerank_score", 0))
    except ImportError:
        logger.warning("sentence-transformers未安装，跳过Rerank")
    except Exception as e:
        logger.warning(f"Rerank failed: {e}")

    return candidates[:top_k]


def rewrite_query(query: str, client) -> str:
    """查询改写：将模糊问题转换为更适合检索的精确表述

    示例：
      原始：茅台最近怎么样
      改写：贵州茅台2023年营业收入净利润同比增长率经营情况
    """
    if not client:
        return query

    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{
            "role": "system",
            "content": "你是检索查询优化专家。将用户的问题改写为更适合从年度报告中检索信息的精确查询语句。保留关键实体（公司名、年份、财务指标），扩展相关关键词，不要超过50字。直接输出改写后的查询语句，不要解释。"
        }, {"role": "user", "content": query}],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.info(f"Query rewritten: {query!r} -> {rewritten!r}")
    return rewritten


def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """将检索结果组装为Prompt上下文"""
    parts = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        stock = item.get("stock_code", "")
        year = item.get("year", "")
        page = item.get("page_num", "")
        section = item.get("section", "")

        label = f"[{i}] {stock} {year}年报"
        if section:
            label += f" · {section}"
        if page and page != -1:
            label += f" · 第{page}页"

        content = item.get("parent_content") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client) -> str:
    """调用LLM生成回答"""
    if not client:
        return "LLM客户端未初始化，请设置DASHSCOPE_API_KEY环境变量"

    user_msg = f"【参考资料】\n{context}\n\n【问题】\n{query}\n\n请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def rag_pipeline(
    query: str,
    stock_code: str = None,
    year: str = None,
    use_bm25: bool = True,
    use_rerank: bool = True,
    use_query_rewrite: bool = False,
) -> dict:
    """完整的RAG问答流水线"""
    client = get_client()
    vector_store = VectorStore()

    if use_query_rewrite:
        query = rewrite_query(query, client)

    filter_meta = {}
    if stock_code:
        filter_meta["stock_code"] = stock_code
    if year:
        filter_meta["year"] = year

    vec_results = vector_store.search(query, top_k=TOP_K_RETRIEVE, filter_meta=filter_meta)
    logger.info(f"Vector search returned {len(vec_results)} results")

    bm25_results = []
    if use_bm25:
        bm25_store = BM25Store()
        bm25_results = bm25_store.search(query, top_k=TOP_K_RETRIEVE)
        logger.info(f"BM25 search returned {len(bm25_results)} results")

    if use_bm25 and bm25_results:
        fused_results = reciprocal_rank_fusion(vec_results, bm25_results)
    else:
        fused_results = vec_results

    if use_rerank:
        reranked_results = rerank(query, fused_results[:TOP_K_RERANK * 2], top_k=TOP_K_RERANK)
    else:
        reranked_results = fused_results[:TOP_K_RERANK]

    if not reranked_results:
        return {"answer": "根据提供的资料无法回答此问题", "sources": [], "retrieved": []}

    max_score = reranked_results[0].get("rerank_score", reranked_results[0].get("rrf_score", 0))
    if max_score < SCORE_THRESHOLD:
        return {"answer": "根据提供的资料无法回答此问题", "sources": [], "retrieved": reranked_results}

    context, citations = build_context(reranked_results)
    answer = call_llm(query, context, client)

    return {
        "answer": answer,
        "sources": citations,
        "retrieved": reranked_results,
        "context": context,
    }


def interactive_mode():
    """交互式问答模式"""
    print("=" * 60)
    print("RAG 上市公司年报问答系统")
    print("=" * 60)
    print("输入问题进行查询，输入 'exit' 或 'quit' 退出")
    print("示例：贵州茅台2023年营业收入是多少？")
    print("=" * 60)

    while True:
        query = input("\n请输入问题：").strip()
        if query.lower() in ["exit", "quit", "退出"]:
            break

        if not query:
            continue

        try:
            result = rag_pipeline(query)
            print("\n【回答】")
            print(result["answer"])

            if result["sources"]:
                print("\n【来源】")
                for src in result["sources"]:
                    print(f"  {src['source']}")
        except Exception as e:
            logger.error(f"Error: {e}")
            print("查询失败，请稍后重试")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG问答系统")
    parser.add_argument("--query", type=str, help="查询问题")
    parser.add_argument("--stock", type=str, help="股票代码过滤（如600519）")
    parser.add_argument("--year", type=str, help="年份过滤（如2023）")
    parser.add_argument("--no-bm25", action="store_true", help="禁用BM25检索")
    parser.add_argument("--no-rerank", action="store_true", help="禁用CrossEncoder重排")
    parser.add_argument("--query-rewrite", action="store_true", help="启用查询改写")
    args = parser.parse_args()

    if args.query:
        result = rag_pipeline(
            query=args.query,
            stock_code=args.stock,
            year=args.year,
            use_bm25=not args.no_bm25,
            use_rerank=not args.no_rerank,
            use_query_rewrite=args.query_rewrite,
        )
        print("【回答】")
        print(result["answer"])
        if result["sources"]:
            print("\n【来源】")
            for src in result["sources"]:
                print(f"  {src['source']}")
    else:
        interactive_mode()