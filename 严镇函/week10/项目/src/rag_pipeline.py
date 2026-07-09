"""
RAG 问答流水线（刑法版）

流程：
  向量检索（DashScope embedding + FAISS）
        +
  BM25 关键词检索（jieba + rank_bm25）
        ↓
  RRF 融合排名
        ↓
  LLM 生成（DashScope qwen-plus）+ 引用标注

使用方式：
  python rag_pipeline.py                            # 交互式问答
  python rag_pipeline.py --query "故意杀人罪判几年"

依赖：
  pip install faiss-cpu rank_bm25 jieba openai numpy
  set DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 路径配置（根据你的实际路径调整）
BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH      = Path(r"D:\PythonStudy\yzh_study\Embedding\week10\vectorstore\faiss_index.bin")
META_PATH       = Path(r"D:\PythonStudy\yzh_study\Embedding\week10\vectorstore\faiss_meta.json")

# API 配置
DASHSCOPE_URL = "https://ws-4hvtvp46nkb7dftd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL     = "text-embedding-v3"    # 向量模型
EMBED_DIM       = 1024                   # 向量维度
LLM_MODEL       = "qwen-plus"            # 问答模型，可换 qwen-turbo（更快）/ qwen-max（更强）

# 检索参数
TOP_K_RETRIEVE  = 10   # 初始召回条数
TOP_K_RERANK    = 4    # 最终给 LLM 的条数
SCORE_THRESHOLD = 0.2  # 相关性阈值，低于此值拒绝回答

# 系统提示词：告诉 AI 怎么回答刑法问题
SYSTEM_PROMPT = """你是一个专业的中国刑法助手，专门回答关于《中华人民共和国刑法》的问题。

回答规则：
1. 只根据【参考资料】中的法条内容回答，不得编造法条外的内容
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体法条时，在句末标注来源编号，如：根据刑法第二百三十二条[1]
4. 法条引用要精确到条号，不得模糊表达
5. 回答简洁，重点突出，避免无关废话"""


# ── DashScope 客户端 ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        # 如果环境变量没有，直接在这里填（仅供测试）
        api_key = "sk-ws-H.EMERMYL.YoR7.MEYCIQDk2OkODYwBYANDFmDpOuM38y4OdLKOk3v1ak7STlm7-AIhAMDt20IxGX_pjFAT-u3Y3qzSt1ox3FI2pqYOqGqkr662"
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)
# ── 向量检索 ──────────────────────────────────────────────────────────────────

class VectorStore:
    """向量检索类：用 FAISS 索引找最相关的法条"""

    def __init__(self, client: OpenAI):
        import faiss
        self.client = client

        # 加载 FAISS 索引
        logger.info(f"加载 FAISS 索引: {INDEX_PATH}")
        self.index = faiss.read_index(str(INDEX_PATH))

        # 加载元数据
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def _embed_query(self, query: str) -> np.ndarray:
        """把用户的问题转成向量"""
        resp = self.client.embeddings.create(
            model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        # L2 归一化
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        """
        向量检索：找与问题最相关的 top_k 条法条

        返回：按相似度排序的列表，每条包含 content、article_num、score 等
        """
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, top_k * 2)  # 多取一些备选

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)  # 余弦相似度，0~1
            results.append(item)
            if len(results) >= top_k:
                break

        return results
# ── BM25 关键词检索 ───────────────────────────────────────────────────────────

class BM25Store:
    """
    基于 jieba + rank_bm25 的关键词检索

    为什么需要 BM25？
      向量检索擅长"找意思相近的"，但有时用户问"死刑"，
      法条里写的是"判处死刑"，向量可能匹配到"死缓"——意思相近但法条不同。
      BM25 是精确关键词匹配，刚好互补。
    """

    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        # 加载元数据（所有法条文本）
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        # 对每条法条做中文分词
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info("BM25 索引完成")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        """BM25 检索：找包含查询关键词的法条"""
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:  # 完全不相关 → 跳过
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results
# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
        vec_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion（倒数排名融合）

    把向量检索和 BM25 检索的结果合并排序。
    公式：score(d) = Σ 1/(k + rank_i(d))

    比如一条法条在向量检索排第2，在 BM25 排第5：
      score = 1/(60+2) + 1/(60+5) = 0.016 + 0.015 = 0.031
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict] = {}

    # 向量检索的排名
    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    # BM25 的排名
    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    # 按 RRF 分数从高到低排序
    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


# ── LLM 生成 ──────────────────────────────────────────────────────────────────

def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """
    把检索到的法条组装成 Prompt 上下文

    返回：
      context: 格式化的上下文字符串
      citations: 引用列表（供溯源）
    """
    parts = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        article_num = item.get("article_num", "")
        section_path = item.get("section_path", "")

        # 标签：如 "[1] 第二条 · 第一章"
        label = f"[{i}]"
        if article_num:
            label += f" {article_num}"
        if section_path:
            # 只保留最后两级
            parts_path = section_path.split(" > ")
            if len(parts_path) > 1:
                label += f" · {parts_path[-2]}" if len(parts_path) >= 2 else ""

        content = item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({
            "index": i,
            "source": label,
            "chunk_id": item.get("chunk_id", ""),
            "article_num": article_num,
        })

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client: OpenAI) -> str:
    """调用大模型生成回答"""
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用法条处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,  # 低温度 = 更精确，减少编造
    )
    return resp.choices[0].message.content


# ── 完整流水线 ────────────────────────────────────────────────────────────────

class RAGPipeline:
    """RAG 问答流水线：把上面的所有组件串起来"""

    def __init__(self, use_bm25: bool = True):
        self.client = get_client()
        self.vec_store = VectorStore(self.client)
        self.use_bm25 = use_bm25
        self.bm25_store = BM25Store() if use_bm25 else None
        logger.info("RAG Pipeline 初始化完成！")

    def query(self, question: str) -> dict:
        """
        执行完整的 RAG 问答流程

        参数 question: 用户的问题
        返回: {"answer": 答案, "citations": 引用列表, "retrieved": 检索到的法条}
        """
        # ① 向量检索
        vec_results = self.vec_store.search(question, TOP_K_RETRIEVE)
        logger.info(f"向量召回: {len(vec_results)} 条")

        # ② BM25 + RRF 融合
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(question, TOP_K_RETRIEVE)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
            logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 融合后: {len(candidates)} 条")
        else:
            candidates = vec_results

        # ③ 取 Top-K
        final = candidates[:TOP_K_RERANK]

        # ④ 相关性阈值检查
        if not final:
            return {
                "answer": "未找到相关法条，无法回答此问题。",
                "citations": [],
                "retrieved": [],
            }

        # ⑤ LLM 生成
        context, citations = build_context(final)
        answer = call_llm(question, context, self.client)

        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="刑法 RAG 问答系统")
    parser.add_argument("--query", type=str, default=None, help="直接提问，如'故意杀人罪判几年'")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25（消融实验用）")
    args = parser.parse_args()

    # 初始化流水线
    pipeline = RAGPipeline(use_bm25=not args.no_bm25)

    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"📝 问题：{q}")
        print(f"{'='*60}")
        print(f"\n💡 答案：\n{result['answer']}")
        if result["citations"]:
            print("\n📖 引用法条：")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        # 单次问答模式
        result = pipeline.query(args.query)
        print_result(args.query, result)
    else:
        # 交互式问答模式
        print("\n" + "=" * 50)
        print("📚 刑法 RAG 问答系统")
        print("=" * 50)
        print("输入问题开始查询，输入 'exit' 退出\n")

        while True:
            try:
                q = input("❓ 问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break

            result = pipeline.query(q)
            print_result(q, result)


if __name__ == "__main__":
    main()