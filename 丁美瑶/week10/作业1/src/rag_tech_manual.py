import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_tech"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-small-zh-v1.5"

INDEX_PATH      = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH       = VECTORSTORE_DIR / "faiss_meta.json"

EMBED_DIM       = 384
TOP_K_RETRIEVE  = 5
TOP_K_FINAL     = 3
SCORE_THRESHOLD = 0.2

SYSTEM_PROMPT = """你是一个专业的技术文档助手，专门回答关于芯片和电子元器件技术手册的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：工作电压为5V[1]
4. 回答简洁，重点突出，避免无关废话"""


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    def __init__(self, embeddings):
        import faiss
        self.embeddings = embeddings
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        query_vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["score"] = float(score)
            results.append(item)
        return results


def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    parts     = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        filename = item.get("filename", "")
        page     = item.get("page_num", "")

        label = f"[{i}] {filename}"
        if page and page != -1:
            label += f" · 第{page}页"

        content = item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str):
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY\n  Windows: set DASHSCOPE_API_KEY=sk-xxx")

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model="qwen3.7-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


class TechManualQA:
    def __init__(self, use_llm: bool = True):
        self.embeddings = get_embeddings()
        self.vec_store  = VectorStore(self.embeddings)
        self.use_llm    = use_llm and os.getenv("DASHSCOPE_API_KEY") is not None

        if not self.use_llm:
            logger.warning("未配置 DASHSCOPE_API_KEY，将使用基于检索的直接回答模式")

    def query(self, question: str, verbose: bool = False) -> dict:
        results = self.vec_store.search(question, TOP_K_RETRIEVE)

        if verbose:
            logger.info(f"向量召回: {len(results)} 条，最高分={results[0]['score']:.3f}" if results else "向量召回: 0 条")

        if not results:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [], "retrieved": [],
            }

        top_score = results[0].get("score", 0.0)
        if top_score < SCORE_THRESHOLD:
            return {
                "answer": f"检索到的内容相关性较低（最高得分: {top_score:.3f}），以下是相关片段供参考：",
                "citations": [], "retrieved": results[:TOP_K_FINAL],
            }

        final = results[:TOP_K_FINAL]
        context, citations = build_context(final)

        if self.use_llm:
            answer = call_llm(question, context)
            return {"answer": answer, "citations": citations, "retrieved": final}
        else:
            return {"answer": context, "citations": citations, "retrieved": final}


def main():
    parser = argparse.ArgumentParser(description="技术手册 RAG 问答系统")
    parser.add_argument("--query", type=str, default=None, help="要查询的问题")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM，只输出检索结果")
    args = parser.parse_args()

    qa = TechManualQA(use_llm=not args.no_llm)

    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        result = qa.query(args.query, verbose=True)
        print_result(args.query, result)
    else:
        print("技术手册 RAG 问答系统")
        print(f"向量库：{INDEX_PATH}")
        print(f"模式：{'LLM生成' if qa.use_llm else '检索直接输出'}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            result = qa.query(q, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()
