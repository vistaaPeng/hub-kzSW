#!/usr/bin/env python
"""
FastAPI 封装 —— 将 RAG 查询暴露为 REST API
---
用法: python scripts/app.py [--port 8000]
端点:
  POST /query   {"question": "Rust 有哪些关键字？", "rerank": false}
  GET  /health  {"status": "ok"}
  GET  /stats   评估统计（chunk 数、索引信息、评估结果）
  GET  /history 查询日志最近 100 条
  POST /parse   增量数据解析（HTML → 结构化 JSON）
"""

import sys
import time
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.generator import RAGGenerator
from src.reranker.reranker import ReRanker
from src.pipeline import RAGPipeline, source_dicts
from src.parser import parse_html_to_json
import json

INDEX_DIR = Path("vectorstore")

# ── 查询历史（环形缓冲，最多保留 200 条，接口暴露最近 100 条）────
_query_history: deque[dict] = deque(maxlen=200)

# 全局加载（启动时一次）
generator = RAGGenerator()
reranker = ReRanker(enabled=False)  # 默认关闭重排
pipeline = RAGPipeline.from_index(INDEX_DIR, generator=generator, rewrite=True, reranker=reranker)
retriever = pipeline.retriever
all_chunks = pipeline.all_chunks

# ── 拒绝回答检测 ──────────────────────────────────────────
OFF_TOPIC_KEYWORDS = ["忽视", "忽略", "忘记", "ignore", "forget", "prompt",
                      "指令", "instruction", "system", "你是一个", "你是",
                      "前面", "之前", "above", "previous"]
RUST_TOPIC_WORDS = ["rust", "所有权", "borrow", "借用", "cargo", "struct",
                    "enum", "trait", "async", "宏", "macro", "unsafe", "生命周期",
                    "lifetime", "crate", "模块", "错误处理", "泛型", "generic"]


def should_refuse(question: str, merged: list) -> tuple:
    q_lower = question.lower().strip()
    # 注入检测
    kw_count = sum(1 for k in OFF_TOPIC_KEYWORDS if k in q_lower)
    if kw_count >= 2 and len(q_lower) > 30:
        return True, "检测到可疑指令，已拒绝回答。"
    # 非 Rust 检测
    if not merged:
        return True, "未检索到任何相关文档，无法回答。"
    if not any(rw in q_lower for rw in RUST_TOPIC_WORDS):
        if merged[0][1] < 0.01:
            return True, "该问题与 Rust 文档知识库不相关，请提出 Rust 技术相关问题。"
    return False, ""


def do_query(question: str, rerank: bool = False) -> dict:
    """执行完整 RAG 查询，返回结构化结果。"""
    result = pipeline.retrieve(question, top_k=10, rerank=rerank)

    # 拒绝回答检测
    refuse, reason = should_refuse(question, result.child_results)
    if refuse:
        return {
            "question": question,
            "rewrites": result.rewrites,
            "answer": f"⚠️ {reason}",
            "sources": [],
            "refused": True,
        }

    # 生成回答
    result = pipeline.generate_answer(result, max_chunks=8)
    answer = result.answer

    # ── 记录查询历史 ──────────────────────────────────────────
    _query_history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "rewrites": result.rewrites,
        "answer_preview": answer[:300],
        "source_count": len(result.parent_results),
        "top_source_url": (
            result.parent_results[0][0].metadata.get("source_url", "")
            if result.parent_results else ""
        ),
    })

    return {
        "question": question,
        "rewrites": result.rewrites,
        "answer": answer,
        "sources": source_dicts(result.parent_results, limit=8),
    }


# ── FastAPI ──────────────────────────────────────────────
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    print("⚠️ fastapi 未安装。请运行: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="Rag Scratch API", version="1.0")


class QueryRequest(BaseModel):
    question: str
    rerank: bool = False


class ParseRequest(BaseModel):
    html: str
    source_url: str = ""
    source_name: str = "api_incremental"


class EvalRequest(BaseModel):
    question: str
    answer: str
    sources: list[dict] = []


@app.post("/query")
def query_endpoint(req: QueryRequest):
    return do_query(req.question, rerank=req.rerank)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(all_chunks)}


# ── 新增端点 ──────────────────────────────────────────────────

@app.get("/stats")
def stats():
    """评估统计：chunk 数量、索引信息、评估汇总（如有）。"""
    # 索引统计
    index_info = {
        "faiss_index": str(INDEX_DIR / "faiss.index"),
        "children_index": str(INDEX_DIR / "children.json"),
        "bm25_index": str(INDEX_DIR / "bm25.pkl"),
    }
    try:
        index_info["faiss_ntotal"] = retriever.vector_store.index.ntotal
    except Exception:
        index_info["faiss_ntotal"] = None

    try:
        index_info["bm25_doc_count"] = len(retriever.bm25_store.chunks)
    except Exception:
        index_info["bm25_doc_count"] = None

    stats_data = {
        "total_chunks": len(all_chunks),
        "index_info": index_info,
        "query_history_count": len(_query_history),
    }

    # 尝试加载最新评估报告
    eval_dir = Path("evaluation/results")
    if eval_dir.exists():
        reports = sorted(eval_dir.glob("eval_report_*.json"), reverse=True)
        if reports:
            try:
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                stats_data["latest_evaluation"] = {
                    "file": reports[0].name,
                    "summary": report.get("summary", {}),
                    "meta": report.get("meta", {}),
                }
            except Exception:
                stats_data["latest_evaluation"] = {"error": "failed to load report"}

    return stats_data


@app.get("/history")
def history():
    """返回最近 100 条查询日志。"""
    entries = list(_query_history)
    return {
        "total": len(entries),
        "entries": entries[-100:],  # 最近 100 条
    }


@app.post("/parse")
def parse_endpoint(req: ParseRequest):
    """增量数据解析：将 HTML 字符串解析为结构化 JSON 元素列表。"""
    doc = parse_html_to_json(req.html, req.source_url, req.source_name)
    return {
        "source_url": doc.source_url,
        "source_name": doc.source_name,
        "title": doc.title,
        "elements": doc.elements,
        "element_count": len(doc.elements),
    }


@app.post("/evaluate")
def evaluate_endpoint(req: EvalRequest):
    """计算 Faithfulness + Answer Relevancy。"""
    from evaluation.evaluator import RAGEvaluator
    from src.chunkers.narrative import Chunk

    evaluator = RAGEvaluator(retriever, generator, top_k=len(req.sources) or 5)
    chunks = [Chunk(
        chunk_id=s.get("chunk_id", f"src_{i}"),
        text=s.get("text", s.get("preview", "")),
        metadata={"source_url": s.get("source_url", ""),
                   "headings": s.get("headings", ""),
                   "source_name": "eval"},
    ) for i, s in enumerate(req.sources)]

    try:
        ff_score, ff_detail = evaluator.compute_faithfulness(
            req.question, req.answer, chunks)
    except Exception:
        ff_score, ff_detail = None, {}

    try:
        ar_score, ar_detail = evaluator.compute_answer_relevancy(
            req.question, req.answer)
    except Exception:
        ar_score, ar_detail = None, {}

    return {
        "faithfulness": round(ff_score, 4) if ff_score is not None else None,
        "answer_relevancy": round(ar_score, 4) if ar_score is not None else None,
        "faithfulness_detail": ff_detail if isinstance(ff_detail, dict) else None,
        "relevancy_detail": ar_detail if isinstance(ar_detail, dict) else None,
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"🚀 API 启动: http://{args.host}:{args.port}")
    print(f"   POST /query  |  GET /health  |  GET /stats")
    print(f"   GET /history |  POST /parse  |  POST /evaluate")
    uvicorn.run(app, host=args.host, port=args.port)
