"""
RAGAS 自动化评估脚本 — 文献问答版

对原生版 RAG 和 LangChain 版 RAG 进行标准化评测，输出 4 项 RAGAS 核心指标：
  - Faithfulness（忠实度）
  - Answer Relevancy（答案相关性）
  - Context Precision（上下文精确率）
  - Context Recall（上下文召回率）

使用方式：
  python evaluate.py --pipeline native       # 评估原生版
  python evaluate.py --pipeline langchain    # 评估 LangChain 版
  python evaluate.py --pipeline both         # 两版都评估并对比
  python evaluate.py --pipeline native --question-ids 1,2,3

依赖：
  pip install ragas datasets openai
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
EVAL_DIR   = Path(__file__).parent
RESULT_DIR = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL     = "qwen-plus"
EMBED_MODEL   = "text-embedding-v3"


# ── 加载评测题集 ──────────────────────────────────────────────────────────────

def load_questions(question_ids: list[int] = None) -> list[dict]:
    qpath = EVAL_DIR / "questions.json"
    with open(qpath, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    return questions


# ── 原生版 RAG 调用 ──────────────────────────────────────────────────────────

def run_native_rag(questions: list[dict]) -> list[dict]:
    sys.path.insert(0, str(BASE_DIR / "src"))
    from rag_pipeline import RAGPipeline

    logger.info("初始化原生 RAG Pipeline...")
    pipeline = RAGPipeline(use_bm25=True, use_rerank=False)

    results = []
    for q in questions:
        logger.info(f"[{q['id']:02d}] {q['question'][:40]}...")
        try:
            result = pipeline.query(q["question"], verbose=False)
            results.append({
                "question":     q["question"],
                "answer":       result["answer"],
                "contexts":     [r["content"] for r in result["retrieved"]],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
            })
        except Exception as e:
            logger.error(f"  题目 {q['id']} 失败: {e}")
            results.append({
                "question":     q["question"],
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
            })

    return results


# ── LangChain 版 RAG 调用 ────────────────────────────────────────────────────

def run_langchain_rag(questions: list[dict]) -> list[dict]:
    sys.path.insert(0, str(BASE_DIR / "src_langchain"))

    import importlib.util
    spec   = importlib.util.spec_from_file_location(
        "rag_chain_lc",
        BASE_DIR / "src_langchain" / "rag_chain_lc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    embeddings  = module.get_embeddings()
    vectorstore = module.get_vectorstore(embeddings)
    chain, retriever = module.build_chain(vectorstore)

    results = []
    for q in questions:
        logger.info(f"[{q['id']:02d}] {q['question'][:40]}...")
        try:
            retrieved_docs = retriever.invoke(q["question"])
            contexts       = [doc.page_content for doc in retrieved_docs]
            answer         = chain.invoke(q["question"])
            results.append({
                "question":     q["question"],
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
            })
        except Exception as e:
            logger.error(f"  题目 {q['id']} 失败: {e}")
            results.append({
                "question":     q["question"],
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
            })

    return results


# ── RAGAS 评估 ────────────────────────────────────────────────────────────────

def run_ragas_eval(results: list[dict], pipeline_name: str) -> dict:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    logger.info(f"开始 RAGAS 评估（{pipeline_name}）...")

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=DASHSCOPE_URL,
        temperature=0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=DASHSCOPE_URL,
        dimensions=1024,
        check_embedding_ctx_length=False,
    ))

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm       = ragas_llm
        metric.embeddings = ragas_embeddings

    # 过滤掉 should_refuse 类型
    eval_results = [r for r in results if r.get("question_type") != "should_refuse"]

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in eval_results
    ])

    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    import numpy as np
    def _mean(metric_name: str) -> float:
        vals = [v for v in score[metric_name] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    faithfulness_v     = _mean("faithfulness")
    answer_relevancy_v = _mean("answer_relevancy")
    context_precision_v = _mean("context_precision")
    context_recall_v   = _mean("context_recall")

    logger.info(f"\n{'='*50}")
    logger.info(f"RAGAS 评估结果 - {pipeline_name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Faithfulness（忠实度）:       {faithfulness_v:.4f}")
    logger.info(f"  Answer Relevancy（答案相关性）: {answer_relevancy_v:.4f}")
    logger.info(f"  Context Precision（上下文精确率）:{context_precision_v:.4f}")
    logger.info(f"  Context Recall（上下文召回率）:  {context_recall_v:.4f}")

    return {
        "pipeline":           pipeline_name,
        "faithfulness":       faithfulness_v,
        "answer_relevancy":   answer_relevancy_v,
        "context_precision":  context_precision_v,
        "context_recall":     context_recall_v,
    }


# ── 按题型分析 ────────────────────────────────────────────────────────────────

def analyze_by_type(results: list[dict]):
    from collections import defaultdict

    type_stats = defaultdict(list)
    for r in results:
        qtype  = r.get("question_type", "unknown")
        answer = r.get("answer", "")
        refused = any(kw in answer for kw in ["无法回答", "无法提供", "不在", "不包含", "超出"])
        type_stats[qtype].append({"refused": refused, "answer_len": len(answer)})

    print("\n── 按题型统计 ──")
    for qtype, stats in type_stats.items():
        refuse_rate = sum(1 for s in stats if s["refused"]) / len(stats)
        avg_len     = sum(s["answer_len"] for s in stats) / len(stats)
        print(f"  {qtype:<25} 题数={len(stats)}  拒绝率={refuse_rate:.0%}  平均回答长度={avg_len:.0f}字")


# ── 保存结果 ──────────────────────────────────────────────────────────────────

def save_results(raw_results: list[dict], scores: dict, pipeline_name: str):
    import pandas as pd
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULT_DIR / f"{pipeline_name}_{timestamp}.json"

    output = {
        "pipeline":    pipeline_name,
        "timestamp":   timestamp,
        "scores":      scores,
        "raw_results": raw_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存 → {out_path}")

    df = pd.DataFrame([{
        "id":     r["question_id"],
        "type":   r["question_type"],
        "question": r["question"][:30] + "...",
        "answer_len": len(r["answer"]),
        "context_count": len(r["contexts"]),
    } for r in raw_results])
    csv_path = RESULT_DIR / f"{pipeline_name}_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 摘要已保存 → {csv_path}")


# ── 两版对比输出 ──────────────────────────────────────────────────────────────

def print_comparison(native_scores: dict, lc_scores: dict):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels  = {
        "faithfulness":       "忠实度",
        "answer_relevancy":   "答案相关性",
        "context_precision":  "上下文精确率",
        "context_recall":     "上下文召回率",
    }

    print(f"\n{'='*65}")
    print(f"{'指标':<20} {'原生版（DashScope Embed）':>20} {'LangChain版（BGE Embed）':>20}")
    print(f"{'='*65}")
    for m in metrics:
        n_val = native_scores.get(m, 0)
        l_val = lc_scores.get(m, 0)
        diff  = n_val - l_val
        arrow = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "≈")
        print(f"{labels[m]:<20} {n_val:>20.4f} {l_val:>20.4f}  {arrow}")
    print(f"{'='*65}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="文献 RAG 系统 RAGAS 评估")
    parser.add_argument("--pipeline",     choices=["native", "langchain", "both"], default="native")
    parser.add_argument("--question-ids", type=str, default=None, help="指定题号，逗号分隔")
    parser.add_argument("--skip-ragas",   action="store_true", help="跳过 RAGAS 打分")
    args = parser.parse_args()

    q_ids = [int(x) for x in args.question_ids.split(",")] if args.question_ids else None
    questions = load_questions(q_ids)
    logger.info(f"加载 {len(questions)} 道测试题")

    native_results = lc_results = None
    native_scores  = lc_scores  = None

    if args.pipeline in ("native", "both"):
        native_results = run_native_rag(questions)
        analyze_by_type(native_results)
        if not args.skip_ragas:
            native_scores = run_ragas_eval(native_results, "native")
            save_results(native_results, native_scores, "native")

    if args.pipeline in ("langchain", "both"):
        lc_results = run_langchain_rag(questions)
        analyze_by_type(lc_results)
        if not args.skip_ragas:
            lc_scores = run_ragas_eval(lc_results, "langchain")
            save_results(lc_results, lc_scores, "langchain")

    if args.pipeline == "both" and native_scores and lc_scores:
        print_comparison(native_scores, lc_scores)


if __name__ == "__main__":
    main()
