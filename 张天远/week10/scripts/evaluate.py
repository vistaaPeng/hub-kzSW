#!/usr/bin/env python
"""
RAG 评估脚本 —— 四指标全量评估 + 生成详细报告
---
用法: python scripts/evaluate.py [--no-llm] [--questions evaluation/questions.json]
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.generator import RAGGenerator
from evaluation.evaluator import RAGEvaluator, EvalQuestion
from src.pipeline import RAGPipeline


def load_default_questions() -> list[EvalQuestion]:
    """从 evaluation/questions.json 加载，不存在则生成默认题。"""
    qpath = Path("evaluation/questions.json")
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8"))
        return [EvalQuestion(**d) for d in data]

    # 默认 10 道测试题
    defaults = [
        ("q01", "Rust 中什么是所有权？", ["所有权", "owner", "内存"]),
        ("q02", "Rust 有哪些关键字？", ["关键字", "keyword", "as", "async"]),
        ("q03", "Rust 中如何定义结构体？", ["struct", "结构体", "定义"]),
        ("q04", "Rust 的借用规则是什么？", ["借用", "borrow", "引用"]),
        ("q05", "Rust 中 match 表达式怎么使用？", ["match", "模式匹配", "分支"]),
        ("q06", "Rust 的泛型如何定义？", ["泛型", "generic", "类型参数"]),
        ("q07", "Rust 中什么是生命周期？", ["生命周期", "lifetime", "引用有效"]),
        ("q08", "Rust 如何使用 cargo 创建项目？", ["cargo", "new", "项目"]),
        ("q09", "Rust 的 trait 是什么？", ["trait", "特质", "接口"]),
        ("q10", "Rust 如何处理错误？", ["错误处理", "Result", "panic", "unwrap"]),
    ]
    return [EvalQuestion(qid, q, kw) for qid, q, kw in defaults]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG 四指标评估")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 指标")
    parser.add_argument("--no-rewrite", action="store_true", help="禁用查询重写")
    parser.add_argument("--questions", type=str, help="自定义问题 JSON 路径")
    parser.add_argument("--output", type=str, default="evaluation/report.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # 加载
    print("🔧 加载索引...")
    generator = RAGGenerator()
    pipeline = RAGPipeline.from_index(
        "vectorstore",
        generator=generator,
        rewrite=not args.no_rewrite,
    )

    # 加载问题
    if args.questions:
        data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        questions = [EvalQuestion(**d) for d in data]
    else:
        questions = load_default_questions()

    evaluator = RAGEvaluator(top_k=args.top_k, pipeline=pipeline)

    print(f"\n📊 评估 {len(questions)} 道题 | LLM指标: {'✅' if not args.no_llm else '❌'}")
    print(f"   查询重写: {'✅' if pipeline.rewriter else '❌'}\n")

    results = evaluator.evaluate_all(questions, compute_llm=not args.no_llm)

    # 生成报告
    report = evaluator.generate_report(questions, results, args.output, not args.no_llm)
    print(f"\n📄 报告: {report}")

    # 控制台摘要
    s = evaluator.summarize(results)
    print(f"\n📊 评估摘要:")
    print(f"   Context Precision: {s['avg_precision']}")
    print(f"   MRR:               {s['avg_mrr']}")
    if s['avg_faithfulness']:
        print(f"   Faithfulness:      {s['avg_faithfulness']}")
    if s['avg_relevancy']:
        print(f"   Answer Relevancy:  {s['avg_relevancy']}")


if __name__ == "__main__":
    main()
