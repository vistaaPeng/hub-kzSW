"""
医学科普 RAG 问答 — 命令行入口

用法：
  python src/qa.py                                          # 交互模式
  python src/qa.py --query "感冒有哪些症状"                    # 单次提问
  python src/qa.py --query "头痛伴发热怎么办" --retrieve-only  # 仅检索
  python src/qa.py --query "收缩压140" --no-bm25               # 关闭 BM25 对比
"""

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from rag import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEMO_QUESTIONS = [
    "感冒有哪些主要症状？",
    "高血压的诊断标准是什么？",
    "糖尿病典型的三多一少是指什么？",
    "突发剧烈胸痛伴随大汗应该怎么办？",
    "咳嗽超过3周不好应该考虑哪些原因？",
]


def print_result(question: str, result: dict, retrieve_only: bool = False):
    print(f"\n{'=' * 60}")
    print(f"问题：{question}")
    print(f"{'=' * 60}")

    if retrieve_only:
        print("\n── 检索结果 ──")
        for item in result.get("retrieved", []):
            score_parts = []
            if "vec_score" in item:
                score_parts.append(f"vec={item['vec_score']:.3f}")
            if "bm25_score" in item:
                score_parts.append(f"bm25={item['bm25_score']:.2f}")
            if "rrf_score" in item:
                score_parts.append(f"rrf={item['rrf_score']:.4f}")
            if not score_parts:
                score_parts.append(f"score={item.get('score', 0):.3f}")
            score_str = " ".join(score_parts)
            print(f"  [{score_str}] {item.get('source', '')} · {item.get('section', '')}")
            print(f"         {item['content'][:120]}...")
        return

    print(f"\n{result['answer']}")
    if result.get("citations"):
        print("\n── 来源 ──")
        for c in result["citations"]:
            print(f"  {c['source']}")


def main():
    parser = argparse.ArgumentParser(description="医学科普 RAG 问答系统")
    parser.add_argument("--query", type=str, default=None, help="单次提问")
    parser.add_argument("--retrieve-only", action="store_true", help="仅检索，不调用 LLM")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回数量")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25，仅用向量检索")
    args = parser.parse_args()

    pipeline = RAGPipeline(use_bm25=not args.no_bm25)

    if args.query:
        if args.retrieve_only:
            retrieved = pipeline.retrieve(args.query, args.top_k)
            print_result(args.query, {"retrieved": retrieved}, retrieve_only=True)
        else:
            result = pipeline.query(args.query, args.top_k)
            print_result(args.query, result)
        return

    print("医学科普 RAG 问答系统")
    print(f"检索模式：{'向量 + BM25 混合' if pipeline.use_bm25 else '仅向量'}")
    print("【免责声明】本系统仅供参考，不能替代医生诊断。输入 'exit' 退出，'demo' 运行示例问题\n")

    while True:
        try:
            q = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "demo":
            for dq in DEMO_QUESTIONS:
                result = pipeline.query(dq)
                print_result(dq, result)
            continue

        result = pipeline.query(q)
        print_result(q, result)


if __name__ == "__main__":
    main()
