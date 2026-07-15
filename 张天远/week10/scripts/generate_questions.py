"""
生成 60 道评估题（利用 question_generator + 真实 chunk）
用法: python scripts/generate_questions.py
"""
import json
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.narrative import Chunk
from evaluation.question_generator import generate_questions

# 采样 60 个子块作为种子
def main():
    children_path = Path("vectorstore/children.json")
    data = json.loads(children_path.read_text(encoding="utf-8"))
    children = [Chunk(d["chunk_id"], d["text"], d["metadata"],
                      d.get("parent_chunk_id"), d.get("is_parent", False))
                for d in data]

    # 按 source 均匀采样，确保覆盖 6 个来源
    by_source = {}
    for c in children:
        src = c.metadata.get("source_name", "other")
        by_source.setdefault(src, []).append(c)

    sampled = []
    per_source = max(10, 60 // len(by_source))
    for src, chunks in by_source.items():
        n = min(per_source, len(chunks))
        sampled.extend(random.sample(chunks, n))

    random.shuffle(sampled)
    sampled = sampled[:60]

    print(f"用 {len(sampled)} 个子块生成 60 道题...")
    questions = generate_questions(sampled, n=60)

    # 过滤代码补全类问题（缺少上下文，无法合理评估）
    def is_conceptual(q: dict) -> bool:
        text = q.get("question", "")
        bad_patterns = ["补全缺失的代码", "补全代码", "fn example()",
                       "TODO:", "fn main()", "#[cfg", "#![allow"]
        return not any(p in text for p in bad_patterns)

    questions = [q for q in questions if is_conceptual(q)]
    print(f"  过滤后保留 {len(questions)} 道概念类题目（剔除代码补全题）")

    # 转换为 EvalQuestion 格式
    formatted = []
    for i, q in enumerate(questions, 1):
        formatted.append({
            "question_id": f"q{i:03d}",
            "question": q.get("question", ""),
            "keywords": q.get("keywords", []),
        })

    out_path = Path("evaluation/questions.json")
    out_path.write_text(
        json.dumps(formatted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ {len(formatted)} 道题 → {out_path}")


if __name__ == "__main__":
    main()
