"""
测试 Faith=0 的三道题的根因

用法: python tests/test_zero_faith_root_cause.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

path = "evaluation/results/eval_20260706_183519.json"
r = json.load(open(path, encoding="utf-8"))
zero_ids = ["m26", "m34", "m36"]
details = {d["question_id"]: d for d in r["details"]}

for tid in zero_ids:
    d = details[tid]
    answer = d.get("answer", "")
    print(f"=== {tid} ===")
    print(f"Q: {d['question']}")
    print(f"AR: {d.get('answer_relevancy', '?')}")
    print(f"P:  {d.get('context_precision', '?')}")
    print()

    # 检查三个关键pattern
    patterns = {
        "诚实答'不知道'": "文档中未提供" in answer or "无法进一步说明" in answer or "只说了" in answer,
        "含LLM自造术语": "超类trait" in answer or "超类" in answer,
        "含代码片段(不在chunk中)": "```" in answer,
        "问题含'区别/对比'(推导类)": "区别" in d["question"] or "对比" in d["question"],
        "只引述文档未综合": "来源" in answer and len(answer) < 500,
    }
    for name, detected in patterns.items():
        flag = "⚠️" if detected else "  "
        print(f"  {flag} {name}: {detected}")
    print()

# 总结
print("=" * 50)
print("结论:")
print("m36: 评估器惩罚了正确的'不知道'回答 → 需在prompt中豁免")
print("m26: 推导类问题 + 代码示例 → 需强化推导类prompt")
print("m34: 回答质量好但评估太严 → prompt需对'引用文档X'声明更宽容")
