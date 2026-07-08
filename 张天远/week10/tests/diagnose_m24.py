"""
诊断 m24 (Clone vs Copy) Faith=0 的完整答案和检索来源

用法: python tests/diagnose_m24.py
"""
import json
from pathlib import Path

results_dir = Path("evaluation/results")
files = sorted(results_dir.glob("eval_*.json"), reverse=True)
latest = files[0]

r = json.load(open(latest, encoding="utf-8"))
m24 = next(d for d in r["details"] if d["question_id"] == "m24")

print("=" * 70)
print(f"Q: {m24['question']}")
print(f"\nFaith: {m24.get('faithfulness')}")
print(f"Precision: {m24.get('context_precision')}")
print(f"MRR: {m24.get('mrr')}")
print(f"Relevancy: {m24.get('answer_relevancy')}")
print(f"Sources: {m24.get('num_sources')}")

print(f"\n{'─' * 70}")
print("完整回答:")
print(m24.get("answer", "N/A"))

print(f"\n{'─' * 70}")
print("检索片段:")
for i, src in enumerate(m24.get("sources_preview", [])[:5]):
    print(f"  [{i+1}] {src[:200]}")
