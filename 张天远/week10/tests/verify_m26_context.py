"""
验证 m26 的 /evaluate 端点收到的 context 是否正确

用法: python tests/verify_m26_context.py
"""
import json, sys, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

API = "http://127.0.0.1:8000"

# 从最新的 eval 结果中取 m26 的数据
results_dir = Path("evaluation/results")
files = sorted(results_dir.glob("eval_*.json"), reverse=True)
latest = files[0]
r = json.load(open(latest, encoding="utf-8"))
m26 = next(d for d in r["details"] if d["question_id"] == "m26")

question = m26["question"]
answer = m26.get("answer", "")
retrieved = m26.get("retrieved_chunks", [])

print(f"Question: {question[:80]}")
print(f"Answer (first 200): {answer[:200]}")
print(f"\nRetrieved chunks ({len(retrieved)}):")
for i, rc in enumerate(retrieved):
    text = rc.get("text", rc.get("preview", ""))
    print(f"\n  [{i+1}] headings={rc.get('headings','')[:60]}")
    print(f"       text_len={len(text)}")
    print(f"       text_first={text[:200]}")

# Now call /evaluate and check what it returns
print(f"\n\n=== Calling /evaluate endpoint ===")
resp = requests.post(
    f"{API}/evaluate",
    json={
        "question": question,
        "answer": answer,
        "sources": [{"text": rc.get("text", rc.get("preview", "")),
                     "preview": rc.get("preview", "")}
                    for rc in retrieved[:5]],
    },
    timeout=120,
)
if resp.status_code == 200:
    data = resp.json()
    print(f"Faith: {data.get('faithfulness')}")
    fd = data.get("faithfulness_detail")
    if fd and fd.get("claims"):
        for c in fd["claims"][:6]:
            print(f"  {'✅' if c['supported'] else '❌'} {c['statement'][:100]}")
else:
    print(f"ERROR: {resp.status_code} {resp.text[:200]}")
