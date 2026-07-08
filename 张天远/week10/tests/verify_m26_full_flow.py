"""
验证 m26 完整链路：/query → /evaluate（模拟 Web eval 的 sources[:8]）
"""
import requests, json

API = "http://127.0.0.1:8000"
Q = "Rust 中什么是函数指针？fn 类型和 Fn trait 有什么区别？"

print(f"=== Step 1: /query ===")
r1 = requests.post(f"{API}/query", json={"question": Q}, timeout=60)
q = r1.json()
sources = q.get('sources', [])
answer = q.get('answer', '')
print(f"Sources: {len(sources)}, Answer: {len(answer)} chars")

# 显示每个 source 的关键信息
for i, s in enumerate(sources[:8]):
    has_fn = '函数指针' in s.get('text','')[:200]
    text_len = len(s.get('text',''))
    print(f"  [{i+1}] {'✓' if has_fn else ' '} text_len={text_len} headings={s.get('headings','')[:50]}")

print(f"\n=== Step 2: /evaluate (sources[:8]) ===")
eval_sources = [{"text": s.get("text", s.get("preview", "")),
                  "preview": s.get("preview", "")[:300]}
                 for s in sources[:8]]
r2 = requests.post(f"{API}/evaluate",
    json={"question": Q, "answer": answer[:1000], "sources": eval_sources},
    timeout=120)
e = r2.json()

print(f"Faith: {e.get('faithfulness')}")
print(f"Faith detail claims:")
for c in e.get('faithfulness_detail',{}).get('claims',[])[:5]:
    status = '✅' if c.get('supported') else '❌'
    print(f"  {status} {c.get('statement','')[:100]}")
