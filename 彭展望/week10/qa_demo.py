"""
第十周作业：最简单的检索增强问答（RAG）demo

文档：贵州茅台2023年年度报告（复用课程素材里已解析好的 JSON，
      ../week10 检索增强生成RAG/rag_annual_report/data/parsed/ 下的 PDF 解析结果）

方法：
  1. 切片 —— 按解析出的原始 block 切片：表格独立成片，避免表格内容被和
     前后无关的正文混在一起；短文本块顺序拼接到 ~200 字再切一片。
  2. 检索 —— BM25（rank_bm25）+ jieba 分词。只保留长度 >=2 的词（过滤掉
     "的/是/了" 这类单字虚词），否则财报里反复出现的"贵州茅台/2023/年"
     这种高频词会把真正相关但只出现一次关键词（比如"净利润"）的片段淹没。
  3. 生成 —— 把 Top-K 片段和问题一起丢给 LLM（DeepSeek，OpenAI 兼容接口），
     要求"只根据参考资料回答，不要编造"；没有配置 API Key 时自动降级为
     直接展示检索到的原文片段，所以整个流程在没有 key 的情况下也能跑通。

运行：
  1) 把 .env.example 复制成 .env，填入你的 DEEPSEEK_API_KEY（不会被提交到 git）
  2) python qa_demo.py                     # 跑内置的几个示例问题
     python qa_demo.py -q "贵州茅台2023年的净利润是多少？"
"""

import os
import re
import json
import argparse
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).parent
DOC_PATH = (
    BASE_DIR.parent
    / "week10 检索增强生成RAG"
    / "rag_annual_report"
    / "data"
    / "parsed"
    / "600519_2023_贵州茅台_贵州茅台2023年年度报告.json"
)

CHUNK_SIZE = 200  # 非表格正文合并成一片的大致字数
TOP_K = 5

SYSTEM_PROMPT = "你是一个财报问答助手。只根据用户提供的【参考资料】回答问题，不要编造资料之外的数据；如果参考资料里找不到答案，就明确说找不到。回答尽量简短，一两句话说清楚。"


def load_env_file(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_chunks(doc_path: Path, chunk_size: int = CHUNK_SIZE) -> list[str]:
    blocks = json.loads(doc_path.read_text(encoding="utf-8"))["blocks"]

    chunks, buf = [], ""
    for b in blocks:
        content = b["content"].strip()
        if not content:
            continue
        # 表格 / 长文本块单独成片，避免和无关内容混在一起稀释检索信号；
        # 但前面挂着的短标题（比如"三、...业务情况"）要接着一起入片，
        # 不然标题和它下面的正文/表格拆成两片，检索时经常只命中标题那片
        if b["block_type"] == "table" or len(content) >= 120:
            if buf.strip():
                content = buf.strip() + "\n" + content
                buf = ""
            chunks.append(content)
        else:
            buf += content + "\n"
            if len(buf) >= chunk_size:
                chunks.append(buf.strip())
                buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def normalize(text: str) -> str:
    # PDF 表格单元格换行会把词切断（比如"净利 润"），先去掉所有空白再分词
    return re.sub(r"\s+", "", text)


def tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(normalize(text)) if len(w) >= 2 or w.isdigit()]


def build_index(chunks: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(c) for c in chunks])


def retrieve(question: str, chunks: list[str], bm25: BM25Okapi, top_k: int = TOP_K):
    scores = bm25.get_scores(tokenize(question))
    top_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def llm_answer(question: str, contexts: list[str]):
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    if os.environ.get("DEEPSEEK_API_KEY"):
        base_url, model = "https://api.deepseek.com", "deepseek-chat"
    elif os.environ.get("DASHSCOPE_API_KEY"):
        base_url, model = "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"
    else:
        base_url, model = None, "gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    context_block = "\n---\n".join(contexts)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context_block}\n\n【问题】{question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def answer(question: str, chunks: list[str], bm25: BM25Okapi, top_k: int = TOP_K):
    hits = retrieve(question, chunks, bm25, top_k)
    if not hits:
        return "未在文档中检索到相关内容。", hits

    contexts = [text for text, _ in hits]
    generated = llm_answer(question, contexts)
    if generated:
        return generated, hits
    return "[未配置 LLM key，展示检索到的最相关原文片段]\n" + contexts[0], hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=str, default=None)
    args = parser.parse_args()

    load_env_file(BASE_DIR / ".env")

    chunks = load_chunks(DOC_PATH)
    bm25 = build_index(chunks)

    questions = [args.question] if args.question else [
        "公司2023年的营业收入是多少？",
        "公司2023年的净利润是多少？",
        "公司的法定代表人是谁？",
        "公司主要从事什么业务？",
    ]

    print(f"文档：{DOC_PATH.name}，共切出 {len(chunks)} 个检索片段\n")
    for q in questions:
        ans, hits = answer(q, chunks, bm25)
        print(f"问：{q}")
        print(f"答：{ans}")
        print(f"[命中片段 BM25 得分] {[round(s, 2) for _, s in hits]}")
        print()


if __name__ == "__main__":
    main()
