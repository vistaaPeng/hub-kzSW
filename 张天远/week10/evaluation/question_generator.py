"""
LLM 驱动的测试题生成器 —— 从 Rust 文档 chunks 中自动生成 RAG 测试题。

题型多样化：概念解释、代码补全、对比分析、术语定义。
如果 DeepSeek API 不可用，退化为基于关键词提取的简单生成。
"""

import os
import random
import re
from openai import OpenAI

from src.chunkers.narrative import Chunk

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

# 题型列表
QUESTION_TYPES = [
    "概念解释",
    "代码补全",
    "对比分析",
    "术语定义",
]

SYSTEM_PROMPT = """你是一个 Rust 语言教学评估专家。请根据提供的文档内容生成一道测试题。

要求：
1. 题目必须基于提供的文档内容，不能编造
2. 题型从以下随机选择一种：概念解释、代码补全、对比分析、术语定义
3. 题目用中文编写，简洁清晰
4. 同时输出题目中的关键概念词（keywords），用逗号分隔

输出格式（严格按以下 JSON 格式，不要输出其他内容）：
{"question": "...", "keywords": ["关键词1", "关键词2", ...]}"""


_api_key_cache: str | None = None

def _get_api_key() -> str:
    """从环境变量获取 DeepSeek API Key（结果缓存）。"""
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        import subprocess
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-ItemProperty 'HKCU:\\Environment' -Name DEEPSEEK_API_KEY"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "DEEPSEEK_API_KEY" in line:
                    key = line.split(":", 1)[-1].strip()
                    break
        except Exception:
            pass
    _api_key_cache = key
    return key


def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """从文本中提取关键概念词（基于简单的启发式规则）。"""
    # 匹配 Rust 相关术语（大写/驼峰/下划线命名、英文关键词）
    rust_terms = re.findall(
        r'\b[A-Z][a-zA-Z0-9_]{2,}\b'   # 大写开头的标识符
        r'|\b[a-z_]+(?:\(\))?\b',       # 小写函数/变量名
        text
    )
    # 过滤掉常见英文停用词和太短的词
    stop_words = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was',
        'can', 'use', 'has', 'its', 'not', 'but', 'all', 'one', 'two',
        'you', 'your', 'will', 'when', 'how', 'what', 'which', 'where',
        'into', 'over', 'than', 'then', 'just', 'like', 'some', 'each',
        'also', 'been', 'have', 'had', 'does', 'did', 'more', 'most',
        'other', 'only', 'such', 'very', 'here', 'there', 'their',
        'about', 'after', 'before', 'between', 'through',
    }
    keywords = []
    seen = set()
    for term in rust_terms:
        low = term.lower().rstrip('()')
        if low in stop_words or len(low) < 3:
            continue
        if low not in seen:
            seen.add(low)
            keywords.append(term)
            if len(keywords) >= max_keywords:
                break
    return keywords


def _generate_fallback_question(chunk: Chunk, qtype: str) -> dict:
    """基于关键词提取的简单测试题生成（API 不可用时的降级方案）。"""
    text = chunk.text
    keywords = _extract_keywords(text)

    # 提取第一个句子作为上下文
    first_sentence = text.split("。")[0].strip() + "。" if "。" in text else text[:100]

    templates = {
        "概念解释": (
            f"请解释以下 Rust 概念：{keywords[0] if keywords else '上述内容'}。"
            f"（提示：{first_sentence[:80]}）"
        ),
        "代码补全": (
            f"阅读以下说明，补全缺失的代码：\n"
            f"// {first_sentence[:100]}\n"
            f"fn example() {{\n    // TODO: 实现上述功能\n}}"
        ),
        "对比分析": (
            f"请对比分析 Rust 中的相关概念。"
            f"（参考：{first_sentence[:100]}）"
        ),
        "术语定义": (
            f"请定义以下 Rust 术语并说明其用途："
            f"{keywords[0] if keywords else '上述术语'}。"
            f"（上下文：{first_sentence[:80]}）"
        ),
    }

    question = templates.get(qtype, templates["概念解释"])

    return {
        "question": question,
        "keywords": keywords if keywords else ["Rust"],
        "source_chunk_id": chunk.chunk_id,
    }


def _generate_llm_question(
    client: OpenAI,
    model: str,
    chunk: Chunk,
    qtype: str,
) -> dict | None:
    """使用 LLM 生成一道测试题。"""
    user_message = f"""请基于以下文档内容生成一道「{qtype}」类型的测试题。

## 文档内容

{chunk.text}

请用中文输出 JSON 格式的题目和关键词。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return None

    # 尝试解析 JSON
    import json
    # 提取 JSON 部分（可能被 markdown 包裹）
    json_match = re.search(r'\{[^}]*"question"[^}]*\}', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "question": parsed.get("question", ""),
                "keywords": parsed.get("keywords", []),
                "source_chunk_id": chunk.chunk_id,
            }
        except json.JSONDecodeError:
            pass

    # JSON 解析失败，回退到简单提取
    return None


def generate_questions(
    chunks: list[Chunk],
    n: int = 60,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> list[dict]:
    """
    从文档 chunks 中生成测试题。

    Args:
        chunks: 文档 chunk 列表
        n: 需要生成的题目数量
        model: LLM 模型名
        base_url: API 地址

    Returns:
        测试题列表，每道题：
        {"question": "...", "keywords": ["..."], "source_chunk_id": "..."}
    """
    if not chunks:
        return []

    api_key = _get_api_key()
    use_llm = bool(api_key)

    client = None
    if use_llm:
        client = OpenAI(api_key=api_key, base_url=base_url)

    # 随机抽取 chunks（有放回，允许同一 chunk 出多道不同类型的题）
    if n <= len(chunks):
        selected = random.sample(chunks, n)
    else:
        selected = random.choices(chunks, k=n)

    questions: list[dict] = []
    for chunk in selected:
        qtype = random.choice(QUESTION_TYPES)

        if use_llm and client:
            q = _generate_llm_question(client, model, chunk, qtype)
            if q is None:
                # LLM 调用失败，退化为简单生成
                q = _generate_fallback_question(chunk, qtype)
        else:
            q = _generate_fallback_question(chunk, qtype)

        questions.append(q)

    return questions
