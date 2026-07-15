"""
LLM 生成器 —— 调用 DeepSeek API 基于检索到的 chunk 生成答案。
"""

import os
from openai import OpenAI

from src.chunkers.narrative import Chunk
from src.glossary import get_glossary, format_glossary_for_prompt

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = """你是一个 Rust 语言知识助手。请根据提供的文档内容回答用户的问题。

要求：
1. 严格只使用提供的文档内容回答，禁止编造或推测
2. 如果文档中没有明确说明，请说"文档中未提供，无法回答"，不要补充外部知识
3. 引用具体的文档片段时，注明来源
4. 保持回答全面、准确，覆盖所有相关要点
5. 所有 Rust 专业术语必须使用文档或术语表中的标准译名（如"父 trait"而非自造"超类trait"，"关联类型"而非"关联类型占位符"）。术语表提供了权威的中英对照，请严格遵循，禁止自行翻译或创造新词"""


def _get_api_key() -> str:
    """从环境变量或 Windows 注册表获取 DeepSeek API Key。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key and key.startswith("sk-") and len(key) > 20:
        return key
    # 回退：从 Windows 注册表读取
    try:
        import winreg
        reg = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        key, _ = winreg.QueryValueEx(reg, "DEEPSEEK_API_KEY")
        winreg.CloseKey(reg)
        if key:
            return key
    except Exception:
        pass
    # 最后尝试 PowerShell
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
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请在环境变量中设置 API Key。"
        )
    return key


class RAGGenerator:
    """RAG 答案生成器"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or _get_api_key()
        self.client = OpenAI(
            api_key=api_key or _get_api_key(),
            base_url=base_url,
        )
        # 加载术语表（首次调用会下载）
        self.glossary = get_glossary()

    def generate(
        self,
        query: str,
        chunks: list[Chunk],
        max_chunks: int = 5,
        temperature: float = 0.1,
        glossary_terms: list[dict] | None = None,
    ) -> str:
        """
        基于检索到的 chunk 生成答案。

        Args:
            query: 用户问题
            chunks: 检索到的相关 chunk 列表（按相关度排序）
            max_chunks: 最多使用的 chunk 数量
            temperature: LLM 温度

        Returns:
            生成的答案文本
        """
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            src = chunk.metadata.get("source_name", "unknown")
            headings = chunk.metadata.get("headings", "")
            header = f"[文档 {i}] 来源: {src}"
            if headings:
                header += f" | 章节: {headings}"
            context_parts.append(f"{header}\n{chunk.text}")

        context = "\n\n---\n\n".join(context_parts)

        user_message = f"""请根据以下文档内容回答问题。

## 文档内容

{context}

## 用户问题

{query}

请用中文回答。"""

        # 构建 system prompt（含术语表——动态检索优先）
        system_prompt = SYSTEM_PROMPT
        if glossary_terms:
            term_lines = ["## 相关 Rust 术语中英对照（请使用以下标准译名）",
                          "| 英文 | 中文 |",
                          "|------|------|"]
            for t in glossary_terms[:10]:
                term_lines.append(f"| {t['english']} | {t['chinese']} |")
            system_prompt += "\n\n" + "\n".join(term_lines)
        elif self.glossary:
            glossary_text = format_glossary_for_prompt(self.glossary)
            system_prompt += f"\n\n{glossary_text}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1024,
        )

        return response.choices[0].message.content or ""


def build_prompt(query: str, chunks: list[Chunk], max_chunks: int = 5) -> str:
    """
    构建发送给 LLM 的 prompt（用于调试和评估）。

    Args:
        query: 用户问题
        chunks: chunk 列表
        max_chunks: 最多使用的 chunk 数量

    Returns:
        完整的 prompt 文本
    """
    context_parts = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        context_parts.append(f"[文档 {i}]\n{chunk.text}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

## 文档内容

{context}

## 用户问题

{query}

请用中文回答。"""
