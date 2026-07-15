"""
LLM 查询重写器 —— 将用户自然语言问题改写为 2-3 个标准化检索查询。

消解同义表达（"有哪些"/"列出"/"关键字列表"）对 BM25 的不稳定性。
"""

import json
import os
import subprocess
from openai import OpenAI


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

REWRITE_PROMPT = """你是一个搜索查询优化器。将用户的问题改写为 2-3 个更具体的检索查询，
用于在 Rust 文档知识库中检索。

规则：
1. 消除口语化表达，使用文档中可能出现的关键词
2. 每个改写版本从不同角度切入（如：概念定义、代码示例、术语列表）
3. 保持原问题的意图不变
4. 只返回 JSON 数组，不要其他内容

示例：
用户: "rust有哪些关键字"
返回: ["Rust 关键字列表", "Rust 保留字 严格关键字", "Rust 语言关键字 as async await"]

用户: "rust里怎么定义结构体"
返回: ["Rust struct 定义语法", "Rust 结构体 示例代码", "struct 关键字 字段 方法"]

现在请改写以下问题，只返回 JSON 数组："""


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
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")
    return key


class QueryRewriter:
    """LLM 驱动的查询重写器"""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL):
        self.client = OpenAI(
            api_key=_get_api_key(),
            base_url=base_url,
        )
        self.model = model

    def rewrite(self, query: str, n_variants: int = 3) -> list[str]:
        """
        将用户查询改写为 n 个标准化检索查询。

        Args:
            query: 用户原始查询
            n_variants: 生成的变体数量

        Returns:
            改写后的查询列表（始终包含原始查询作为第一个元素）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=256,
            )
            text = response.choices[0].message.content or "[]"
            # 提取 JSON 数组
            variants = _parse_json_array(text)

            # 始终将原始查询放在第一位
            result = [query]
            for v in variants:
                if v != query and v not in result:
                    result.append(v)
            return result[: n_variants + 1]
        except Exception:
            return [query]


def _parse_json_array(text: str) -> list[str]:
    """从 LLM 输出中提取 JSON 数组。"""
    text = text.strip()
    # 去除 markdown 代码块标记
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x) for x in arr if x]
    except json.JSONDecodeError:
        pass
    return []
