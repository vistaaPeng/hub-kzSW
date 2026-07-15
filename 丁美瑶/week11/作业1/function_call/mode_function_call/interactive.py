"""
interactive.py — 交互式 Function Call 运行模式

使用方式：
  python mode_function_call/interactive.py --provider dashscope

特性：
  - 连续对话模式，无需每次重启
  - 支持多轮上下文
  - 输入 exit/quit/q 退出
  - 输入 clear 清空对话历史
  - 输入 help 查看帮助
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies
from src.weather_backend import get_weather

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                            "把公司名写进 query 会稀释检索精度。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定位置的当前天气及未来3天预报。可以通过城市名查询，也可以直接传入经纬度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'。若提供 lat/lon，则忽略此参数"},
                    "lat": {"type": "number", "description": "纬度，例如 26.64"},
                    "lon": {"type": "number", "description": "经度，例如 119.52"},
                },
                "required": [],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_weather": get_weather,
}

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。本回合你可以一次调用多个工具。"
)


def run_once(client, model: str, messages: list):
    t0 = time.time()
    round_num = 0
    max_rounds = 10

    while round_num < max_rounds:
        round_num += 1
        print(f"  ── 第 {round_num} 轮 ──")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"  → [tool] {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    answer = msg.content or ""
    elapsed = time.time() - t0
    print(f"  → [llm] 回答（共 {round_num} 轮，{elapsed:.1f}s）")
    return answer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="交互式 Function Call")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    args = parser.parse_args()

    client, model = build_client(args.provider)
    print(f"[交互式 Function Call] provider={args.provider} model={model}")
    print("=" * 60)
    print("输入问题开始对话，输入 exit/quit/q 退出，输入 clear 清空历史")
    print("支持查询：A股年报数据（贵州茅台/五粮液/宁德时代/海康威视/中国平安）")
    print("支持查询：城市天气")
    print("=" * 60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = input("\n你: ").strip()
        except EOFError:
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        if question.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("对话历史已清空")
            continue

        if question.lower() == "help":
            print("""
可用命令：
  exit/quit/q   - 退出程序
  clear         - 清空对话历史
  help          - 显示帮助

支持的查询：
  1. A股年报数据（仅限以下公司的2021-2023年数据）：
     - 贵州茅台 (600519)
     - 五粮液 (000858)
     - 宁德时代 (300750)
     - 海康威视 (002415)
     - 中国平安 (601318)
     
  2. 天气查询（支持城市名或经纬度）：
     - 城市名：如 "北京天气"、"宁德今天天气"
     - 经纬度：如 "北纬39.9度东经116.4度天气"、"26.64, 119.52天气"

示例问题：
  - 宁德时代2023年营收和净利润是多少？
  - 贵州茅台2023年营收是多少？
  - 北京今天天气如何？
  - 北纬26.64度东经119.52度的天气怎么样？
  - 宁德时代2023年营收多少？另外宁德天气怎么样？
""")
            continue

        messages.append({"role": "user", "content": question})
        answer = run_once(client, model, messages)
        messages.append({"role": "assistant", "content": answer})
        print("\nAI:")
        print(answer)


if __name__ == "__main__":
    main()
