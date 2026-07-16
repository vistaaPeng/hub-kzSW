"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）— 多轮对话模式

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
  2. 循环调用闭环：模型输出 tool_call → 宿主执行工具 → 结果以 role=tool 回填 → 模型生成回答
     → 用户继续提问 → 重复循环（保持对话历史）
  3. 并行工具调用：模型一次输出多个 tool_call，宿主逐个执行后一并回填
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 交互式多轮对话（循环调用）
  python mode_function_call/run_function_call.py

  # 单个问题（兼容旧用法）
  python mode_function_call/run_function_call.py --question "宁德时代2023年营收和净利润？"

  # 内置示例问题（演示并行工具调用）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402

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
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
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
    "涉及天气时调用 get_weather。每回合你可以一次调用多个工具。"
)


def process_turn(client, model: str, messages: list, verbose: bool = True) -> str:
    """
    处理单轮对话：根据当前消息列表调用模型，处理工具调用，返回最终回答。
    会修改 messages 列表，添加工具调用和结果。
    """
    t0 = time.time()
    has_tool_call = True

    while has_tool_call:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
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
                if verbose:
                    print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            has_tool_call = False

    messages.append(msg)
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 回答（{elapsed:.1f}s）")
    return msg.content or ""


def run_single(client, model: str, question: str, verbose: bool = True) -> dict:
    """单问题模式：创建新对话，返回结果。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = process_turn(client, model, messages, verbose=verbose)
    tool_calls = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    return {"answer": answer, "tool_calls": tool_calls}


def run_loop(client, model: str, verbose: bool = True):
    """
    循环调用模式：交互式多轮对话，用户可以不断提问，保持对话历史。
    """
    print(f"[Function Call] provider={provider} model={model}")
    print("=" * 60)
    print("欢迎使用 A股年报智能问答助手（多轮对话模式）")
    print("输入问题进行查询，输入 'exit' 或 'quit' 退出")
    print("知识库：贵州茅台/五粮液/宁德时代/海康威视/中国平安（2021-2023年报）")
    print("=" * 60)
    print()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_count = 0

    while True:
        try:
            question = input("你：").strip()
        except EOFError:
            print("\n再见！")
            break

        if question.lower() in ("exit", "quit", "bye"):
            print("再见！")
            break

        if not question:
            continue

        turn_count += 1
        print(f"\n--- 第 {turn_count} 轮 ---")
        print(f"问题：{question}")
        print()

        messages.append({"role": "user", "content": question})
        answer = process_turn(client, model, messages, verbose=verbose)

        print("\n助手：")
        print(answer)
        print()


DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call（多轮对话）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    global provider
    provider = args.provider
    client, model = build_client(provider)

    if args.json:
        questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
        results = []
        for q in questions:
            result = run_single(client, model, q, verbose=False)
            result["question"] = q
            results.append(result)
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
        return

    if args.question:
        print(f"[Function Call] provider={provider} model={model}\n")
        print("=" * 60)
        print(f"问题：{args.question}")
        print("=" * 60)
        result = run_single(client, model, args.question, verbose=not args.quiet)
        print("\n最终回答：")
        print(result["answer"])
        return

    if args.demo:
        print(f"[Function Call] provider={provider} model={model}\n")
        for i, q in enumerate(DEMO_QUESTIONS, 1):
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
            result = run_single(client, model, q, verbose=not args.quiet)
            print("\n最终回答：")
            print(result["answer"])
            print()
        return

    run_loop(client, model, verbose=not args.quiet)


if __name__ == "__main__":
    main()
