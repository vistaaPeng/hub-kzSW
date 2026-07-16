"""
run_mcp.py — 方式二：MCP Host（连接多 Server，循环调用多轮对话）

教学重点：
  1. 工具来自"协议发现"而非手写：connect_all_servers 一次走完
     stdio_client 建管道 → initialize() 握手 → list_tools() 发现工具
  2. MCP 工具描述要转成 LLM 能懂的 OpenAI tools schema（inputSchema → parameters）
  3. 循环调用闭环：保持对话历史，多轮对话中模型可以反复调用工具
  4. AsyncExitStack 统一管理多个 Server 子进程的生命周期

使用方式：
  # 交互式多轮对话（循环调用）
  python mode_mcp/run_mcp.py

  # 单个问题（兼容旧用法）
  python mode_mcp/run_mcp.py --question "宁德时代2023年营收和净利润？"
  python mode_mcp/run_mcp.py --demo

依赖：
  pip install mcp openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）
            DASHSCOPE_API_KEY（Embedding，rag_server 内部用）
"""

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

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


def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "rag": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "rag_server.py")],
            env={**os.environ},
        ),
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


async def connect_all_servers(stack: AsyncExitStack):
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        print(f"  ✓ [{label}]  {', '.join(t.name for t in tools_result.tools)}", file=sys.stderr)

    print(f"\n共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。每回合你可以一次调用多个工具。"
)


async def process_turn(client, model: str, messages: list,
                       tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> str:
    """处理单轮对话：调用模型，处理工具调用，返回最终回答。会修改 messages 列表。"""
    t0 = time.time()
    has_tool_call = True

    while has_tool_call:
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"  → [mcp] {name}({args})")

                session, label = tool_registry.get(name, (None, None))
                if session is None:
                    result = f"未知工具：{name}"
                else:
                    call_result = await session.call_tool(name, args)
                    result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

                preview = (result or "")[:120].replace("\n", " ")
                if verbose:
                    print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            has_tool_call = False

    messages.append(msg)
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 回答（{elapsed:.1f}s）")
    return msg.content or ""


async def run_single_async(client, model: str, question: str,
                           tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> dict:
    """单问题模式：创建新对话，返回结果。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = await process_turn(client, model, messages, tool_registry, openai_tools, verbose=verbose)
    tool_calls = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    return {"answer": answer, "tool_calls": tool_calls}


async def run_loop_async(client, model: str, provider: str,
                         tool_registry: dict, openai_tools: list[dict], verbose: bool = True):
    """循环调用模式：交互式多轮对话。"""
    print(f"[MCP] provider={provider} model={model}")
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
            question = await asyncio.to_thread(input, "你：").strip()
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
        answer = await process_turn(client, model, messages, tool_registry, openai_tools, verbose=verbose)

        print("\n助手：")
        print(answer)
        print()


DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",
]


async def main_async(provider: str, question: str | None, demo: bool, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)

        if as_json:
            questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
            results = []
            for q in questions:
                result = await run_single_async(client, model, q, tool_registry, openai_tools, verbose=False)
                result["question"] = q
                results.append(result)
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
            return

        if question:
            print("=" * 60)
            print(f"问题：{question}")
            print("=" * 60)
            result = await run_single_async(client, model, question, tool_registry, openai_tools, verbose=verbose)
            print("\n最终回答：")
            print(result["answer"])
            return

        if demo:
            for i, q in enumerate(DEMO_QUESTIONS, 1):
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
                result = await run_single_async(client, model, q, tool_registry, openai_tools, verbose=verbose)
                print("\n最终回答：")
                print(result["answer"])
                print()
            return

        await run_loop_async(client, model, provider, tool_registry, openai_tools, verbose=verbose)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP（多轮对话）")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.provider, args.question, args.demo, verbose=not args.quiet, as_json=args.json))


if __name__ == "__main__":
    main()
