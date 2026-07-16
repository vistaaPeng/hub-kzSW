"""
run_single_round.py — 对照组：单轮闭环（原版模式）

与 run_weather_loop.py 的唯一核心差异：
  · 本文件：模型输出 tool_calls 后只执行一轮，再请求一次 LLM 即结束
  · loop 版：while 循环，直到模型不再输出 tool_calls

用于作业对比演示「单轮 vs 循环」的行为差异。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent / "function_call_mcp_cli"
sys.path.insert(0, str(PROJECT_ROOT))

from src.weather_backend import get_weather  # noqa: E402
from run_weather_loop import (  # noqa: E402
    DEMO_QUESTIONS,
    PROVIDERS,
    SYSTEM_PROMPT,
    TOOL_DISPATCH,
    TOOLS_SCHEMA,
    build_client,
    execute_tool_call,
)


def run_single_round(client, model: str, question: str, verbose: bool = True) -> dict:
    """单轮闭环：最多执行一次 tool_calls，然后生成最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log: list[dict] = []

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        if verbose:
            print("  ── 单轮：执行 tool_calls（仅此一轮）──")
        messages.append(msg)
        for tc in msg.tool_calls:
            log_entry, result = execute_tool_call(tc, verbose)
            tool_call_log.append(log_entry)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # 第二次请求即结束，无论模型是否还想再调工具
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 单轮结束（工具 {len(tool_call_log)} 次，{elapsed:.1f}s）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "rounds": 1 if tool_call_log else 0,
        "elapsed": elapsed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="天气查询 — 单轮闭环（对照组）")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    args = parser.parse_args()

    client, model = build_client(args.provider)
    print(f"[Weather Single Round] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[1]])
    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"Q{i}：{q}")
        print("=" * 60)
        result = run_single_round(client, model, q)
        print(f"\n工具调用次数：{len(result['tool_calls'])}")
        print("\n最终回答：")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
