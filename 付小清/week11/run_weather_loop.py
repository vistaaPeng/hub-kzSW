"""
run_weather_loop.py — 作业：天气查询工具调用改造为循环调用（ReAct Agent）

教学对比：
  · 单轮闭环（function_call_mcp_cli 原版）：模型最多调一次工具 → 执行 → 生成最终回答
  · 循环调用（本文件）：while 循环，模型可多次调 get_weather，直到不再输出 tool_calls

典型场景：
  · 多城市依次查询："北京、上海、广州哪个今天最热？"
  · 链式依赖："先查宁德，若下雨再查福州对比"
  · 纠错重试：某城市 geocoding 失败，模型换写法再查

使用方式：
  python run_weather_loop.py -q "北京、上海、广州今天哪个最热？"
  python run_weather_loop.py --demo
  python run_weather_loop.py --demo --max-rounds 5

依赖：
  pip install openai httpx
  环境变量：DEEPSEEK_API_KEY（或 --provider dashscope + DASHSCOPE_API_KEY）
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 复用教学项目的天气后端，避免重复实现
PROJECT_ROOT = Path(__file__).parent.parent / "function_call_mcp_cli"
sys.path.insert(0, str(PROJECT_ROOT))

from src.weather_backend import get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

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

DEFAULT_MAX_ROUNDS = 8


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 工具 Schema（仅天气，聚焦循环调用）──────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的当前天气及未来3天预报。"
                "城市用中文名，如 '宁德'、'北京'、'上海'。"
                "若一次查多个城市，请分多次调用本工具，每次一个城市。"
            ),
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

TOOL_DISPATCH = {"get_weather": get_weather}

SYSTEM_PROMPT = (
    "你是天气查询助手。回答用户关于天气的问题时，必须通过 get_weather 工具获取真实数据，"
    "不要编造温度或降水。若需对比多个城市，请逐个城市调用工具后再总结。"
    "若某城市查询失败，可尝试加'市'后缀或换写法重查。"
    "拿到足够信息后，直接给出最终回答，不要再调用工具。"
)


def execute_tool_call(tc, verbose: bool) -> tuple[dict, str]:
    """执行单个 tool_call，返回 (log_entry, result_text)。"""
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")
    log_entry = {"name": name, "args": args}
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
        print(f"    <- {preview}{'...' if len(result or '') > 120 else ''}\n")
    return log_entry, result


def run_loop(
    client,
    model: str,
    question: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    verbose: bool = True,
) -> dict:
    """
    循环调用：提问 → [模型 tool_call → 执行 → 回填] × N → 最终回答。

    与单轮闭环的区别：内层 while 在模型仍输出 tool_calls 时继续请求 LLM，
    直到模型返回纯文本或达到 max_rounds 上限。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log: list[dict] = []
    rounds_used = 0

    while rounds_used < max_rounds:
        rounds_used += 1
        if verbose:
            print(f"  ── 第 {rounds_used} 轮 LLM 请求 ──")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型不再调工具 → 循环结束，msg.content 即最终回答
        if not msg.tool_calls:
            answer = msg.content or ""
            elapsed = time.time() - t0
            if verbose:
                print(f"  → [llm] 无 tool_calls，循环结束（共 {rounds_used} 轮，{elapsed:.1f}s）")
            return {
                "answer": answer,
                "tool_calls": tool_call_log,
                "rounds": rounds_used,
                "elapsed": elapsed,
            }

        # 仍有 tool_calls → 执行工具并回填，进入下一轮
        messages.append(msg)
        for tc in msg.tool_calls:
            log_entry, result = execute_tool_call(tc, verbose)
            tool_call_log.append(log_entry)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 达到轮数上限：强制要最终回答（不再传 tools，避免继续调工具）
    if verbose:
        print(f"  ⚠ 已达 max_rounds={max_rounds}，强制生成最终回答")
    messages.append({
        "role": "user",
        "content": "已达到最大工具调用轮数，请根据已有工具结果直接给出最终回答。",
    })
    resp = client.chat.completions.create(model=model, messages=messages)
    answer = resp.choices[0].message.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 强制收尾（共 {rounds_used} 轮，{elapsed:.1f}s）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "rounds": rounds_used,
        "elapsed": elapsed,
        "truncated": True,
    }


DEMO_QUESTIONS = [
    "宁德今天天气怎么样？",
    "分别查询北京、上海、广州的天气，告诉我今天哪个城市最高温？",
    "先查深圳天气，如果今天有雨再查相邻的东莞天气做对比，帮我判断哪个更适合户外活动。",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="天气查询 — 循环工具调用（ReAct）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="最大循环轮数")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Weather Loop] provider={args.provider} model={model} max_rounds={args.max_rounds}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run_loop(client, model, q, max_rounds=args.max_rounds, verbose=not args.json)
        result["question"] = q
        results.append(result)
        if not args.json:
            print(f"\n工具调用次数：{len(result['tool_calls'])}，循环轮数：{result['rounds']}")
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
