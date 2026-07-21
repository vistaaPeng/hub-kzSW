"""
第十一周作业：把天气查询的工具调用改造为循环调用

课堂代码（../week11 工具调用/function_call_mcp_cli/mode_function_call/run_function_call.py）
里的 run() 是"单轮闭环"：请求一次 → 如果有 tool_calls 就执行 → 再请求一次拿最终回答，
第二次请求不再允许模型继续调工具。这种写法在"参数一次性摆齐"的场景没问题（比如同时查
多个城市，模型会在同一轮里并行发出多个 tool_call），但遇到"要看到上一次工具结果才能
决定下一步查什么"的场景就不够用——模型明明还想再调一次工具，却被强制去憋一个最终回答。

本作业把 get_weather 的调用改成 while 循环：只要模型还在产出 tool_calls 就一直执行、
回填结果、再请求，直到模型不再要求调用工具为止（即"循环调用"），并加最大轮数兜底防止
死循环。用一个必须分两步决策的问题验证：先查 A 城市天气，模型根据结果自己判断要不要
再查 B 城市。

运行：
  export DEEPSEEK_API_KEY=sk-xxx   # 已写进 ~/.zshrc，一般不用重复设置
  python weather_loop_demo.py            # 跑内置的两个示例问题
  python weather_loop_demo.py -q "自定义问题"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# 复用课堂素材里的天气后端，不重复实现 geocoding / 天气请求逻辑
COURSE_PROJECT = (
    Path(__file__).parent.parent / "week11 工具调用" / "function_call_mcp_cli"
)
sys.path.insert(0, str(COURSE_PROJECT))
from src.weather_backend import get_weather  # noqa: E402

MAX_ROUNDS = 5  # 循环上限，防止模型异常时无限调工具

TOOLS_SCHEMA = [
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

SYSTEM_PROMPT = (
    "你是一个出行助手，负责回答天气相关问题。需要查天气时调用 get_weather 工具，城市名用中文。"
    "如果问题需要先看到一个城市的天气结果才能决定要不要再查另一个城市，就先只调用第一个城市，"
    "拿到结果后自己判断是否需要继续调用；不要在没有依据的情况下一次性猜测调用多个城市。"
    "拿到所有需要的信息后再给出最终回答。"
)


def build_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("错误：未设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), "deepseek-chat"


def run_loop(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    循环调用：只要模型还要调工具就一直"请求 → 执行 → 回填"，直到模型不再产出
    tool_calls 为止，而不是像单轮闭环那样固定只跑两次请求。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_call_log = []
    rounds = 0

    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:
            return {
                "answer": "[超过最大循环轮数，强制结束]",
                "tool_calls": tool_call_log,
                "rounds": rounds - 1,
            }

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型不再要求调用工具 → 循环结束，返回最终回答
        if not msg.tool_calls:
            return {"answer": msg.content or "", "tool_calls": tool_call_log, "rounds": rounds}

        if verbose:
            print(f"  第 {rounds} 轮：模型要求调用 {len(msg.tool_calls)} 个工具")

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"round": rounds, "name": name, "args": args})
            if verbose:
                print(f"    → {name}({args})")
            if name == "get_weather":
                result = get_weather(**args)
            else:
                result = f"未知工具：{name}"
            preview = (result or "")[:80].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}...")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # 不 return，回到 while 顶部继续下一轮，直到模型主动停止调工具


DEMO_QUESTIONS = [
    "先查一下哈尔滨现在的天气，如果气温低于10度，就再帮我查一下三亚的天气，我要决定去哪避寒；"
    "如果哈尔滨不冷就不用查三亚了。",
    "北京、上海、宁德这三个城市今天天气怎么样？",
]


def main():
    parser = argparse.ArgumentParser(description="第十一周作业：天气查询工具调用改造为循环调用")
    parser.add_argument("--question", "-q", help="单个问题")
    args = parser.parse_args()

    client, model = build_client()
    questions = [args.question] if args.question else DEMO_QUESTIONS

    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"Q{i}：{q}")
        print("=" * 60)
        result = run_loop(client, model, q)
        print(f"\n共循环 {result['rounds']} 轮，累计工具调用 {len(result['tool_calls'])} 次")
        print("最终回答：")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
