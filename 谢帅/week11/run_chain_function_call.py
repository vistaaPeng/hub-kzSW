"""
run_chain_function_call.py — 链式（多轮）Function Call 演示

改造目标：
  原 run_function_call.py 只跑"单轮闭环"——模型一次性输出所有 tool_call，
  宿主执行后回填，模型立刻产出最终回答。这种结构下，一个工具的结果
  无法成为另一个工具的输入。

  本文件把它改成"多轮闭环"（while 循环）：
    第 1 轮：模型调 get_coordinates("北京") → 宿主回填经纬度
    第 2 轮：模型看到经纬度，用它调 get_weather(lat, lon) → 宿主回填天气
    第 3 轮：模型看到天气，产出最终回答（不再调用工具）→ 跳出循环
  ——前一个工具的结果，成为下一次调用的输入，多轮自动串成"调用链"。

使用方式：
  # 配置环境变量（默认用 DeepSeek）
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx

  python run_chain_function_call.py --demo                 # 查北京/上海/天津
  python run_chain_function_call.py -q "天津现在天气怎么样？"
  python run_chain_function_call.py --provider dashscope   # 切到 qwen-plus

依赖：
  pip install openai httpx
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI

from weather_backend import get_coordinates, get_weather

# 强制标准输出/错误用 UTF-8，避免中文 Windows 控制台（默认 GBK 码页）在遇到
# emoji 等星光平面字符时编码失败、并啃坏相邻中文（如 上海→上海海）。
# 不依赖 chcp / PYTHONIOENCODING，脚本自带兜底。VS Code 终端按 UTF-8 解码即正常显示。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 工具 Schema：两个单一职责的工具 ─────────────────────────────────────────
# 拆分后的关键：get_coordinates 的输出（经纬度）正是 get_weather 的输入。
# 模型据此把两次调用串成链——这就是"链式调用"在 schema 层面的体现。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": (
                "查询城市的经纬度坐标。返回 JSON，含 latitude/longitude。"
                "查天气前必须先用本工具拿到经纬度。城市用中文名，如 '北京'、'上海'、'天津'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "根据经纬度查询当前天气及未来3天预报。"
                "latitude/longitude 必须来自 get_coordinates 的返回结果，不要自己编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，来自 get_coordinates"},
                    "longitude": {"type": "number", "description": "经度，来自 get_coordinates"},
                    "name": {"type": "string", "description": "可选，地点显示名，用于报告标题"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# 工具名 → 后端函数的 dispatch 表
TOOL_DISPATCH = {
    "get_coordinates": get_coordinates,
    "get_weather": get_weather,
}


# ── 多轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气助手。查询任意城市的天气时，必须分两步链式调用工具："
    "第一步调用 get_coordinates 得到该城市的经纬度；"
    "第二步把上一步返回的 latitude/longitude 作为参数调用 get_weather 得到天气。"
    "只依据工具返回的数据作答，不要编造经纬度或天气。"
)


def run(client, model: str, question: str, verbose: bool = True, max_rounds: int = 6) -> dict:
    """
    多轮闭环：反复"请求模型 → 执行工具 → 回填"，直到模型不再要求调用工具。
    前一个工具的结果会成为下一轮某个工具的输入，从而自动串成调用链。

    返回 {answer, tool_calls, rounds, elapsed} 便于观察调用链。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型不再要求调用工具 → 本轮即最终回答，跳出链条
        if not msg.tool_calls:
            break

        if verbose:
            print(f"  ── 第 {rounds} 轮：模型要求调用 {len(msg.tool_calls)} 个工具")

        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"round": rounds, "name": name, "args": args})
            if verbose:
                print(f"     → [tool] {name}({args})")
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
                print(f"       ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            # 工具结果以 role=tool 回填，供模型在下一轮读取——这是"链"的接缝
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        # 循环正常结束（用尽 max_rounds 仍在要求调用工具），做一次收尾兜底
        if verbose:
            print(f"  ── 达到最大轮数 {max_rounds}，强制收尾")

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{rounds} 轮，{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "rounds": rounds, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "北京现在的经纬度和天气如何？",
    "上海现在的经纬度和天气如何？",
    "天津现在的经纬度和天气如何？",
]


def main():
    parser = argparse.ArgumentParser(description="链式（多轮）Function Call：先查经纬度，再查天气")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例（北京/上海/天津）")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    print(f"[链式 Function Call] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"Q{i}：{q}")
        print("=" * 60)
        result = run(client, model, q, verbose=not args.quiet)
        print("\n最终回答：")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
