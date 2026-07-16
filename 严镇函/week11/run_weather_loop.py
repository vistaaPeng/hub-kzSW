"""
run_weather_loop.py — 天气查询循环调用（多轮对话）

功能特性：
  1. 多轮对话支持：持续接收用户输入，保持对话历史
  2. 工具调用集成：自动调用天气查询工具获取实时数据
  3. 上下文理解：支持连续追问（如"北京天气如何？"→"那上海呢？"）
  4. 用户友好界面：支持退出、清空历史等命令

使用方式：
  # 启动交互式循环
  python run_weather_loop.py

  # 带初始问题启动
  python run_weather_loop.py --question "北京天气如何？"

依赖：
  pip install openai httpx
  环境变量：DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY
"""

import json
import os
import sys
import time

from openai import OpenAI

from weather_backend import get_weather

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        # "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "api_key": "sk-deb5c4a2c7464200a3802fdad8c7175b",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": "sk-deb5c4a2c7464200a3802fdad8c7175b",
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


# ── 工具 Schema ────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '北京'、'上海'、'宁德'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '北京'"},
                },
                "required": ["city"],
            },
        },
    },
]

# ── 工具分发 ───────────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "get_weather": get_weather,
}


# ── 系统提示 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名专业的天气查询助手。用户询问天气相关问题时，必须调用 get_weather 工具获取实时数据，"
    "只依据工具返回的结果回答，不要编造信息。如果用户的问题不涉及天气，"
    "请礼貌地告知用户你只能回答天气相关问题。"
)


# ── 单轮工具调用 ───────────────────────────────────────────────────────────

def process_single_turn(client, model: str, messages: list) -> tuple[str, list]:
    """
    处理单轮对话：发送消息给模型，处理工具调用，返回回答和更新后的消息列表
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        messages: 当前对话消息历史列表
        
    Returns:
        tuple: (answer, updated_messages)
    """
    t0 = time.time()
    
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
            print(f"  → [工具调用] {name}({args})")
            
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

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or "暂无回答"
    elapsed = time.time() - t0
    print(f"  → [回答]（耗时 {elapsed:.1f}s）")
    
    messages.append({"role": "assistant", "content": answer})
    
    return answer, messages


# ── 主循环 ─────────────────────────────────────────────────────────────────

def main_loop(client, model: str, initial_question: str = None):
    """
    启动交互式多轮对话循环
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        initial_question: 可选的初始问题
    """
    print("=" * 60)
    print(f"    天气查询助手（{model}）")
    print("=" * 60)
    print("输入问题查询天气")
    print("命令：exit/quit（退出） | clear（清空历史）")
    print("=" * 60 + "\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if initial_question:
        print(f"用户：{initial_question}")
        messages.append({"role": "user", "content": initial_question})
        answer, messages = process_single_turn(client, model, messages)
        print(f"助手：{answer}\n")

    while True:
        try:
            user_input = input("用户：").strip()
        except EOFError:
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("对话历史已清空\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            answer, messages = process_single_turn(client, model, messages)
            print(f"助手：{answer}\n")
        except Exception as e:
            print(f"错误：{e}\n")
            messages.pop()


# ── 入口 ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="天气查询循环调用（多轮对话）")
    parser.add_argument("--question", "-q", help="初始问题")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    args = parser.parse_args()

    client, model = build_client(args.provider)
    main_loop(client, model, args.question)


if __name__ == "__main__":
    main()