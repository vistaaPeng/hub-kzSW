"""
run_cli.py — 方式三：CLI（命令行即工具）— 多轮对话模式，两种形态

教学重点：
  1. 形态 A（具名 run_cli）：LLM 调一个 run_cli(command, args) 工具，command 是白名单 enum，
     host 拼出子命令执行。安全可控，但每加一个命令要改代码
  2. 形态 B（通用 run_bash）：LLM 自己拼完整 shell 命令，host 在沙箱里执行。
     最灵活、最危险——教学重点是沙箱设计（白名单/黑名单/超时/工作目录锁定）
  3. 循环调用闭环：保持对话历史，多轮对话中模型可以反复调用工具

使用方式：
  # 交互式多轮对话（循环调用）
  python mode_cli/run_cli.py --mode named
  python mode_cli/run_cli.py --mode bash

  # 单个问题（兼容旧用法）
  python mode_cli/run_cli.py --mode named --question "宁德时代2023年营收和净利润？"
  python mode_cli/run_cli.py --mode bash --question "宁德时代2023年营收和净利润？"

依赖：
  pip install openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）
            DASHSCOPE_API_KEY（Embedding，fincli 内部用）
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CLI_DIR = Path(__file__).parent / "cli"
PY = sys.executable

_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

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


NAMED_COMMANDS = {
    "rag_search": {
        "argv": FINCLI_ARGV + ["search"],
        "arg_map": {
            "query": "--query",
            "stock_code": "--stock-code",
            "year": "--year",
            "top_k": "--top-k",
        },
    },
    "rag_list_companies": {
        "argv": FINCLI_ARGV + ["list-companies"],
        "arg_map": {},
    },
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}


def run_named(command: str, args: dict) -> str:
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)})"

    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])

    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\bdeltree\b",
    r"\bformat\b", r"\bmkfs\b", r"\bdd\b",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r"[>;]\s*(?:rm|del|format)\b",
    r"\bcurl\b.*\|\s*sh",
    r"\bwget\b.*\|\s*sh",
    r"\bsudo\b", r"\bchmod\b.*-R", r"\bchown\b.*-R",
    r"\bnc\b", r"\bnetcat\b",
    r"/etc/passwd", r"/etc/shadow",
    r"\bTaskkill\b", r"\bStop-Process\b",
]

ALLOWED_HEADS = {"fincli", "python", "python3", "py", "git", "ls", "dir", "cat", "echo", "type"}


def sandbox_check(command: str) -> str | None:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None


def run_bash(command: str) -> str:
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"

    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "可查 A 股年报（rag_search/list_companies）和天气（weather）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "rag_search（查年报，需 query+可选 stock_code/year/top_k）/"
                                       " rag_list_companies（列公司）/"
                                       " weather（查天气，需 city）",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。rag_search: {query, stock_code?, year?, top_k?}; weather: {city}",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱里执行一条 shell 命令并返回 stdout。"
                "可用工具 fincli（一条真实命令）："
                "fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3；"
                "fincli list-companies；"
                "fincli weather --city 宁德。"
                "危险命令（rm/del/format/sudo/curl|sh 等）会被拦截；只允许白名单可执行文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令字符串"},
                },
                "required": ["command"],
            },
        },
    },
]

MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}

SYSTEM_PROMPT_NAMED = (
    "你是一名金融分析助手。通过 run_cli 工具调用预批准命令查 A 股年报与天气。"
    "回答年报问题前必须先 run_cli(command='rag_search', args={...}) 检索原文，只依据返回段落作答，不要编造。"
    "知识库仅含：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/海康威视(002415)/中国平安(601318)，年份 2021-2023。"
    "rag_search 的 query 不要含公司名/年份（已由 stock_code/year 过滤），用简短术语如 '营收和净利润'。"
    "不在库内的公司请明确告知，不要臆测。每回合可一次调用多个工具。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名金融分析助手。通过 run_bash 工具在沙箱里执行 fincli 命令查 A 股年报与天气。"
    "查年报：fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3"
    "（query 不要含公司名/年份，用简短财务术语）。"
    "列公司：fincli list-companies。"
    "查天气：fincli weather --city 南京。"
    "回答必须依据命令返回的原文，不要编造。知识库仅含 5 家公司（茅台/五粮液/宁德时代/海康威视/中国平安），"
    "不在库内的明确告知。每回合可一次调用多个工具。"
)


def process_turn(client, model: str, messages: list, mode: str, verbose: bool = True) -> str:
    """处理单轮对话：调用模型，处理工具调用，返回最终回答。会修改 messages 列表。"""
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": sys_prompt})

    t0 = time.time()
    has_tool_call = True

    while has_tool_call:
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"  → [{mode}] {tc.function.name}({args})")
                try:
                    result = executor(args)
                except Exception as e:
                    result = f"[{mode}] 执行异常：{e}"
                preview = (result or "")[:120].replace("\n", " ")
                if verbose:
                    print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
        else:
            has_tool_call = False

    messages.append(msg)
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 回答（{elapsed:.1f}s）")
    return msg.content or ""


def run_single(client, model: str, question: str, mode: str, verbose: bool = True) -> dict:
    """单问题模式：创建新对话，返回结果。"""
    tools_schema, _ = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    answer = process_turn(client, model, messages, mode, verbose=verbose)
    tool_calls = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    return {"answer": answer, "tool_calls": tool_calls, "mode": mode}


def run_loop(client, model: str, mode: str, provider: str, verbose: bool = True):
    """循环调用模式：交互式多轮对话。"""
    print(f"[CLI/{mode}] provider={provider} model={model}")
    print("=" * 60)
    print("欢迎使用 A股年报智能问答助手（多轮对话模式）")
    print("输入问题进行查询，输入 'exit' 或 'quit' 退出")
    print("知识库：贵州茅台/五粮液/宁德时代/海康威视/中国平安（2021-2023年报）")
    print("=" * 60)
    print()

    tools_schema, _ = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH
    messages = [{"role": "system", "content": sys_prompt}]
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
        answer = process_turn(client, model, messages, mode, verbose=verbose)

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
    parser = argparse.ArgumentParser(description="方式三：CLI（多轮对话）")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[CLI/{args.mode}] provider={args.provider} model={model}\n", file=sys.stderr)

    if args.json:
        questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
        results = []
        for q in questions:
            result = run_single(client, model, q, args.mode, verbose=False)
            result["question"] = q
            results.append(result)
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
        return

    if args.question:
        print("=" * 60)
        print(f"问题：{args.question}")
        print("=" * 60)
        result = run_single(client, model, args.question, args.mode, verbose=not args.quiet)
        print("\n最终回答：")
        print(result["answer"])
        return

    if args.demo:
        for i, q in enumerate(DEMO_QUESTIONS, 1):
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
            result = run_single(client, model, q, args.mode, verbose=not args.quiet)
            print("\n最终回答：")
            print(result["answer"])
            print()
        return

    run_loop(client, model, args.mode, args.provider, verbose=not args.quiet)


if __name__ == "__main__":
    main()
