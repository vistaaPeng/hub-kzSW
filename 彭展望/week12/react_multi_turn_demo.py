"""
第十二周作业：为 agent 流程增加多轮对话能力

课堂代码（../week12 agent/react_financial_agent/src/react_manual.py 和
react_function_calling.py）里的 run() 每次调用都从零构建 messages（只有
system prompt + 当次问题），完全不记得上一轮聊了什么。用户追问"那五粮液呢？"
这种依赖上文指代的问题，模型看不到上一轮问的是"茅台的毛利率"，答不上来。

本作业给 ReAct 循环加一个 history 参数，实现多轮对话记忆。关键设计取舍：

跨轮历史只保留"精简 Q&A 对"（问题 + 最终回答），不带入某一轮完整的
Thought/Action/Observation 轨迹——一轮 ReAct 可能有 5~10 步工具调用，
如果原样带入下一轮，历史会指数级膨胀；而且旧轮次的工具调用细节对新问题
没有直接帮助，只会引入噪音干扰推理。所以每轮结束后只提炼出
"user: 问题 / assistant: 最终回答" 这一对追加进跨轮历史，供下一轮组装
messages 时使用；同时只保留最近 MAX_HISTORY_TURNS 轮，防止长对话下历史
无限增长。

课堂代码本身未做任何改动，本文件复用其 SYSTEM_PROMPT / 正则解析 / LLM
客户端 / 工具集（TOOLS_MAP），只新写一个支持 history 的 run_multi_turn()，
并提供交互式 REPL 和一段自动跑的指代消解追问 demo。

运行：
  export DASHSCOPE_API_KEY=sk-xxx   # 已写进 ~/.zshrc，一般不用重复设置
  python react_multi_turn_demo.py           # 交互式多轮对话
  python react_multi_turn_demo.py --demo    # 跑内置的指代消解追问示例（非交互）
"""

import sys
import time
import json
import argparse
from pathlib import Path

# 复用课堂素材：ReAct 手写版的 SYSTEM_PROMPT / 正则解析 / LLM 客户端 / 工具集，
# 不重复实现，只新增多轮对话所需的 history 拼接逻辑
COURSE_SRC = Path(__file__).parent.parent / "week12 agent" / "react_financial_agent" / "src"
sys.path.insert(0, str(COURSE_SRC))

from react_manual import SYSTEM_PROMPT, _parse_step, client, MODEL, _c  # noqa: E402
from tools import TOOLS_MAP  # noqa: E402

MAX_HISTORY_TURNS = 6  # 只保留最近 N 轮问答，防止历史无限增长


def append_turn(history: list, question: str, answer: str) -> list:
    """把本轮的精简问答（不含 ReAct 轨迹）追加进跨轮历史，并裁剪长度"""
    new_history = list(history) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return new_history[-2 * MAX_HISTORY_TURNS:]


def run_multi_turn(question: str, history: list | None = None, max_steps: int = 8):
    """
    在课堂版 react_manual.run() 基础上加 history 拼接：
    messages = [system] + 跨轮精简历史 + [本轮问题]，
    本轮内部 Thought/Action/Observation 的循环逻辑与课堂版完全一致。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            yield {"step": step, "type": "final", "thought": parsed["thought"], "answer": parsed["answer"]}
            return
        if parsed["type"] == "unparseable":
            yield {"step": step, "type": "error", "observation": f"格式解析失败：{llm_output[:200]}"}
            return

        tool_name = parsed["action"]
        tool_args = parsed["action_input"]
        tool_fn = TOOLS_MAP.get(tool_name)
        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        yield {
            "step": step, "type": "action", "thought": parsed["thought"],
            "action": tool_name, "action_input": tool_args, "observation": str(observation),
        }

        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}\n"})

    yield {"step": max_steps + 1, "type": "max_steps", "answer": f"已达最大步数 {max_steps}，未能得出最终答案"}


def run_and_print(question: str, history: list | None, max_steps: int = 8) -> str | None:
    print(f"\n{'='*60}")
    print(f"你: {question}")
    print("=" * 60)
    start = time.time()
    answer = None

    for step_data in run_multi_turn(question, history=history, max_steps=max_steps):
        stype = step_data["type"]
        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))
        elif stype == "final":
            answer = step_data["answer"]
            elapsed = time.time() - start
            print(_c("final", f"\n✅ Agent: {answer}"))
            print(f"（共 {step_data['step']} 步，耗时 {elapsed:.1f}s）")
        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))

    return answer


DEMO_QUESTIONS = [
    "贵州茅台和五粮液的股票代码分别是多少？",
    "贵州茅台2023年的毛利率是多少？",
    "那五粮液呢？",          # 指代消解：必须记得上一轮问的是"2023年毛利率"
    "两者相差多少个百分点？",  # 必须记得前两轮问出的具体数字
]


def run_demo():
    """自动跑一段必须依赖多轮记忆才能正确回答的追问序列"""
    history: list = []
    for q in DEMO_QUESTIONS:
        answer = run_and_print(q, history)
        if answer:
            history = append_turn(history, q, answer)


def interactive():
    history: list = []
    print("进入多轮对话模式（输入 exit/quit 退出，new 清空历史）\n")
    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("再见！")
            return
        if question.lower() == "new":
            history = []
            print("已清空历史，开始新一轮对话。\n")
            continue
        answer = run_and_print(question, history)
        if answer:
            history = append_turn(history, question, answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第十二周作业：为 agent 流程增加多轮对话能力")
    parser.add_argument("--demo", action="store_true", help="跑内置的指代消解追问示例（非交互）")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        interactive()
