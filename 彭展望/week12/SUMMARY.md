# 第十二周作业 — 为 agent 流程增加多轮对话能力

> 作业要求：为 agent 流程增加多轮对话能力。
> 完整代码见同目录 [`react_multi_turn_demo.py`](./react_multi_turn_demo.py)。

---

## 一、课堂代码的问题

课堂素材 `../week12 agent/react_financial_agent/src/react_manual.py`（以及
`react_function_calling.py`）里的 `run(question, max_steps)` 每次调用都是从零
构建 `messages`：

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": question},
]
```

只有 system prompt + 当次问题，完全不带上一轮聊了什么。用户如果追问"那五粮液
呢？"这种依赖上文指代的问题，模型看不到上一轮问的是"贵州茅台2023年的毛利率"，
根本答不上来——每次提问都像是跟一个失忆的助手重新开始对话。

## 二、改造方式：只把"精简 Q&A 对"带入下一轮，不带完整 ReAct 轨迹

最直接的想法是"把上一轮的完整 messages 原样接到下一轮"，但这样不对：

1. 一轮 ReAct 可能有 5~10 步工具调用（Thought/Action/Observation），原样带入
   下一轮会让历史指数级膨胀，几轮后就会超长；
2. 旧轮次里"调用了哪个工具、参数是什么"这些细节对新问题没有帮助，反而是噪音，
   会干扰模型对新问题的推理。

所以改成：**每轮问答结束后，只把 `{"role":"user","content":问题}` +
`{"role":"assistant","content":最终回答}` 这一对追加进跨轮历史**，下一轮开始时
用 `[system] + 精简历史 + [本轮问题]` 重新组装 messages，本轮内部的 ReAct 循环
（Thought → Action → Observation）逻辑完全不变。同时只保留最近
`MAX_HISTORY_TURNS`（默认6）轮，防止长对话下历史无限增长。

核心差异：

```python
# 课堂代码：messages 只有 system + 当次问题，跨轮无记忆
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}]

# 本作业：加一段跨轮精简历史
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if history:
    messages.extend(history)          # 之前几轮的 [user问题, assistant最终回答]...
messages.append({"role": "user", "content": question})

# 本轮结束后，只把"问题+最终回答"这一对（不含中间轨迹）存回 history
history = append_turn(history, question, answer)
```

课堂代码本身**未做任何修改**，本作业复用其 `SYSTEM_PROMPT` / 正则解析
`_parse_step` / LLM 客户端 `client` / 工具集 `TOOLS_MAP`，只新写了带 `history`
参数的 `run_multi_turn()` 和历史管理函数 `append_turn()`。

## 三、验证效果（原始输出见 `run_output.log`）

用一段**必须靠多轮记忆才能答对**的追问序列验证：

```
Q1: 贵州茅台和五粮液的股票代码分别是多少？
Q2: 贵州茅台2023年的毛利率是多少？
Q3: 那五粮液呢？            ← 指代消解：没有历史记忆的话，模型不知道"那"在问什么
Q4: 两者相差多少个百分点？   ← 需要记住 Q2、Q3 里问出的具体数字
```

实际运行结果：

| 轮次 | Agent 的表现 |
|------|-------------|
| Q1 | 分别调用 `company_lookup` 查到 600519 / 000858 |
| Q2 | 查到贵州茅台 2023 年毛利率 **91.96%** |
| Q3 | **没有追问"你是想问哪家公司/哪一年"**，直接调用 `financial_indicator("000858")`，答出五粮液 2023 年毛利率 **75.79%**——说明模型从跨轮历史里正确取到了"公司=五粮液"（Q1 提过代码）和"指标=2023年毛利率"（Q2 的问法）两个隐含信息 |
| Q4 | 直接算 `91.96 - 75.79 = 16.17`，两个数字精确对应 Q2、Q3 各自答案里的数值，没有重新查询，证明这两个数字确实是从跨轮历史的 `assistant` 消息里读出来的，而不是模型瞎编 |

作为对照，如果去掉 `history` 参数（即退回课堂代码的单轮模式）单独问 Q3
"那五粮液呢？"，模型会因为完全没有上下文，只能反问"您想了解五粮液的什么信息？"
或给出宽泛的公司简介，而不会知道要查"毛利率"这个具体指标。

## 四、如何复现

```bash
cd week12-作业答案-zw
pip install -r requirements.txt
export DASHSCOPE_API_KEY=sk-xxx        # 已写进 ~/.zshrc，一般不用重复设置
python react_multi_turn_demo.py        # 交互式多轮对话，可自己连续追问
python react_multi_turn_demo.py --demo # 跑上面这段内置的指代消解追问示例
```

交互模式内置两个命令：`exit`/`quit` 退出，`new` 清空历史开始新一轮对话。

## 五、一点体会

一开始想的方案是"把上一轮的 messages 整个存下来接到下一轮"，实现起来最省事，
但仔细想了下会有两个问题：一是 ReAct 的中间轨迹本来就是"用完即扔"的思考过程，
不是对话内容本身，混进跨轮历史里，模型看到的更像是一堆调试日志而不是"之前聊了
什么"；二是财务问答场景下一轮经常有好几步工具调用，几轮下来 prompt 长度会失控。
只保留"问题+最终答案"这个精简版本后，Q3 那种指代消解的效果反而更稳定——因为
Final Answer 本身就是模型自己对上一轮做的"总结"，直接复用这个总结比让模型重新
从一堆 Observation 里翻找相关信息更可靠。这也解释了为什么大多数生产级对话系统
（包括打字机式的 ChatGPT 网页版）展示的"历史"其实都是问答对，而不是模型内部的
思维链或工具调用日志。
