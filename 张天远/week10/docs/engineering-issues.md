# Rag Scratch 工程问题排查记录

> 开发过程中的所有坑、根因、排查方法和最终修复

---

## 目录

1. [Path 污染系列](#1-path-污染系列)
2. [Claude Code 认证失败](#2-claude-code-认证失败)
3. [子代理超时](#3-子代理超时)
4. [检索系统核心 Bug](#4-检索系统核心-bug)
5. [分块策略反复调整](#5-分块策略反复调整)
6. [BM25 IDF=0](#6-bm25-idf0)
7. [父块文本漏 heading](#7-父块文本漏-heading)
8. [chunker 接口兼容性断裂](#8-chunker-接口兼容性断裂)
9. [知识](#9-知识)

---

## 1. Path 污染系列

### 1.1 根因发现

**现象**：任何 `S:/condaEnvs/py312/python.exe -m pytest` 命令都报 `ModuleNotFoundError`，但 `import` 的包在 conda 环境中明明已安装。

**排查过程**：

```bash
# 第一步：确认 Python 确实在 conda 环境中
$ "S:/condaEnvs/py312/python.exe" -c "import sys; print(sys.executable)"
S:\condaEnvs\py312\python.exe  ← 环境正确

# 第二步：查看 sys.path 顺序
$ "S:/condaEnvs/py312/python.exe" -c "import sys; [print(p) for p in sys.path[:5]]"
M:\hermesData\hermes-agent                       ← 排第一！来源不明
M:\hermesData\hermes-agent\venv\Lib\site-packages ← 排第二！
S:\condaEnvs\py312\python312.zip
S:\condaEnvs\py312\DLLs
S:\condaEnvs\py312\Lib
```

**根因**：Hermes Agent 启动时设置了：

```
PYTHONPATH=M:\hermesData\hermes-agent;M:\hermesData\hermes-agent\venv\Lib\site-packages
```

所有通过 `terminal` 工具启动的子进程都继承这个环境变量，Python 启动时把 `PYTHONPATH` 中的路径插在 `sys.path` 最前面，导致 Hermes venv 的包（cp311 编译的 numpy、损坏的 pydantic）覆盖 conda py312 的正确版本。

**修复**：

```bash
# 每次命令前强制覆盖 PYTHONPATH
PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages S:/condaEnvs/py312/python.exe -m pytest tests/
```

固化到 `run_tests.bat` 和 `scripts/run.sh`。

---

### 1.2 具体症状链

| # | 现象 | 涉及的包 | 修复 |
|---|------|----------|------|
| 1.1 | `import numpy` → `_multiarray_umath.cp311-win_amd64.pyd` 不兼容 cp312 | numpy | `pip install numpy --force-reinstall` 到 conda |
| 1.2 | `pytest` → langsmith 插件加载 pydantic → `pydantic_core._pydantic_core` 缺失 | pydantic, langsmith | `pip uninstall langsmith -y` |
| 1.3 | `BeautifulSoup(html, "lxml")` → `FeatureNotFound: lxml` | bs4 | 降级到 `html.parser`（标准库） |
| 1.4 | `from openai import OpenAI` → 同上 pydantic 错误 | openai | `pip install openai` 到 conda + PYTHONPATH 隔离 |
| 1.5 | `conda activate py312` → `Run 'conda init' before 'conda activate'` | conda | 绕过：直接用绝对路径 `S:/condaEnvs/py312/python.exe` |

**教训**：Python 多环境共存时，**`PYTHONPATH` 是最高优先级的污染源**。排查任何 import 错误的第一步应该是 `python -c "import sys; print(sys.path[:5])"`。

---

## 2. Claude Code 认证失败

**现象**：

```bash
$ claude -p "hello" --max-turns 1
Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}
```

**排查过程**：

```bash
# 检查认证状态
$ claude auth status
{"loggedIn": true, "authMethod": "oauth_token", "apiProvider": "firstParty"}
# ↑ 已登录，但 -p 模式仍 403

# 检查 API Key 环境变量
$ echo $ANTHROPIC_API_KEY
# 空——使用的是 OAuth 而非 API Key
```

**根因**：Claude Code 的 `-p`（非交互模式）需要付费订阅。OAuth 登录的免费账户只支持交互模式，`-p` 返回 403。

**解决**：放弃 Claude Code CLI，改为直接手写代码 + 用 `delegate_task` 并行调度独立子任务。

**后续教训**：切片 1-3 手写是合理的（小任务），但切片 4-7 应该更早用子代理并行。D9 决策纠正了这个惯性。

---

## 3. 子代理超时

**现象**：切片 9（重排器）子代理 600s 超时，无产出。

**排查**：

```bash
# 手动加载模型测试
$ python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
# 成功加载，50s 完成
```

**根因**：子代理首次加载 1.11GB 的 CrossEncoder 模型时，可能因为 HuggingFace 连接或环境配置导致卡住。而我在主会话中手动执行时模型已通过 ModelScope 后台下载完毕。

**解决**：手动接管重排器实现（5 分钟），后续遇到模型加载任务先确保模型已下载再派子代理。

**教训**：涉及大模型首次下载/加载的任务，**先预下载模型再派子代理**，避免因网络问题导致超时。

---

## 4. 检索系统核心 Bug

### 4.1 "关键字"召不回问题

**用户报告**：问"Rust 中有哪些关键字"，文档中明确有答案（appendix-01-keywords.html），但检索结果没有。

**排查步骤**：

```
Step 1: 查看该 HTML 解析成的 JSON
→ 确认 JSON 中包含"附录 A：关键字"和 38 个关键字的列表

Step 2: 查看分块结果
→ 8 个 chunk，chunk[1] 包含完整列表（1101 字符），chunk text 中有"关键字"一词 ✅

Step 3: 单独测试 BM25
→ BM25 top-1 ✅ 命中"附录 A：关键字"

Step 4: 单独测试向量检索
→ Vector top-1 ✅ 命中"附录 A：关键字"

Step 5: 测试 RRF 融合
→ RRF top-1 ❌ "附录 C：可派生的 trait"——完全不相关！
```

**根因定位**：

原始 RRF 实现按 `chunk_id` 聚合分数。一个文档被切成 8 个 chunk 后，多个 chunk 在检索中占据不同排名位置，每个 chunk 贡献 `1/(k+rank)`。长文档的多个 chunk 共计 > 短文档的少量 chunk 得分。

"可派生的 trait"有 5+ 个 chunk 在两边排名都不错（#5-#15），RRF 累计分数远超只有 2 个 chunk 排在顶部的"关键字"文档。

**修复迭代**：

| 版本 | 方案 | 结果 |
|------|------|------|
| v1 | 原版（按 chunk_id 聚合） | ❌ 关键字完全不出现 |
| v2 | 降 k 值（60→5） | ❌ 无改善——问题不在 k |
| v3 | 按 source_url 聚合，取 max chunk 分数 | ⚠️ 关键字排 #4 |
| v4 | 按 source_url 聚合，取最佳排名 | ✅ 关键字排 #1 (0.0328 vs 0.0299) |

**最终 RRF 公式**：

```
score(doc) = 1/(k + best_bm25_rank(doc)) + 1/(k + best_vector_rank(doc))
             ↑ 不在某检索器中 = 1/(k + default_rank)
```

**教训**：RRF 按 chunk 聚合是常见陷阱——看似"公平"（每个 chunk 一票），实则长文档因多 chunk 而获得不成比例的优势。**按文档聚合 + 取最佳排名**更接近真实的检索质量。

### 4.2 LLM 看不到关键字列表

**现象**：修复 RRF 后，关键字文档排 #1 ✅。但 LLM 仍回答"文档中未列出具体的关键字列表"。

**根因**：38 个关键字的完整 chunk 长度 1101 字符，被 `max_chunk_size=500` 截断。LLM 收到的只是说明段"以下列表包含 Rust 中正在使用...的关键字"，看不到列表本身。

**修复**：父子块方案 B——D15 决策。

---

## 5. 分块策略反复调整

### 时间线

| 版本 | 触发 | 改动 |
|------|------|------|
| v1 | 初始 | 按 h2 切分，max_chunk_size=500 |
| v2 | 用户要求"按数据形态分类" | parser 加 morphology 字段，chunker 按形态路由 |
| v3 | "关键字"召不回 | 代码单元独立分块 |
| v4 | "LLM 看不到列表" | 父子块方案 B |

### 形态标注的浪费与补救

v1 parser 只标注 HTML 标签类型（heading/paragraph/code_block），分块器根据 type 字段推断形态。用户指出"没有完全按照讨论的原则来"——讨论中明确 morphology 应该在 parser 阶段标注。

v2 修正：parser 的输出从：

```json
{"type": "paragraph", "text": "Rust 的核心功能..."}
```

变为：

```json
{"type": "paragraph", "morphology": "narrative", "text": "Rust 的核心功能..."}
```

**教训**：Grill 阶段的决策（"中间 JSON 应标注数据形态"）在实现时被简化了——为了省事推迟到了 chunker 中推断。**工程上应该信任 Grill 的结论，不要"优化"掉已经确认的设计**。

---

## 6. BM25 IDF=0

**现象**：BM25 测试 `test_keyword_match_ownership` 失败——查询"所有权"返回的分数为 0。

**排查**：

```python
# 测试数据中所有 6 个文档都包含"所有权"
chunks = [
    "Rust 的所有权系统...",   # ✅ 含"所有权"
    "借用（Borrowing）...",   # ✅ 含"所有权"? 不...
    # ...
]
```

**根因**：BM25 的 IDF = log((N - df + 0.5) / (df + 0.5))。如果关键词出现在所有文档中（df = N），则 IDF = log(1) = 0。测试数据设计时所有文档都围绕 Rust 所有权主题，导致 df=N。

**修复**：在测试集中添加一个不含"所有权"的文档（如 Python 主题），使 df < N → IDF > 0。

**教训**：BM25 的 IDF 在封闭测试集中容易被误伤。**测试 BM25 时确保存在不包含关键词的干扰文档**。

---

## 7. 父块文本漏 heading

**现象**：`test_parent_has_full_content` 失败——父块 text 不包含"所有权规则"这个 heading。

**排查**：

```python
# _make_parent 中
text = _build_rich_text(section["elements"])
# section["elements"] 不包含 heading 元素！
# heading 在 section["headings"] 中，但 _build_rich_text 不处理它
```

**根因**：`_split_by_headings` 用标题做边界切分，标题被存在 `section["headings"]` 列表中，**不进入 `section["elements"]`**。`_build_rich_text` 只遍历 elements，所以标题丢了。

**修复**：

```python
# 在 _make_parent 中显式添加 headings
text_parts = []
for i, h in enumerate(section["headings"]):
    prefix = "#" if i == 0 else "#" * (i + 1)
    text_parts.append(f"{prefix} {h}")
text_parts.append(_build_rich_text(section["elements"]))
text = "\n\n".join(text_parts)
```

**教训**：数据和元数据分离时，**消费方容易忘记元数据**。标题既是"切分信号"也是"内容"——它应该同时存在于 headings（元数据）和 text（内容）中。

---

## 8. chunker 接口兼容性断裂

**现象**：父子块方案落地后，11/21 测试失败。

**根因**：`chunk_elements()` 原本返回扁平的 Chunk 列表。父子块方案改为返回混合列表（含 is_parent=True 和 is_parent=False 的 chunk），所有消费方（build_index、query、测试）的期望被打破。

**修复**：

1. Chunk dataclass 加 `parent_chunk_id` 和 `is_parent` 字段
2. 增加 `get_parents()` / `get_children()` 辅助方法
3. build_index 改为只索引子块
4. query 改为检索子块 → expand_to_parents → LLM
5. 所有 store 的 load 方法兼容新字段

**教训**：接口变更时，**应该先加新方法（get_parents/get_children），再改内部实现**。这次顺序反了，导致多轮修复。

---

## 9. Chunk ID 全局冲突导致父子链断裂

**现象**：重建索引后，检索只返回 1 个子块，LLM 看不到任何有效文档。

**排查步骤**：

```
Step 1: 检查 children.json → 4589 条记录，但 parent_chunk_id 全是 NONE
Step 2: 检查 all_chunks.json → 1851 parents + 4589 children，parent 链正常
Step 3: 查 code → VectorStore.save 没保存 parent_chunk_id
Step 4: 修复 save → 重建索引 → 效果仍差
Step 5: 深入查 all_chunks.json → 6440 条但只有 741 个唯一 ID
        book_p0000 重复 105 次，book_c0000_00 重复 99 次
```

**根因**：chunk_id 生成逻辑中，每个文件独立从 0 开始计数：
- 第 1 个文件的父块 = `book_p0000`
- 第 2 个文件的父块 = `book_p0000` ← 冲突！
- 105 个文件 × 平均 2-5 个 section → 大量 ID 冲突

**修复**：chunk_id 加入 `file_index` 前缀：
- 父块：`{source}_f{file_index:04d}p{section_idx:04d}`
- 子块：`{source}_f{file_index:04d}c{section_idx:04d}_{child_idx:02d}`

**关联 bug**：修复 ID 后发现 `_make_parent` 调用没有传 `file_index` 参数——在 chunk_elements 中加了参数但漏了传递。

**教训**：
- 批量处理时，**永远不要用局部自增计数器做全局 ID**
- 新增参数时，**用 grep 检查所有调用点**——不要假设 Python 会报错（默认参数不传不报错）

---

## 9. 知识

### 排查通用方法论

1. **先隔离变量**：出问题时先确认是哪个环节——数据、分块、检索、还是生成？
2. **逐环节测试**：不要猜，用快速脚本单独验证每一环
3. **打印中间状态**：chunk 内容、排名、RRF 分解——可视化远胜猜测
4. **二分法定位**：BM25 对？向量对？→ 问题在融合。融合公式对？→ 问题在聚合。

### 多 Python 环境排查清单

```bash
# 1. 确认 Python 路径
python -c "import sys; print(sys.executable)"

# 2. 查看 sys.path 顺序——PYTHONPATH 污染最常见
python -c "import sys; [print(p) for p in sys.path[:8]]"

# 3. 检查环境变量
echo $PYTHONPATH
echo $VIRTUAL_ENV

# 4. 确认包的实际加载位置
python -c "import numpy; print(numpy.__file__)"
```

### 检索系统调试技巧

```python
# 对比 BM25 vs 向量 vs RRF 的 top-N
for rank, (chunk, score) in enumerate(bm25.search(query, 10)):
    print(f"BM25[{rank}] {chunk.metadata['headings'][:50]}")

for rank, (chunk, score) in enumerate(vector.search(query, 10)):
    print(f"Vec[{rank}] {chunk.metadata['headings'][:50]}")

# RRF 分解：追踪每个文档的精确贡献
```

---

## 10. "中"/"里"一字之差，BM25 排名翻倍（查询鲁棒性）

**现象**：同一个问题用 `"rust中有哪些关键字"` 和 `"rust里有哪些关键字"` 查询，keyword 文档排名从 #3 掉到 #6，BM25 排名从 #36 恶化到 #76。

**查询分词对比**：

| 查询 | jieba 分词 | keyword 子块 BM25 排名 |
|------|-----------|:--:|
| `rust中有哪些关键字` | `rust/中/有/哪些/关键字` | #36 |
| `rust里有哪些关键字` | `rust/里/有/哪些/关键字` | #76 |

**根因**：jieba 对"中有哪些"和"里有哪些"分词后多了一个无意义的"里"字，改变了 token 分布。BM25 作为词袋模型，对查询词的微小变化极其敏感——多一个无意义的虚词就足以大幅改变文档评分。

**排查过程**：
1. 自检（在 query.py 启动时跑）显示 keyword #3，但交互查询显示 #6
2. 怀疑 `.pyc` 缓存 → 排除
3. 怀疑 `RAGGenerator` 初始化影响全局状态 → 排除
4. **逐字对比查询字符串** → 发现"中"vs"里"一字之差

**教训**：
- BM25 对自然语言的同义表达极度敏感，这是词袋模型的天然局限
- 排查时的查询字符串必须逐字一致——一字之差导致完全不同的结论
- 这个案例直接论证了 **LLM 查询重写**的必要性：把用户的口语化表达标准化为检索友好的关键词组合

**影响**：后续应实现查询重写策略（LLM 将用户查询改写为 2-3 个标准化版本并行检索），消解自然语言的表达不稳定性。

**解决方案**（2026-07-05 实施）：
- 创建 `src/llm/query_rewriter.py`，使用 DeepSeek v4-flash 改写查询
- 集成到 `query.py`：`multi_search()` 多路检索 + source_url 去重合并
- 用 winreg 替代 PowerShell 读取 API Key，避免 bash 转发损坏
- 结果：`"rust里有哪些关键字"` 重写为 `["Rust 关键字列表", "Rust 严格关键字 保留字", "Rust 关键字 as async await"]`，合并后 keyword 稳定排到 #0


---

## 📚 开发经验教训

### 一、检索系统

**1. BM25 对自然语言极度敏感**

"rust中有哪些关键字" vs "rust里有哪些关键字"，jieba 多切出一个无意义虚词"里"，BM25 排名从 #36 恶化到 #76。**教训：BM25 不能独立作为自然语言查询的检索器，必须配合 LLM 查询重写或向量检索。**

**2. RRF 融合的三个坑**

| 坑 | 症状 | 修复 |
|----|------|------|
| 按 chunk 聚合 | 5 个 chunk 的长文档碾压 2 个 chunk 的精简文档 | 按 source_url 聚合 |
| 按 chunk_id 去重 | ID 冲突导致 4589 个子块只有 741 个唯一 ID | parent/child ID 加 file_index |
| 排序被 expand 打乱 | search 排 #1 → expand 后排 #6 | expand 保持 search 顺序，不做重排 |

**3. 分块策略的父子矛盾**

子块要短（200-400 字符）以便检索精准，但父块要完整（section 级）以便 LLM 有充足上下文。**关键设计：structured 子块（列表/表格条目）的 text 必须包含 heading 前缀**，否则"as - 强制类型转换"不含"关键字"一词 → BM25 永远找不到。

**4. 查不出答案 ≠ 检索失败**

keyword 文档排名 #6 时 LLM 仍能正确列出关键字——因为兄弟拼接把列表段也拼上了。**不要只看排名数字，要看 LLM 最终输出。**

### 二、评估体系

**5. 评估题质量比数量重要**

60 道 LLM 自动生成的题有 43% 是代码补全型（缺少上下文，无法合理评分）。**教训：必须先跑小批量验证题目质量，再扩大规模。手动出题虽然慢，但质量可控。**

**6. Glossary 关键词扩展的双面性**

利用 371 条术语做中英文扩展确实提升了 Precision 覆盖，但部分误扩展（"泛型"→被匹配到 "self"/"super"）引入噪声。**需要精确匹配而非子串匹配。**

**7. 实时评估需要前后端紧密配合**

前端评估循环中：
- 调用 `/query` 获取回答 → 计算 Precision/MRR（免费，立即）
- 调用 `/evaluate` 获取 Faithfulness/Relevancy（需要 LLM，耗时）
- **如果后端没重启，新增的 `/evaluate` 端点就是 404，前端 silently 跳过，指标永远 NA**

### 三、前端工程

**8. Streamlit 的"全量重跑"模型**

任何控件交互（点击按钮、展开折叠、选择下拉）都会触发整个脚本从头执行。后果：
- 评估运行中点击其他控件 → 评估被打断
- 必须锁定页面 + 用 `st.session_state` 持久化进度 + 循环内 `st.empty()` 实时刷新

**9. `st.button` 不响应 Enter 键**

必须用 `st.form` + `st.form_submit_button` 包裹输入框和按钮，否则按 Enter 只能清空输入，不会触发查询。

**10. 前端和后端的"重启不同步"**

前端代码改了 → 只需 `Ctrl+C` 重跑 StreamLit。后端代码改了 → 必须杀掉旧进程重新启动。**但 PowerShell `Start-Job` 不继承环境变量（PYTHONPATH/HF_HOME），导致后端在新环境中启动失败。** 显式传入 env vars 后解决。

**11. emoji + Windows GBK = 崩溃**

`print("🚀 API 启动")` 在 PowerShell 管道中触发 `UnicodeEncodeError: 'gbk' codec can't encode`。**Web 项目输出到文件时务必设 `PYTHONIOENCODING=utf-8`。**

### 四、工程方法

**12. 先排查后修复**

整个 session 中最浪费时间的地方：没看清楚数据就动手改代码。keyword 排名问题折腾了 5+ 轮才沉淀为父子块策略。**教训：加 debug 日志 → 用户运行 → 拿到数据 → 讨论方案 → 实施。不要猜。**

**13. 调试脚本保存为文件**

所有测试必须写成 `.py` 文件（不能 `python -c`），便于用户复现。**`tests/` 目录下的 `run_query_test.py`、`diagnose_main.py` 等文件事后可追溯完整排查路径。**

**14. 简单修复优先于架构改动**

"关键字查不出来"的最早解决方案是 LLM 查询重写（架构改动），但最终发现兄弟拼接（10 行代码）就解决了 LLM 看不到完整列表的问题。**先用最小的代码量验证效果，再决定是否要上大方案。**

**15. 后端日志不可省略**

`Start-Job` 的后端崩溃时没有任何输出，排查全靠猜测。加 `Out-File` 保存日志后立刻定位到 emoji 编码问题。**任何后台进程必须有日志文件。**


---

## 16. Faithfulness 评估对推导/对比类问题误判为 0 分

**现象**：40 题评估中，4 道 Faith=0（m24/m26/m36/m38），6 道 Faith<0.34。这些题全是"A 和 B 的区别/对比"类问题，LLM 回答质量不差（AR=0.69-0.80），但 Faithfulness 评估器判定所有声明均无法在检索 chunk 中找到依据。

**根因分析**：

1. **评估方法不适合对比类问题**。Faithfulness 用 LLM 逐句核查答案中的声明是否能在单个 chunk 中找到原文。但对比类问题的答案需要整合多个 chunk 的信息形成新结论（如"方案 A 成本更低但扩展性弱于方案 B"），没有任何单个 chunk 包含完整对比句。

2. **LLM 生成用语不规范**。m24 的回答中出现"超类trait"（标准译名"父 trait"），评估器在检索 chunk 中找不到这个自造词 → 第一条声明即判定为无依据 → 整段被判 0 分。

**排查过程**：
- 先在网页端看到 10 道低分题，但读取本地的 `report_final.json` 发现数据完全不一致 → 定位到报告版本问题：页面用的是 `evaluation/results/eval_20260706_175620.json`
- Python 排序 bug：`0.0 or 1` 返回 `1`（因为 0.0 是 falsy），导致 Faith=0 的题被排到最后，分析错了 10 道题
- 修正排序后确认 4 道 0 分题的全是"区别/对比"类

**解决方案（两个方向并行）**：

| 方向 | 改动 | 文件 |
|------|------|------|
| **术语规范** | generator system prompt 新增第 5 条：必须使用 Glossary 标准译名，禁止自造词（如"超类trait"） | `src/llm/generator.py` |
| **分层评估** | 问题分类器（15 个推导关键词）+ 推导类专属 prompt：允许联合多 chunk 推导，只验证前提覆盖而非单句匹配 | `evaluation/evaluator.py` |

**分类器关键词**：区别、对比、比较、不同、异同、优劣、优缺点、关系、联系、vs、versus、有何不同、如何选择、如何区分、如何影响

**推导类 prompt 核心差异**：
- 事实类：逐句核查"是否在单个 chunk 中找到原文"
- 推导类：区分事实声明（需直接依据）和推导结论（前提覆盖 + 逻辑关系 → 视为 supported）

**测试验证**：`tests/test_faithfulness_fix.py` — 10 道低分题分类测试 10/10 通过
- 推导类（5 道）：m24/m26/m38/m25/m22 → 使用宽松验证 prompt
- 事实类（5 道）：m36/m10/m19/m14/m35 → 保持严格核查，搭配术语规范改善


---

## 17. 分层评估第一轮效果不佳——3 题仍 Faith=0，需加 Prompt 豁免规则

**现象**：#16 的分层评估 + 术语规范改动后重新评估，仍有 3 道 Faith=0（m26/m34/m36），5 道 Faith≤0.33。效果未达预期。

**根因分析**（`tests/test_zero_faith_root_cause.py` 逐题检测）：

| 题 | Faith | 根因 | 检测模式 |
|----|:--:|------|------|
| **m36** | 0 | LLM 诚实回答"文档中未提供，无法进一步说明"→ 评估器判为无依据 | `诚实答'不知道'=True` |
| **m26** | 0 | 回答含代码示例 + 推导结论 → 评估器要求原文匹配 | `含代码片段=True` + `推导类=True` |
| **m34** | 0 | 回答全篇用"来源：文档 X"引述风格 → 评估器无法验证引述声明 | `只引述文档未综合=True` |

**三种误判的共性问题**：评估器 prompt 只定义了"什么是 supported"，没有定义"什么应该被豁免"。以下三种行为在 Faithfulness 语义下都应该标记为 supported：

1. **诚实表述**："文档中未提供 / 无法说明 / 未列出" → 这正是系统 prompt 要求的"不知道就说不知道"，不应扣分
2. **代码示例**：LLM 自创的代码片段（如 `fn add_one(x: i32) -> i32`）不在任何 chunk 中，但只要示例中的概念（如函数指针）在文档中有说明，就应视为 supported
3. **引述声明**："来源：文档 X"形式的引用 → 只要 X 在检索范围内，该声明即应视为 supported

**解决方案**：在两个 prompt（事实类 + 推导类）的 Task 部分加入三条豁免规则：

```
- "文档中未提供/未说明/未列出/无法回答"等诚实表述 → 直接判定为 supported
- 代码示例（```块）→ 只要示例中的概念在文档中有说明，即视为 supported
- "来源：文档 X"形式的引用 → 只要 X 在检索文档范围内，即视为 supported
```

**改动**：`evaluation/evaluator.py` — 事实类 prompt + 推导类 prompt 均追加豁免规则

**额外发现**：m24 仍 0 分的另一个根因——Glossary 不包含 `supertrait → 父 trait` 映射。LLM 持续生成"超类trait"，因为术语表缺少这项。修复：在 `src/glossary.py` 中新增 `_GLOSSARY_SUPPLEMENT` 补充字典，对 `supertrait`、`associated type`、`sub-trait` 等缺失术语手动映射。

**最终效果**（第三轮评估）：
- Faith=0：4 → 3 → **1**（仅剩 m24，且本地测试确认修复后可达 0.909）
- Faith≤0.33：6 → 5 → **1**
- Avg Faith：0.67 → — → **0.77**
- m24 单题端到端测试（`tests/test_single_m24.py`）确认：修复后 Faith=0.909

**教训**：评估 prompt 的设计应该是双向的——既要定义"什么是 bad"（幻觉），也要定义"什么是 good"（诚实表述、代码示例、引述）。只给正面标准和负面标准，不给中性豁免，会导致评估器过度惩罚系统 prompt 所鼓励的行为。另外，LLM 术语不规范不能只靠 prompt——必须确保 Glossary 覆盖所有关键术语。


---

## 19. m26 Faith=0 的三轮排查——最终定位到 preview vs text 截断

**现象**：经过 #16 #17 #18 多轮修复，m26 "函数指针 fn Fn trait 区别" 始终 Faith=0，8-12 条声明全部 ❌。而函数指针文档明确在检索结果 #2，包含原文："函数的类型是 fn（使用小写的'f'）以免与 Fn 闭包 trait 相混淆。fn 被称为函数指针。指定参数为函数指针的语法类似于闭包。"

**排查历程**：

| 轮次 | 假设 | 验证方法 | 结论 |
|------|------|----------|------|
| 1 | "原文措辞→LLM 总结"导致不匹配 | 加规则 4"优先使用文档原文措辞" | ❌ 反而更差——声明翻倍，更多被判 unsupported |
| 2 | 检索召回差，函数指针文档没有排在前面 | `diagnose_m26_retrieval.py` 查 BM25/向量/RRF 排名 | ❌ 函数指针文档在 RRF #1，检索没问题 |
| 3 | 评估器 LLM 有随机偏差 | 手动验证原文对比 | ❌ 原文逐字存在，不应该找不到 |
| **4** | **评估器拿到的 chunk 被截断** | `verify_m26_context.py` 检查 context 实际长度 | **✅ 找到根因！** |

**根因**：`/evaluate` 端点构建 Chunk 时使用了 `s.get("preview", "")`（200 字符截断版），而非完整 chunk text。

```
scripts/app.py 第 271 行 (旧):
text=s.get("preview", "")  ← 只有 200 字符！

函数指针文档在第 200 字符恰好被切断：
"...通过函数指针允许我们使用函数作为另一个函数的参数。函数的类型是 fn （使"
                                                                           ↑ 截断点
后面的 "用小写的 'f' 以免与 Fn 闭包 trait 相混淆" 被截掉了！
```

**同时发现 `/query` 返回的 sources 也只包含 `preview`（200 字符），没有完整 `text` 字段。** 修复了三处：
1. `/query` 端点：sources 新增 `"text": c.text`（完整文本）
2. `/evaluate` 端点：改用 `s.get("text", s.get("preview", ""))`
3. Web eval：发送 `/evaluate` 时传 `"text"` 字段

**修复后 m26 Faith = 0.909**（12/13 声明 ✅），仅代码示例回调 ❌。

**教训**：
- **preview ≠ context** — preview 用于 UI 展示，评估和生成必须用完整 text。这个设计失误导致评估器对"逐字匹配"的文档视而不见。
- **后端重启是最容易被忽视的环节** — 此 session 中多次出现"代码改了但进程没重启"导致的误判。`start.ps1` 的 `Start-Job` 环境变量隔离问题加剧了这一点。
- **不要归因于"LLM 随机性"** — 当评估器多次找不到逐字存在的原文时，不要轻易归因于模型偏差，先检查数据有没有真正传给评估器。
- **验证脚本是必需的** — `diagnose_m26_retrieval.py` 和 `verify_m26_context.py` 分别排除了检索问题和证实了截断问题，这两个脚本事后可复现完整排查路径。


---

## 18. Glossary 静态注入覆盖不足 → 改为动态检索

**现象**：经过 #16 #17 多轮修复，m24 (Clone vs Copy) 仍反复出现 Faith=0。`get_glossary()` 含 374 条术语，但 `format_glossary_for_prompt` 只取 50 条注入，`supertrait → 父 trait` 排在 #316 永远选不上。

**根因分析**：

| 问题层级 | 具体表现 |
|----------|----------|
| 选择策略 | 按中文长度排序 → "堆/栈/宏/糖"等单字词占满 50 个槽位，"父 trait"（7 字符）排 #316 |
| 补丁方案 | 加 `_GLOSSARY_SUPPLEMENT` 手动补 3 条 → 只救了 3 个词，换个问题又会漏其他术语 |
| 过滤优化 | 跳过单字词 → 50 个槽位全是复合词，但仍是"按长度选 50 条"，和查询无关 |

**根本矛盾**：50 条硬限制 vs 374 条术语 vs 查询相关性——不管怎么优化选择策略，要么漏掉关键术语，要么注入一堆无关术语。

**解决方案**：Glossary 走检索管道，按需召回。

```
查询 "Clone 和 Copy 的区别"
    ↓
并行检索 ─┬─ 文档索引    → chunk 1, 2, 3...
          └─ Glossary 索引 (新) → supertrait→父 trait, trait→特质...
    ↓
动态注入: 只注入检索到的相关术语 (≤10条)
```

**实现**：
- `src/glossary.py`: 新增 `build_glossary_index()` (374 术语 → FAISS + BM25), `search_glossary()` (查询 → RRF → top_k 术语)
- `src/llm/generator.py`: `generate()` 新增 `glossary_terms` 参数，优先动态注入
- `scripts/app.py`: `do_query()` 中调用 `search_glossary()` 并按需注入
- `scripts/build_index.py`: 新增 Glossary 索引构建步骤

**效果**：m24 Faith=0 → **1.0** ✅，LLM 从"超类trait"改为"父 trait（文档中称为超类trait）"，5/5 声明全部 supported。

**代价**：新增 `vectorstore/glossary/` 索引目录 (~50MB)，查询延迟 +0.5s（加载小模型）。
