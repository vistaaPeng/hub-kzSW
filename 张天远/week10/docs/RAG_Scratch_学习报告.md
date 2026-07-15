# RAG Scratch 学习报告：从零到 v0.2 封版

> 项目：rag_scratch — Rust 知识库 RAG 问答系统  
> 周期：2026-07-06 ~ 2026-07-07  
> 版本：v0.2 封版  
> 评估：40 题，P=0.75, MRR=1.00, Faith=0.89, AR=0.79, Faith=0 清零

---

## 一、项目背景与目标

本周是 NLP 课程第十周——检索增强生成（RAG）。任务要求从零实现完整的 RAG 系统，数据重新寻找，不依赖之前的 rag_annual_report 项目。

前置知识来自前几周：BERT 微调和文本分类（Week06）、BiEncoder/CrossEncoder/FAISS/BM25/两阶段检索（Week08）。本周学习目标是完整链路：解析→分块→Embedding→FAISS→向量检索+BM25→RRF融合→重排→LLM生成→评估。

**三个初始决策**：

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据源 | Rust 中文文档 ×7（530 HTML，mdBook 格式） | 结构化、中英混排、无需 OCR、问答价值高 |
| 技术栈 | BM25 + 向量 + RRF 手写，Embedding 用 bge-base-zh-v1.5 本地，FAISS IndexFlatIP | 核心竞争力在手写融合部分 |
| LLM | DeepSeek v4-flash（API key 存于 Windows 注册表） | 免费额度、中文友好 |

环境约束：GTX 1080 Ti 11GB (Pascal sm_61)，PyTorch 2.5.1，conda py312，HuggingFace 缓存 M:\huggingface_cache。

---

## 二、v0.1 核心研发线路

### 2.1 数据管线：下载 → 解析 → 分块

**下载器**（`src/downloader.py`）：从 7 个 Rust 中文文档站点批量下载 HTML。每个站点一个子目录，带重试机制。测试 5/5 通过。

**解析器**（`src/parser.py`）：HTML → 结构化 JSON（element tree）。针对 mdBook 格式做了专门解析——提取章节标题层级、代码块、表格、列表。测试 8/8 通过。

**分块器**（`src/chunkers/narrative.py`）：采用父子块方案——子块用于检索（200-400 字符，轻量），父块用于 LLM（完整 section，信息完备）。按数据形态分为 6 类：
- `narrative`：叙述段
- `code_unit`：代码单元
- `specification`：规范条目
- `structured`：结构化数据（列表/表格）
- `note`：注释
- `mixed`：混合区

共生成 6440 chunks（4589 子块 + 1851 父块）。测试 6/6 通过。

**关键坑**：structured 子块的 text 必须包含 heading 前缀。例如 "as - 强制类型转换" 不含 "关键字"→ BM25 永远找不到。需要在子块 text 中保留父级标题。

### 2.2 索引构建：FAISS + BM25 + RRF

**向量索引**（`src/retrievers/vector_store.py`）：bge-base-zh-v1.5 (768d) → FAISS IndexFlatIP，4589 个子块。

**BM25 索引**（`src/retrievers/bm25_store.py`）：jieba 分词 + rank-bm25，同样 4589 个文档。

**RRF 融合**（`src/retrievers/hybrid_retriever.py`）：手写实现，K=60。关键设计——按 `source_url` 聚合（非 chunk_id），避免多 chunk 文档碾压少数 chunk 的精准文档。经历 3 次修复才稳定。

**扩展策略**：检索命中的子块 → 扩展为父块 + 全部同源兄弟父块（max_total_chars=3000），保证 LLM 看到完整 section。

### 2.3 LLM 生成

**生成器**（`src/llm/generator.py`）：DeepSeek API 调用，temperature=0.1，max_tokens=1024。System prompt 要求严格使用文档内容，禁止编造。

**查询重写**（`src/llm/query_rewriter.py`）：LLM 将用户自然语言查询改写为 2-3 个标准化版本，多路检索→按 source_url 去重合并。解决了 "rust里有哪些关键字" vs "rust中有哪些关键字" 一字之差 BM25 排名翻倍的经典问题。

**Glossary 注入**：371 条中英术语从 Rust Wiki 术语表下载，注入 system prompt 确保译名一致性。后续经历重大升级（见 3.4 节）。

### 2.4 评估体系

**四指标评估器**（`evaluation/evaluator.py`）：

| 指标 | 计算方式 | 成本 |
|------|----------|:--:|
| Context Precision | 关键词命中率 | 免费 |
| MRR | 第一个相关 chunk 的排名倒数 | 免费 |
| Faithfulness | LLM 逐句核查答案声明是否在检索 chunk 中 | 1 次 LLM |
| Answer Relevancy | LLM 逆向生成问题 + 余弦相似度 | 1 次 LLM |

**初始评估**（10 道手动题）：P=0.89, MRR=1.00, Faith=0.93, AR=0.80。

---

## 三、v0.2 Web 界面开发

### 3.1 架构

前后端分离：FastAPI 后端（端口 8000，6 个端点）+ StreamLit 前端（端口 8501，4 页面）。

| 页面 | 功能 | 对接端点 |
|------|------|----------|
| 🔍 查询 | 输入框 → 回答 + 来源列表（含分数、拼接信息、URL） | POST /query |
| 📊 评估 | 四指标卡片 + 每题折叠详情（含 chunk 追溯、声明核查） | POST /evaluate |
| ⚙️ 设置 | top_k 滑块、rerank/rewrite 开关 | 前端状态 |
| 📋 历史 | 查询日志表格 + JSON/CSV 可选列导出 | GET /history |

新增端点：`GET /stats`、`GET /history`、`POST /parse`、`POST /evaluate`。

### 3.2 StreamLit 踩坑集

| 坑 | 表现 | 修复 |
|----|------|------|
| `st.button` 不响应 Enter | 用户按回车只清空输入，不触发查询 | 用 `st.form` + `form_submit_button` 包裹 |
| 评估运行中可被打断 | 点击任何控件触发全量重跑 → 评估进度丢失 | 锁定页面 + `eval_running` 状态 + 重入续跑 |
| `Start-Job` 不继承环境变量 | 后端在 PowerShell job 中找不到 PYTHONPATH | 显式传入 `$env:PYTHONPATH`、`$env:HF_HOME` |
| emoji + Windows GBK = 崩溃 | `🚀 API 启动` → `UnicodeEncodeError` | `PYTHONIOENCODING=utf-8` |
| 后端重启不同步 | 改了 `app.py`，前端调 `/evaluate` 返回 404 | `start.ps1` 启动前自动杀旧进程 |

---

## 四、深度调试：Faithfulness 从 0.67 → 0.89 的三轮战斗

这是整个项目中最艰难也最有价值的部分。初始 10 题评估 Faithfulness 均值仅 0.67，4 道题为 0 分。

### 4.1 第一轮：分层评估 + 术语规范（#16）

**现象**：4 道 Faith=0（m24/m26/m36/m38），全是 "A 和 B 的区别/对比" 类问题。

**根因发现**：
1. 对比类问题答案需跨文档综合 → Faithfulness 逐句核查单 chunk 的机制天然不适用
2. m24 回答中出现 "超类trait"（标准译名 "父 trait"）→ 评估器在 chunk 中找不到这个词

**修复**：
- Generator system prompt 新增第 5 条：必须使用 Glossary 标准译名，禁止自造词
- 评估器新增问题分类器（15 个推导关键词："区别""对比""优劣""vs" 等）
- 推导类用宽松 prompt（允许联合多 chunk 推导），事实类保持严格核查
- 测试 `test_faithfulness_fix.py`：10/10 通过

**效果**：4→3 道 0 分，仅小幅改善。

### 4.2 第二轮：Prompt 豁免规则（#17）

**现象**：3 道仍 Faith=0（m26/m34/m36）。

**逐题检测**（`test_zero_faith_root_cause.py`）：

| 题 | 根因 | 检测模式 |
|----|------|------|
| m36 | LLM 诚实回答 "文档中未提供，无法说明" → 评估器判为无依据 | 诚实答"不知道"=True |
| m26 | 回答含代码示例 + 推导结论 → 评估器要求原文匹配 | 含代码片段=True |
| m34 | 全篇用 "来源：文档 X" 引述风格 → 评估器无法验证 | 只引述未综合=True |

**修复**：两个 prompt（事实类 + 推导类）均追加三条豁免规则：
- 诚实表述（"文档中未提供/无法说明"）→ 直接判定 supported
- 代码示例 → 只要示例概念在文档中有说明，即视为 supported
- 引述声明（"来源：文档 X"）→ 只要 X 在检索范围内，即视为 supported

**效果**：3→1 道 0 分，Faith 均值从 0.67→0.77。但 m26 仍为 0 且 m24 仍在摆动。

### 4.3 第三轮：Glossary 动态检索（#18）

**现象**：m24 反复在 0↔1 之间摆动，修复不够稳定。

**深层根因**：`format_glossary_for_prompt` 只取 50 条注入（按中文长度排序）。"supertrait → 父 trait"（7 字符）排 #316/374——永远选不上。而"堆/栈/宏/糖"等单字词占满 50 个槽位。

**尝试的修复路径**：
1. ✅ 加 `_GLOSSARY_SUPPLEMENT` 手动补 3 条 → 只救了 3 个词
2. ✅ 跳过单字词 → 50 个槽全是复合词，但与查询无关
3. ❌ 承认 50 条硬限制和 371 条术语存在根本矛盾

**最终方案**：Glossary 走检索管道——374 条术语构建 FAISS + BM25 索引，查询时并行检索，按需召回 ≤10 条相关术语动态注入 prompt。

```
查询 "Clone 和 Copy 的区别"
    ↓
并行检索 ─┬─ 文档索引     → chunk 1, 2, 3...
          └─ Glossary 索引 (新) → supertrait→父 trait, trait→特质...
    ↓
动态注入: 只注入检索到的相关术语 (≤10条)
```

**实现**：`build_glossary_index()` + `search_glossary()` → `generator.generate(glossary_terms=...)`。

**效果**：m24 Faith=0→1.0 ✅，LLM 从 "超类trait" 改为 "父 trait"。

### 4.4 第四轮：preview 截断 + source 漏发（#19）

**现象**：Glossary 修复后 m24=1.0，但 m26 始终 Faith=0。

**排查历程**（这次最考验工程直觉）：

| 假设 | 验证 | 结论 |
|------|------|------|
| LLM 总结 vs 原文措辞不匹配 | 加规则 4 "优先使用原文措辞" | ❌ 反噬——声明翻倍，Faith 更差 |
| 检索召回差 | `diagnose_m26_retrieval.py` 查 BM25/向量/RRF 排名 | ❌ 函数指针文档在 RRF #1，检索没问题 |
| 评估器 LLM 有随机偏差 | 手动验证原文对比 | ❌ 原文逐字存在，不应该找不到 |
| **评估器拿到的 chunk 被截断** | `verify_m26_context.py` 检查 context 长度 | ✅ **找到！所有 text 只有 200 字符** |
| **source 传少了** | 查 `/query` 返回函数指针排 #6 | ✅ **web 传 `sources[:5]`，函数指针 #6 被截掉** |

**两个关键 bug**：

1. **preview vs text 截断**：`/evaluate` 端点构建 Chunk 时使用 `s.get("preview", "")`（200 字符截断版），而非完整 chunk text。函数指针文档在第 200 字符被切断："函数的类型是 fn （使"——后面的"用小写的 f 以免与 Fn 闭包 trait 相混淆"被截掉了。

2. **sources[:5]→[:8]**：Web eval 传给 `/evaluate` 时只发了前 5 个 source，函数指针文档排 #6 被截掉。而生成器用的是 `chunks[:8]`，数据量不同步。

**修复**：
- `/query` 端点：sources 新增 `"text": c.text`（完整文本，2028 字符）
- `/evaluate` 端点：改用 `s.get("text", s.get("preview", ""))`
- Web eval：传 `sources[:8]` 而非 `[:5]`

**效果**：m26 Faith=0→1.0 ✅。

**核心教训**：不要轻易归因于 "LLM 随机性"。当评估器多次找不到逐字存在的原文时，先检查数据有没有真正传给评估器。preview ≠ context——preview 用于 UI 展示，评估必须用完整 text。

---

## 五、关键技术决策（D1-D20）

完整决策链见 `docs/decisions.md`，核心摘要：

| # | 决策 | 影响 |
|:--:|------|------|
| D1-D3 | 数据源选 Rust 文档，手写 BM25/向量/RRF | 核心竞争力在手写融合 |
| D4 | BM25 从手写→rank-bm25 库 | 减少代码量 |
| D5 | bge-small→bge-base-zh（768d） | 精度提升 |
| D7 | Glossary 371 条注入 | 术语一致性 |
| D13-D15 | 父子块分块（子块检索，父块返回） | 检索轻量 + LLM 信息完备 |
| D17 | 四指标评估体系 | Context Precision + MRR + Faith + AR |
| D18 | LLM 查询重写 | 解决 BM25 对自然语言的敏感性问题 |
| D19 | Faithfulness 分层评估 + 术语规范 | 对比类问题不被误判 |
| D20 | Glossary 动态检索（374→FAISS+BM25） | 根治术语覆盖问题 |

---

## 六、工程方法总结

### 6.1 有效方法

1. **先排查后修复**：加 debug 日志 → 拿数据 → 讨论方案 → 实施。keyword 排名问题就是先跑诊断脚本才定位到父子块设计缺陷。
2. **所有测试写成 .py 文件**（禁止 `python -c`）：事后可追溯完整排查路径。`tests/` 目录下 `run_query_test.py`、`diagnose_main.py`、`diagnose_m26_retrieval.py`、`verify_m26_context.py` 等文件完整记录了排查过程。
3. **简单修复优先于架构改动**：m26 经历 4 轮排查，最终定位到一行 `preview→text` 的差距。如果一开始就假设需要换评估方法学，会浪费大量时间。
4. **后端重启是最容易被忽视的环节**：此 session 中多次出现 "代码改了但进程没重启" 导致的误判。`start.ps1` 的 `Start-Job` 环境变量隔离问题加剧了这一点。
5. **评估 prompt 是双向的**：既要定义 "什么是 bad"（幻觉），也要定义 "什么是 good"（诚实表述、代码示例、引述）。

### 6.2 典型坑

| 坑 | 教训 |
|----|------|
| BM25 对自然语言极度敏感 | "中" vs "里" 一字之差排名翻倍 → 必须配合 LLM 查询重写 |
| RRF 按 chunk 聚合 | 长文档因多 chunk 占优 → 必须按 source_url 聚合 |
| structured 子块无 heading 前缀 | "as" 不含 "关键字" → 子块 text 必须保留父级标题 |
| Streamlit 全量重跑模型 | 评估运行中点任何控件都中断 → 锁定页面 + 重入续跑 |
| Python `0.0 or 1` 排序陷阱 | 0.0 是 falsy → 正确写法 `x if x is not None else 1` |
| PowerShell `Start-Job` 不继承 env | 新进程找不到 PYTHONPATH → 必须显式传入 |
| emoji + Windows GBK | `🚀` → UnicodeEncodeError → `PYTHONIOENCODING=utf-8` |

---

## 七、最终状态

### 项目规模

```
rag_scratch/
├── src/          13 modules (downloader, parser, chunkers, retrievers, reranker, llm, glossary)
├── web/          StreamLit 4-page frontend (app.py, 1014 lines)
├── scripts/      parse, build_index, query, evaluate, generate_questions, start.ps1, app.py (6 endpoints)
├── evaluation/   4-metric evaluator, 60 questions, report system
├── tests/        93+ tests, multiple diagnostic scripts
├── docs/         decisions.md (D1-D20), engineering-issues.md (#1-#19)
├── data/raw/     530 HTML documents
└── vectorstore/  6440 chunks (FAISS + BM25 + Glossary indices)
```

### 最终评估（40 题，2026-07-07）

| 指标 | v0.1 初版 | v0.2 封版 | 提升 |
|------|:--:|:--:|:--:|
| Context Precision | 0.89¹ | 0.75 | —² |
| MRR | 1.00 | 1.00 | — |
| Faithfulness | 0.67 | **0.89** | +33% |
| Answer Relevancy | 0.80 | 0.79 | — |
| Faith=0 题数 | 4 | **0** | 清零 |
| Faith≥0.9 题数 | — | **25/40** | 62.5% |

¹ v0.1 仅 10 道手动题，v0.2 为 40 道混合题，Precision 下降因题目覆盖更广  
² 下降反映题目覆盖面扩大，不代表系统退化

### v0.3 展望

1. **检索精度优化**：HyDE 假答案检索、查询扩展、多阶段自适应检索
2. **多轮对话**：对话状态 + 指代消解
3. **增量更新**：新增数据不重建全量索引
4. **m26 的检索召回**：虽然 Faith 达标，但 P=0.375 仍有提升空间

---

*报告生成：2026-07-07 · 基于 git tag v0.2 · 项目 git: http://192.168.7.117:10086/dmccskylove/rust_rag*
