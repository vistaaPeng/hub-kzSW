# Rag Scratch 决策流程记录

> 从零构建 Rust 知识库 RAG 系统 — 全部关键决策及变更原因

## 决策时间线

```
Phase 1  Grill 盘问
  ├─ D1  数据源：Rust 知识库 ×7
  ├─ D2  分块策略：按数据形态，不按文档类型
  ├─ D3  HTML→JSON 中间格式解耦
  ├─ D4  BM25：全用库 rank-bm25，RRF 手写
  ├─ D5  向量库重评：bge-base-zh + FAISS IndexFlatIP
  ├─ D6  重排：bge-reranker-base，可开关
  ├─ D7  Glossary：不分块，注入 system prompt
  └─ D8  LLM：DeepSeek v4-flash

Phase 2  编码实现
  ├─ D9   执行策略：Claude Code 认证失败 → 手写 + 子代理并行
  ├─ D10  Path 污染：Hermes venv 覆盖 conda → PYTHONPATH 隔离
  ├─ D11  解析→索引分两步骤：parse.py + build_index.py
  ├─ D12  形态标注：parser 阶段标注 morphology 字段
  └─ D13  分块演进：叙述段 → 代码单元 → 父子块方案 B

Phase 3  检索修复
  ├─ D14  "关键字"召不回 → RRF 按文档聚合 + 最佳排名
  └─ D15  父子块：子块检索，父块返回 LLM

Phase 4  评估
  ├─ D16  测试题：LLM 自动生成 60 道
  └─ D17  Context Precision + 预留 Recall
```

---

## Phase 1: Grill 盘问

### D1 — 数据源选择

| 候选 | 评估 | 结论 |
|------|------|------|
| A Python 中文文档 | 结构化好、代码+中文混排 | — |
| B 中国法律条文 | QA 对单调 | ❌ |
| C 维基百科中文 | 条目质量参差不齐 | ❌ |
| D 自有文档 | — | ❌ |
| **A' Rust 知识库** | 7 个来源、教程+参考+代码示例混排 | ✅ |

**最终选择**：Rust 知识库 ×7
- https://rustwiki.org/zh-CN/book/ (105页)
- https://rustwiki.org/zh-CN/reference/ (112页)
- https://rustwiki.org/zh-CN/rust-by-example/ (180页)
- https://rustwiki.org/zh-CN/rust-cookbook/ (66页)
- https://rustwiki.org/zh-CN/edition-guide/ (72页)
- https://rustwiki.org/zh-CN/rustdoc/ (15页)
- https://rustwiki.org/wiki/translate/english-chinese-glossary-of-rust/ (371条术语)

**理由**：代码密集型文档压榨 BM25+向量互补性，分块策略设计空间大，用户自己在学 Rust。

---

### D2 — 分块策略

| 版本 | 方案 | 问题 |
|------|------|------|
| v1（被否） | 按文档来源分类：Book/Reference/RBE 各有不同策略 | 颗粒太粗，同一页面内混合多种内容 |
| v2（采用） | **按数据形态分类**：叙述段 / 代码单元 / 规范条目 / 表格列表 / 混合区 | 基于数据单元自身特征，不预设来源标签 |

**原则**：Chunker 接口统一，内部根据 morphology 字段自动路由。

---

### D3 — HTML 解析中间格式

| 方案 | 流程 | 迭代成本 |
|------|------|----------|
| A 一步到位 | HTML → chunk 列表 | 调分块策略 = 重新下载+解析 |
| **B（采用）** | **HTML → JSON → chunk** | 改分块 = 只改 chunker |

**理由**：分块需要反复调参（块大小、重叠、代码边界），中间 JSON 缓存避免重复网络 I/O。

---

### D4 — BM25 "手写"范围

| 组件 | 手写？ | 最终决定 |
|------|--------|----------|
| 分词 | 用 jieba | ✅ 库 |
| TF/IDF 计算 | — | ✅ rank-bm25 库 |
| BM25 评分公式 | — | ✅ rank-bm25 库 |
| 倒排索引 | — | ✅ rank-bm25 库 |
| **RRF 融合** | **必须手写** | ✅ 手写 |
| **检索管线编排** | **必须手写** | ✅ 手写 |

---

### D5 — 向量库 + Embedding 重评

**约束**：~2000-5000 chunks、中英混合、GTX 1080 Ti 11GB

| 模型 | 维度 | 中文 | 代码 | 决定 |
|------|------|------|------|------|
| bge-small-zh-v1.5 | 512 | ⭐⭐⭐ | ⭐⭐ | 已下载但舍弃 |
| **bge-base-zh-v1.5** | **768** | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| bge-m3 | 1024 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 过重 |

**向量库**：FAISS IndexFlatIP（精确暴力，5000 × 768 < 20MB，单次查询 < 5ms）

---

### D6 — 重排器

| 方案 | 决定 |
|------|------|
| 要重排 | ✅ bge-reranker-base (1.11GB)，Cross-Encoder 精排 |
| 可开关 | ✅ enabled=True/False 参数 |
| 集成到检索器 | ✅ HybridRetriever 可选参数 |

---

### D7 — Glossary 术语表处理

| 方案 | 决定 |
|------|------|
| 不分块，不作为检索源 | ✅ |
| 下载 371 条中英对照 → 注入 LLM system prompt | ✅ |
| 确保生成答案的译名一致性 | ✅ |

---

### D8 — LLM 选型

| 候选 | 评估 | 决定 |
|------|------|------|
| DeepSeek API | Key 已配好（HKCU 环境变量）、中文+代码能力不错 | ✅ |
| DashScope qwen-plus | 需要额外配置 | ❌ |
| **模型** | deepseek-v4-flash（轻量，适合 RAG） | ✅ |

---

## Phase 2: 编码实现

### D9 — Claude Code 认证失败 → 策略调整

| 尝试 | 结果 |
|------|------|
| `claude -p` 403 Forbidden | 免费计划不支持 |
| `claude auth login` OAuth | 已登录但 -p 仍 403 |
| 最终方案 | **手写核心代码 + 子代理并行执行独立模块** |

**教训**：后续切片 8-10 中，无依赖的模块用 `delegate_task` 并行调度（切片 4+5+6、切片 8+9），提升效率。

---

### D10 — Path 污染根因

**问题**：Hermes venv 的 numpy (cp311)、pydantic 覆盖 conda py312 (cp312)，所有 import 报错。

**根因**：
```
PYTHONPATH=M:\hermesData\hermes-agent;M:\hermesData\hermes-agent\venv\Lib\site-packages
```
被所有子进程继承，排在 conda site-packages 前面。

**修复**：所有命令强制覆盖
```bash
PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages S:/condaEnvs/py312/python.exe -m pytest tests/
```
固化到 `run_tests.bat`、`scripts/run.sh`、CONTEXT.md Gotchas。

---

### D11 — 解析与索引分离

**背景**：build_index.py 最初一体化（HTML→JSON→chunk→索引），用户指出改分块策略需要独立命令。

**最终**：
```bash
python scripts/parse.py          # HTML → JSON（纯 I/O）
python scripts/build_index.py    # JSON → chunk → 索引（改分块只跑这个）
python scripts/query.py          # 交互查询
```

---

### D12 — 形态标注演进

**v1**：parser 只标注 HTML 元素类型（heading/paragraph/code_block），不标注数据形态。

**问题**：用户指出"没有完全按照讨论的原则来"——讨论中明确分块基于数据形态，不是 HTML 标签。

**v2**：parser 阶段标注 morphology 字段：

| HTML 元素 | morphology |
|-----------|------------|
| `<p>` | `narrative` |
| `<p>` 含 `<code>` | `mixed` |
| `<p>` 短句+规范关键词 | `specification` |
| `<pre><code>` | `code_unit` |
| `<ul>/<ol>` | `structured` |
| `<table>` | `structured` |
| `<blockquote>` | `note` |

**向后兼容**：chunker 优先读 morphology，回退 type。

---

### D13 — 分块策略演进

| 版本 | 内容 | 触发 |
|------|------|------|
| **v1** | 叙述段按 h2 切分，max_chunk_size=500 | 初始 |
| **v2** | + 代码单元独立（code_block 后跟随说明） | 代码检索准确性 |
| **v3** | + morphology 字段驱动路由 | 用户要求 |
| **v4** | **父子块方案 B**（当前） | "关键字"召回问题 |

#### v4 父子块详情

**子块（检索用）**：100-400 字符，按形态切分
- narrative：按句子/段落边界切分，目标 200-400 字符
- code_unit：独立（代码不拆分）
- specification：独立
- structured：列表按 4 条目/组拆分
- note：合并到前一个 narrative（P2 改进）
- 最小 80 字符（P4 改进）

**父块（LLM 用）**：section 级别完整内容，含标题链

**映射**：子块 `parent_chunk_id` → 父块 `chunk_id`

**P0-P4 改进一并实施**：
| 优先级 | 改进 | 解决问题 |
|--------|------|----------|
| P0 | 过滤 git 元数据 noise section | reference 误召回 |
| P1 | structured 长列表拆分 | 关键字列表被切碎 |
| P2 | note 合并到 narrative | 警告框脱上下文 |
| P3 | specification 独立分块 | Reference 规范条目精度 |
| P4 | 最小 80 字符下限 | 碎片化 |

---

## Phase 3: 检索修复

### D14 — RRF 融合修复

**问题**：用户测试"Rust 有哪些关键字"，BM25 和向量单独检索都对（#1 命中），RRF 融合后完全不相关。

**根因分析**：

| 版本 | 机制 | 问题 |
|------|------|------|
| v1（有 bug） | 按 chunk_id 聚合 RRF | 长文档多 chunk 得多次加分，自然占优 |
| v2（修复） | 按 source_url 聚合，取最佳 chunk 分数 | 单边 #1 不如双边 #10（RRF 公式本质） |
| **v3（最终）** | **按 source_url 聚合，取最佳排名再算 RRF + 默认排名** | ✅ |

**最终 RRF 公式**：`score(doc) = 1/(k+best_bm25_rank) + 1/(k+best_vector_rank)`
- 不在某检索器中的文档，赋予 default_rank = candidate_k + 1

---

### D15 — 父子块解决"关键字列表"问题

**问题**：38 个关键字的 chunk 是 1101 字符，被截断。LLM 只看到 chunk 的说明段"以下是 Rust 的关键字"，没看到列表本身。

**方案 B（采用）**：显式父子块
- 检索命中子块 → expand_to_parents → 返回完整 section 给 LLM
- 子块小而精准（200-400 字符），父块完整（≤3000 字符）

---

## Phase 4: 评估

### D16 — 测试题构建

**最终方案**：LLM 自动生成 60 道（混合 4 种题型：概念解释、代码补全、对比分析、术语定义）

**预留**：手动 40 道 + ground truth 标注（待用户确定方案）

---

### D17 — 评估指标

**已实现**：Context Precision（检索结果中命中关键词的比例）

**预留接口**：Recall（`compute_recall` 方法，暂返回 -1.0）

---

### D18 — LLM 查询重写

**背景**：用户输入 `"rust中有哪些关键字"` vs `"rust里有哪些关键字"`，仅一字之差，BM25 排名从 #36 恶化到 #76，RRF 排名从 #3 掉到 #6。BM25 对自然语言同义表达极度敏感。

**策略选择**：

| 候选策略 | 评估 | 结论 |
|----------|------|------|
| 1. LLM 重写问题 | 把口语化表达标准化，消解 BM25 不稳定性 | ✅ 采用 |
| 2. HyDE（假答案检索） | 对列表/枚举型查询有效，但向量侧已排 #2，非瓶颈 | 后续增强 |
| 3. Reranking | keyword 粗排 #76，Cross-Encoder 看不到 | ❌ 无效 |

**实现**：`src/llm/query_rewriter.py`
- DeepSeek v4-flash 把用户查询改写为 3 个标准化版本
- 多路并行检索 → 按 source_url 去重合并（保留最高 RRF 分）
- 结果：`"rust里有哪些关键字"` 和 `"rust中有哪些关键字"` 均稳定排到 #0

**API Key 获取**：环境变量 → winreg（Windows 注册表）→ PowerShell 三级回退，加校验（`sk-` 前缀 + 长度 > 20），避免损坏的环境变量干扰。

---

### D19 — Faithfulness 评估分层 + LLM 术语规范化

**背景**：40 题评估中 4 道 Faith=0，10 道 Faith<0.34。排查发现：(1) 对比/推导类问题的合成答案无法逐句匹配单个 chunk；(2) LLM 自造术语（"超类trait"）在检索 chunk 中找不到对应原文。

**策略**：

| 方向 | 措施 | 效果 |
|------|------|------|
| **术语规范** | generator system prompt 新增第 5 条：必须使用 Glossary 标准译名，禁止自造词 | 从源头减少术语不匹配 |
| **分层评估** | 问题分类器（关键词："区别""对比""优劣""vs"等 15 个）→ 推导类用宽松 prompt（允许联合多 chunk 推导），事实类保持严格逐句核查 | 推导类不再被误判 0 分 |

**分类器设计**：
- 推导类关键词：区别、对比、比较、不同、异同、优劣、优缺点、关系、联系、vs、versus、有何不同、如何选择、如何区分、如何影响
- "如何"不单独作为推导标志（"如何编写文档"是用法问题，非推导）
- 回退规则：短问题 + "是什么" → 事实类；长问题（>60 字符）→ 推导类

**推导类 prompt 核心**：区分"事实声明"（需直接依据）和"推导结论"（前提覆盖 + 逻辑关系 → supported）。

**实现**：`evaluation/evaluator.py`（`_classify_question_type` + 双 prompt `compute_faithfulness`）、`src/llm/generator.py`（SYSTEM_PROMPT 第 5 条）、`tests/test_faithfulness_fix.py`（10/10 通过）

**v2 精炼（#16→#17）**：第一轮效果不佳（3 题仍 Faith=0），原因是评估 prompt 只定义了"什么是 bad"，没有定义"什么是 good"→ 追加三条豁免规则：诚实表述、代码示例、引述声明均直接判定为 supported。`tests/test_zero_faith_root_cause.py` 逐题检测确认根因。

**v3 精炼（#19）**：m26 被 preview 截断（200 字符）导致原文不可见 → /query 新增 `text` 字段（完整文本），/evaluate 改用完整 text。修复后 Faith=0.909。

---

### D20 — Glossary 动态检索替代静态注入

**背景**：D19 的术语规范 + D7 的 Glossary 50 条注入仍不能根治术语不规范问题——`supertrait → 父 trait` 排在 #316/374，永远注入不到 prompt。不管怎么优化选择策略（跳过单字词、排序策略），50 条硬上限和 374 条术语存在根本矛盾。

**方案**：Glossary 改为检索管道——374 条术语构建 FAISS + BM25 索引，查询时并行检索，按需召回 ≤10 条相关术语动态注入 prompt。

**效果**：m24 Faith=0 → **1.0**。不再需要 `_GLOSSARY_SUPPLEMENT` 补丁。

**实现**：`src/glossary.py`（`build_glossary_index` + `search_glossary`）、`src/llm/generator.py`（`glossary_terms` 参数）、`scripts/app.py`（集成调用）

---

## 技术栈最终清单

| 组件 | 选择 | 变更记录 |
|------|------|----------|
| 数据 | Rust 知识库 ×7 (530 HTML) | — |
| 分词 | jieba | — |
| BM25 | rank-bm25 库 | D4：手写 → 全用库 |
| 向量库 | FAISS IndexFlatIP | D5：bge-small → bge-base-zh |
| Embedding | bge-base-zh-v1.5 (768d) | D5 |
| RRF | 手写 | D14：3 次修复 |
| 重排 | bge-reranker-base (可开关) | D6 |
| LLM | DeepSeek v4-flash | D8 |
| 分块 | 父子块（子块检索，父块返回） | D13、D15 |
| 评估 | Context Precision + MRR + Faithfulness + Answer Relevancy | D17、**D19** |
| 术语 | Glossary 371 条注入 prompt | D7 |
| 查询重写 | LLM 重写为 2-3 个标准化查询 + 多路合并 | D18 |
| LLM 生成 | System prompt 强化术语规范 + 动态 Glossary 检索 | D19、**D20** |

---

## 项目文件清单

```
rag_scratch/
├── CONTEXT.md               # 术语表 + Gotchas
├── README.md                # 使用说明
├── requirements.txt
├── run_tests.bat            # 测试入口（含 path 污染修复）
├── docs/
│   ├── adr/                 # 架构决策记录
│   └── decisions.md         # 本文件
├── scripts/
│   ├── run.sh               # 环境隔离脚本
│   ├── parse.py             # HTML → JSON
│   ├── build_index.py       # JSON → 父子块 → 索引
│   └── query.py             # 交互查询
├── src/
│   ├── downloader.py
│   ├── parser.py            # morphology 标注
│   ├── glossary.py
│   ├── chunkers/narrative.py # 父子块
│   ├── retrievers/
│   │   ├── vector_store.py
│   │   ├── bm25_store.py
│   │   └── hybrid_retriever.py # expand_to_parents
│   ├── reranker/reranker.py
│   └── llm/
│       ├── generator.py
│       └── query_rewriter.py
├── evaluation/
│   ├── evaluator.py
│   └── question_generator.py
├── tests/ (93 tests)
├── data/raw/ (530 HTML)
└── vectorstore/
```
