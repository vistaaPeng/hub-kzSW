# Rag Scratch v0.3 — Rust 知识库 RAG 问答系统

从零实现的检索增强生成（RAG）系统，以 Rust 中文文档（7 源，530 页 HTML）为知识库。
支持 BM25 + 向量 + RRF 融合检索、Cross-Encoder 重排、LLM 查询重写、Glossary 术语注入、父子块分块、四指标评估。

## 快速开始

```bash
cd E:\npl\workspaces\npl_tran\rag_scratch
set PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages
set PYTHONUTF8=1

# 数据已预下载：data/raw/ (530 HTML)
# 1. 解析 HTML → JSON
python scripts/parse.py

# 2. 构建索引（JSON → 父子块 → FAISS + BM25，首次 5-10 分钟）
python scripts/build_index.py
python scripts/check_index.py             # 可选：检查索引完整性

# 3. 交互查询
python scripts/query.py                  # 默认：LLM 查询重写 + RRF 融合
python scripts/query.py --rerank          # 启用 Cross-Encoder 重排
python scripts/query.py --no-rewrite      # 关闭查询重写
```

**两步分离**：改分块策略只需重跑 `build_index.py`，无需重解析 HTML。
Glossary (371 条术语) 在首次查询时自动下载并注入 LLM system prompt。
Windows 控制台建议设置 `PYTHONUTF8=1`，避免索引构建日志出现编码错误。

## 检索诊断

v0.3 新增单条查询诊断脚本，用于查看 BM25、向量检索和 RRF 融合各阶段的排名：

```bash
python scripts/diagnose_retrieval.py "rust有哪些关键字" --top-k 5 --candidate-k 50 --expect 关键字 --expect fn --expect let
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--top-k` | 输出每组排名的条数 |
| `--candidate-k` | 从 BM25/向量侧拉取的候选数 |
| `--expect` | 期望命中的关键词，可重复传入 |

## Web 界面

```bash
# 一键启动（前端 8501 + 后端 8000）
.\scripts\start.ps1

# 或分别启动
$env:PYTHONPATH = "S:/condaEnvs/py312/Lib/site-packages"
python scripts/app.py --port 8000          # 后端
python -m streamlit run web/app.py --server.port 8501  # 前端
```

打开 `http://localhost:8501`：

| 页面 | 功能 |
|------|------|
| 🔍 查询 | 自然语言提问 → 回答 + 8 来源（可点击跳转原文） |
| 📊 评估 | 40 题四指标评估 + 实时进度 + 历史回溯 + 上传替换问题集 |
| ⚙️ 设置 | top_k / rerank / rewrite 开关 |
| 📋 历史 | 查询日志持久化 + 可选列导出 JSON/CSV |

## 检索管线

```
用户问题
    ↓
LLM 查询重写（→ 3 个标准化查询）
    ↓
┌───────────────┬───────────────┐
│ BM25 检索      │ 向量检索       │
│ jieba + rank-bm25 │ bge-base-zh-v1.5 + FAISS IndexFlatIP │
└───────┬───────┴───────┬───────┘
        ↓               ↓
      RRF 融合（手写，按 source_url 聚合）
        ↓
  (可选) Cross-Encoder 重排 (bge-reranker-base)
        ↓
  子块 → expand_to_parents (兄弟父块拼接)
        ↓
  DeepSeek v4-flash 生成
        ↑ Glossary 371 术语注入 system prompt
```

## 评估结果（40 题手动，v0.2）

| 指标 | 均值 | 说明 |
|------|:--:|------|
| Context Precision | **0.75** | 检索结果中关键词命中比例 |
| MRR | **1.00** | 首个相关 chunk 的排名倒数 |
| Faithfulness | **0.89** | LLM 回答中对检索内容的忠实度 |
| Answer Relevancy | **0.79** | 回答与问题的语义切题度 |

> 手动 40 题 (高频 Rust 概念)：Precision 0.88, Faithfulness 0.84, Relevancy 0.79
> 自动 20 题 (niche 术语)：Precision 0.67, Faithfulness 0.81, Relevancy 0.67

评估脚本：`python scripts/evaluate.py --questions evaluation/questions_60.json`
详细报告含每题 chunk 追溯 + 逐条声明核查 → `evaluation/report_final.json`

## 环境要求

| 组件 | 说明 |
|------|------|
| Python | 3.12 (conda py312, `S:\condaEnvs\py312`) |
| GPU | GTX 1080 Ti 11GB |
| 模型 | bge-base-zh-v1.5 + bge-reranker-base (~1.5GB, 缓存于 `M:\huggingface_cache`) |
| LLM | DeepSeek v4-flash (`DEEPSEEK_API_KEY` 环境变量或 Windows 注册表) |
| Path 隔离 | 所有命令加 `PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages` |

运行参数也可以通过 `.env` 覆盖，参考 `.env.example`。

## 数据源

| 来源 | 页面 | 类型 |
|------|:--:|------|
| Rust Book | 105 | 教程 |
| Reference | 112 | 语言规范 |
| Rust By Example | 180 | 代码示例 |
| Cookbook | 66 | 代码片段 |
| Edition Guide | 72 | 版本迁移 |
| Rustdoc | 15 | 工具文档 |
| Glossary | 371 条 | 术语注入 prompt（不分块） |

## 核心模块

| 文件 | 功能 |
|------|------|
| `src/pipeline.py` | 统一 RAG 查询链路（CLI/Web/评估共用） |
| `src/config.py` | 运行配置与环境变量读取 |
| `src/downloader.py` | mdBook 文档下载 |
| `src/parser.py` | HTML → 结构化 JSON（morphology 标注） |
| `src/glossary.py` | Rust 术语表下载、注入 |
| `src/chunkers/narrative.py` | 父子块分块器（narrative/code_unit/spec/structured/note/mixed） |
| `src/retrievers/vector_store.py` | FAISS 向量检索 |
| `src/retrievers/bm25_store.py` | BM25 关键词检索 |
| `src/retrievers/hybrid_retriever.py` | RRF 融合 + expand_to_parents |
| `src/reranker/reranker.py` | Cross-Encoder 重排（可开关） |
| `src/llm/generator.py` | DeepSeek 生成（含 Glossary 注入） |
| `src/llm/query_rewriter.py` | LLM 查询重写 + 多路合并 |
| `evaluation/evaluator.py` | 四指标评估器 + 报告生成 |
| `evaluation/question_generator.py` | LLM 自动出题 |
| `scripts/check_index.py` | 索引完整性检查 |
| `scripts/diagnose_retrieval.py` | 单条 query 检索诊断 |

## 项目结构

```
rag_scratch/
├── README.md, CHANGELOG.md, CONTEXT.md, requirements.txt, run_tests.bat
├── .env.example
├── docs/{decisions,engineering-issues,project-retrospective}.md
├── scripts/
│   ├── start.ps1, parse.py, build_index.py, check_index.py
│   ├── query.py, diagnose_retrieval.py
│   ├── app.py, evaluate.py, generate_questions.py
│   └── run.sh
├── web/
│   └── app.py                 # StreamLit 前端（4 页面）
├── src/
│   ├── config.py, pipeline.py
│   ├── downloader.py, parser.py, glossary.py
│   ├── chunkers/narrative.py
│   ├── retrievers/{vector_store,bm25_store,hybrid_retriever}.py
│   ├── reranker/reranker.py
│   └── llm/{generator,query_rewriter}.py
├── evaluation/
│   ├── evaluator.py, question_generator.py
│   ├── questions_60.json      # 40 道手动评估题
│   └── results/               # 评估历史归档
├── logs/queries.jsonl         # 结构化查询日志
├── tests/ (109 tests)
├── data/raw/ (530 HTML)
└── vectorstore/ (索引文件)
```

## 测试

```bash
# 纯逻辑测试（不加载模型，18s）
python -m pytest tests/ -v -k "not integration and not vector and not reranker"

# 全部测试
set PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages
set HF_HOME=M:/huggingface_cache
python -m pytest tests/ -v -k "not integration"
```
