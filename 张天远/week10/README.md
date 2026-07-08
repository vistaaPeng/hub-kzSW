# rag_scratch

Rust 中文资料 RAG 问答系统。项目以 Rust 中文文档为知识库，提供
BM25、向量检索、RRF 融合、可选 Cross-Encoder 重排、Glossary 动态检索、
父子 chunk 扩展、LLM 生成、Web 查询和评估能力。

当前工程重点已经从“能跑”推进到“能诊断、能验证、能稳定迭代”：

- 统一 `RAGPipeline`，CLI、Web、评估共用同一查询链路。
- 支持索引完整性检查，降低索引过期或数量不一致导致的隐性错误。
- 支持单条查询诊断，展示 BM25、向量和 RRF 排名。
- BM25 tokenizer 注入 Rust 关键词和 Glossary 术语。
- RRF 融合加入 heading、morphology 和 query intent 的轻量 boost。
- 父块扩展以命中位置为中心，减少长文档上下文错位。

## Quick Start

```powershell
cd E:\npl\workspaces\npl_tran\rag_scratch
$env:PYTHONPATH = "S:/condaEnvs/py312/Lib/site-packages"
$env:HF_HOME = "M:/huggingface_cache"
$env:PYTHONUTF8 = "1"
```

### 1. 构建索引

数据已下载并解析时，直接重建索引：

```powershell
S:\condaEnvs\py312\python.exe scripts/build_index.py
```

构建完成后建议检查索引：

```powershell
S:\condaEnvs\py312\python.exe scripts/check_index.py
```

当前一次重建结果：

- 来源：6 个 source
- 总 chunks：6440
- 子块索引：4589
- 父块：1851
- BM25 docs：4589
- Glossary 术语：374

### 2. 命令行查询

```powershell
S:\condaEnvs\py312\python.exe scripts/query.py
S:\condaEnvs\py312\python.exe scripts/query.py --rerank
S:\condaEnvs\py312\python.exe scripts/query.py --no-rewrite
```

### 3. Web 界面

```powershell
.\scripts\start.ps1
```

默认入口：

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

### 4. 检索诊断

对单条 query 查看 BM25、向量和 RRF 排名：

```powershell
S:\condaEnvs\py312\python.exe scripts/diagnose_retrieval.py "rust有哪些关键字" --top-k 5 --candidate-k 50 --expect 关键字 --expect fn --expect let
```

这个工具用于定位召回问题来自 BM25、向量召回、RRF 融合还是 metadata 排序信号。

## Retrieval Pipeline

```text
用户问题
  -> 可选 LLM query rewrite
  -> BM25 检索
       - jieba
       - Rust 关键词注入
       - Glossary 术语注入
  -> 向量检索
       - bge-base-zh-v1.5
       - FAISS IndexFlatIP
  -> RRF 融合
       - 按 source_url 聚合
       - heading match boost
       - morphology/query intent boost
  -> 可选 Cross-Encoder rerank
  -> child chunk 扩展到 parent/siblings
  -> Glossary 动态补充
  -> DeepSeek 生成答案
```

## 真实查询验证

重建索引并启用当前 RRF metadata boost 后，典型查询结果如下：

| Query | 当前 top result |
|---|---|
| `rust有哪些关键字` | `reference > 关键字 > 严格关键字` |
| `如何写Rust函数示例` | `ch03-03-how-functions-work` 进入 top2 |
| `Rust所有权和借用有什么区别` | `认识所有权` |
| `trait是什么` | `reference > 术语表 > 本地 trait` |
| `生命周期是什么` | `rust-by-example > 生命周期` |

## Evaluation

运行测试：

```powershell
S:\condaEnvs\py312\python.exe -m pytest tests/ -q --basetemp .pytest_tmp_all
```

最近一次全量单元测试：

```text
109 passed, 4 warnings
```

运行评估：

```powershell
S:\condaEnvs\py312\python.exe scripts/evaluate.py --questions evaluation/questions_60.json
```

## Core Modules

| Path | Role |
|---|---|
| `src/pipeline.py` | 统一 RAG 查询 pipeline |
| `src/config.py` | 环境变量和运行配置 |
| `src/chunkers/narrative.py` | 父子 chunk 切分和 morphology 标注 |
| `src/retrievers/vector_store.py` | FAISS 向量检索 |
| `src/retrievers/bm25_store.py` | BM25 检索和领域 tokenizer |
| `src/retrievers/hybrid_retriever.py` | RRF 融合、metadata boost、父块扩展 |
| `src/glossary.py` | Glossary 下载、缓存和动态检索 |
| `src/llm/query_rewriter.py` | LLM query rewrite |
| `src/llm/generator.py` | DeepSeek 答案生成 |
| `evaluation/evaluator.py` | 指标评估和报告生成 |
| `scripts/check_index.py` | 索引完整性检查 |
| `scripts/diagnose_retrieval.py` | 单条查询检索诊断 |

## Documentation

| Path | Description |
|---|---|
| `docs/project-retrospective.md` | 从接手到当前的工程复盘 |
| `docs/decisions.md` | 架构和工程决策记录 |
| `docs/engineering-issues.md` | 工程问题排查记录 |
| `CHANGELOG.md` | 版本变更记录 |

## Environment Notes

- Python: 3.12，推荐环境 `S:\condaEnvs\py312`
- Embedding: `bge-base-zh-v1.5`
- Reranker: `bge-reranker-base`
- LLM: DeepSeek v4-flash，需要 `DEEPSEEK_API_KEY`
- Windows 控制台建议设置 `$env:PYTHONUTF8 = "1"`，避免构建脚本日志编码问题。

## Index Artifacts

`vectorstore/` 中的索引文件由 `scripts/build_index.py` 生成。当前 git 状态显示这些索引产物没有进入普通提交；如果后续索引构建成本变高，建议使用 release artifact 或外部存储管理，而不是直接压入 Git。
