# rag_scratch — 项目上下文

Rust 知识库 RAG 问答系统——从零实现的检索增强生成项目。
数据源：7 个 Rust 中文文档网站（530 HTML），技术栈：bge-base-zh-v1.5 + FAISS + BM25 + 手写 RRF + DeepSeek v4-flash。
父子块分块、LLM 查询重写、Glossary 术语注入、四指标评估。

## 核心领域术语

**Chunk（块）**：文档被分割后的最小检索单元。分父子两层——子块（100-400字符）用于检索，父块（section 级）通过兄弟拼接返回 LLM。

**数据形态**：chunk 的分类标签，6 种形态：narrative、code_unit、specification、structured、note、mixed。

**检索管线**：用户问题 → LLM 查询重写（3 个变体）→ 多路并行检索（BM25 + 向量）→ RRF 融合（手写，按 source_url 聚合）→ 子块 expand_to_parents → LLM 生成。

**RRF（倒数秩融合）**：`score(doc) = Σ(1/(k + best_rank_i(doc)))`，手写实现，按文档聚合避免长文档占优。

**父子块**：child 检索、parent 返回。expand_to_parents 支持兄弟父块拼接（前后各一个）。

**查询重写**：用 DeepSeek 将用户自然语言改写为 2-3 个标准化检索查询，消解"中"/"里"等同义表达对 BM25 的不稳定性。

**中间格式（parsed JSON）**：HTML 解析后的结构化中间产物，每个 element 标注 morphology 和 type。分块器消费 JSON，不直接处理 HTML。

## 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据下载 | src/downloader.py | 递归抓取 mdBook 文档 |
| HTML 解析 | src/parser.py | HTML → JSON（morphology 标注） |
| Glossary | src/glossary.py | 371 条 Rust 中英术语 → LLM prompt |
| 分块器 | src/chunkers/narrative.py | 父子块生成（6 形态） |
| 向量检索 | src/retrievers/vector_store.py | bge-base-zh-v1.5 + FAISS IndexFlatIP |
| BM25 | src/retrievers/bm25_store.py | jieba + rank-bm25 |
| 融合检索 | src/retrievers/hybrid_retriever.py | RRF + expand_to_parents |
| 重排 | src/reranker/reranker.py | bge-reranker-base（可选开关） |
| LLM 生成 | src/llm/generator.py | DeepSeek + Glossary 注入 |
| 查询重写 | src/llm/query_rewriter.py | LLM 重写 + 多路合并 |
| 评估 | evaluation/evaluator.py | 四指标 + 详细报告 |

## 关键决策

| 决策 | 内容 | 记录 |
|------|------|:--:|
| 数据源 | 7 个 Rust 中文文档网站 | D1 |
| 分块策略 | 按数据形态，不按来源 URL | D2 |
| BM25+RRF | 库处理 BM25，手写 RRF 融合 | D4 |
| 向量检索 | bge-base-zh-v1.5 + FAISS IndexFlatIP | D5 |
| 重排 | bge-reranker-base 可选开关 | D6 |
| Glossary | 不分块，注入 system prompt | D7 |
| LLM | DeepSeek v4-flash | D8 |
| 父子块 | 子块检索，父块返回 LLM | D13 |
| 查询重写 | LLM 3 路重写 + 多路合并 | D18 |
| 评估 | Precision + MRR + Faithfulness + Relevancy | D17 |

## Gotchas

**Path 污染**：Hermes 设置 `PYTHONPATH=M:\hermesData\hermes-agent\...` → 子进程优先加载 Hermes venv 的包。修复：运行时覆盖 `PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages`。

**BM25 vs 自然语言**：jieba 对同义表达敏感（"中有哪些" vs "里有更多"→BM25 排名翻倍）。查询重写从入口消解此问题。

**structured 子块文本**：列表条目本身不含标题关键词（如"as - 强制类型转换"不含"关键字"），需在子块 text 中保留 headings 前缀。

**chunk_id 唯一性**：file_index 必须在父块/子块 ID 中，否则多文件产生冲突。格式：`{source}_f{file_index:04d}p{section_idx:04d}`。

**API Key 获取**：环境变量 → winreg → PowerShell 三级回退，加校验（`sk-` 前缀 + 长度 > 20）。
