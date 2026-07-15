# CHANGELOG

## v0.3 (2026-07-08) — 检索诊断与召回稳定化

### 新增
- **统一 RAGPipeline**：CLI、Web、评估统一走同一条查询链路，减少入口漂移。
- **索引完整性检查**：新增 `scripts/check_index.py`，检查索引文件、chunk 数量、FAISS/BM25 数量和 metadata 基本一致性。
- **检索诊断报告**：新增 `scripts/diagnose_retrieval.py`，按单条 query 展示 BM25、向量和 RRF 排名，方便定位召回问题。
- **运行配置外置**：新增 `src/config.py` 和 `.env.example`，支持通过环境变量覆盖运行参数。
- **工程复盘文档**：新增 `docs/project-retrospective.md`，归档本轮工程决策、经验和后续建议。

### 改进
- **父块扩展以命中位置为中心**：`expand_to_parents` 不再从文档开头拼接，而是围绕命中的 parent chunk 扩展上下文。
- **Glossary 检索器缓存**：避免重复加载术语表检索器，减少查询链路开销。
- **BM25 tokenizer 注入 Rust 领域词**：注入 Rust 关键字、常见技术术语和 Glossary 术语；对部分英文短语追加 phrase token。
- **RRF metadata boost**：利用 headings、morphology 和 query intent 做轻量排序修正：
  - heading 命中加权；
  - `有哪些/列表/关键字/保留字` 偏好 `structured`；
  - `代码/示例/如何/函数` 偏好 `code_unit`；
  - `区别/对比/比较` 偏好 `narrative/mixed`。
- **重建索引**：tokenizer 改动后已重建索引，当前索引包含 6440 chunks、4589 个子块、1851 个父块、4589 个 BM25 docs、374 个 Glossary 术语。

### 验证
- 全量单元测试：`109 passed, 4 warnings`。
- 真实查询复测：
  - `rust有哪些关键字`：top1 命中 `reference > 关键字 > 严格关键字`；
  - `如何写Rust函数示例`：`ch03-03-how-functions-work` 进入 top2；
  - `Rust所有权和借用有什么区别`：top1 命中 `认识所有权`；
  - `trait是什么`：top1 命中 `reference > 术语表 > 本地 trait`；
  - `生命周期是什么`：top1 命中 `rust-by-example > 生命周期`。

### 注意
- Windows 下重建索引建议设置 `PYTHONUTF8=1`，避免控制台日志编码问题。
- 当前 `diagnose_retrieval.py` 的 RRF breakdown 仍展示基础 RRF，后续建议补充 metadata boost 明细，使诊断分数与最终 `HybridRetriever.search()` 完全对齐。

## v0.2 (2026-07-07) — 封版 — Web 界面 + 完整评估

### 新增
- **StreamLit Web 界面** (`web/app.py`)：查询、评估、设置、历史 4 页面
- **FastAPI 后端扩展**：`GET /stats`、`GET /history`、`POST /parse`、`POST /evaluate`
- **实时评估**：进度条 + 每题实时打分（四指标） + 结果持久化 + 历史回溯
- **一键启动** (`scripts/start.ps1`)：前后端自动启动 + 旧进程清理
- **查询日志固化**：写入 `logs/queries.jsonl`，前端历史页可选列导出
- **评估问题管理**：上传替换问题集 + 翻页浏览（10道/页）
- **安全防护**：注入检测 + 离题拒绝 + 空检索拦截

### 改进
- **Faithfulness + Answer Relevancy**：后端 `/evaluate` 端点 + 评估页实时显示
- **来源全显示**：API 返回全部 8 个来源（对齐 LLM 使用量），URL 可点击
- **查询重写降级**：API 失败自动用原查询，不阻塞
- **兄弟拼接**：±1 → 全部同源父块拼接（上限 3000 字符）
- **prompt 强化 + temperature 0.3→0.1**：减少幻觉
- **max_chunks 5→8**：更多上下文
- **前端缓存**：文件读/API 调用 TTL 缓存，页面响应提速
- **评估中断保护**：运行中锁定控件 + 重入续跑

### 评估结果（40 题手动，v0.2 改进后）
| 指标 | 均值 |
|------|:--:|
| Context Precision | 0.88 |
| MRR | 1.00 |
| Faithfulness | 0.84 |
| Answer Relevancy | 0.79 |

### 改进
- **Glossary 动态检索**：374 术语 → FAISS + BM25 索引，按需召回（替代 50 条静态注入）
- **m24 Faith=0 → 1.0**：动态检索 `supertrait→父 trait`，LLM 不再自造"超类trait"
- **评估报告增强**：含 faithfulness_detail/relevancy_detail/retrieved_chunks

### 文档
- `scripts/start.ps1` — 一键启动脚本
- `web/app.py` — StreamLit 前端（935 行）
- `docs/engineering-issues.md` — 新增 #10-#19
- `docs/decisions.md` — 新增 D18、D19、D20
- `tests/test_faithfulness_fix.py` — Faithfulness 分类器测试（10/10）
- `tests/test_zero_faith_root_cause.py` — Faith=0 逐题根因检测
- `tests/test_single_m24.py` — m24 单题端到端诊断

## v0.1 (2026-07-06) — 首次完整版本

### 核心功能
- **检索**：BM25 (jieba + rank-bm25) + 向量 (bge-base-zh-v1.5 + FAISS IndexFlatIP) + 手写 RRF 融合
- **重排**：bge-reranker-base Cross-Encoder（可选开关）
- **分块**：父子块模式，6 种数据形态（narrative/code_unit/specification/structured/note/mixed）
- **生成**：DeepSeek v4-flash，371 条 Glossary 术语注入 system prompt
- **查询重写**：LLM 自动改写为 3 个标准化查询 + 多路合并，消解自然语言不稳定性
- **评估**：Context Precision + MRR + Faithfulness + Answer Relevancy，详细报告可追溯

### 数据
- 7 源 Rust 中文文档，530 HTML
- 6440 chunks（1851 父块 + 4589 子块）

### 评估结果（60 题混合）
| 指标 | 均值 |
|------|:--:|
| Context Precision | 0.81 |
| MRR | 1.00 |
| Faithfulness | 0.83 |
| Answer Relevancy | 0.75 |

### 可靠性
- 查询重写 API 失败自动降级
- 兄弟父块全部拼接（上限 3000 字符）
- 结构化查询日志（`logs/queries.jsonl`）
- FastAPI 接口（`scripts/app.py`）

### 文档
- `README.md` — 项目概览 + 使用说明
- `CONTEXT.md` — 术语表 + Gotchas
- `docs/decisions.md` — 18 个架构决策
- `docs/engineering-issues.md` — 10 个工程问题排查

---
