# rag_scratch 项目工程复盘

日期：2026-07-07

本文复盘从接手 `rag_scratch` 项目到目前为止的主要判断、工程决策、验证方式和阶段成果。重点不是罗列提交，而是说明：为什么先处理这些问题，哪些经验可以复用，以及后续继续优化召回质量时应该沿着什么路径推进。

## 1. 接手后的总体判断

项目已经具备一个完整的 Rust 中文资料 RAG 原型：解析文档、切分 chunk、构建向量索引与 BM25 索引、混合检索、父块扩展、术语表补充、LLM 生成和评估脚本都已经存在。问题不在于“从零搭系统”，而在于工程链路开始变长之后，几个入口和中间产物之间出现了不一致。

接手时我把问题分成三层：

1. 查询链路一致性：CLI、Web app、批量评估是否走同一条 pipeline。
2. 索引与检索可解释性：召回不准时，能否知道是向量、BM25、RRF、父块扩展还是术语表出了问题。
3. 低风险召回改进：在不引入大规模模型或复杂 reranker 的前提下，用已有 metadata、Rust 术语和文档结构提升候选排序。

这个判断决定了后续顺序：先收敛 pipeline，再修上下文扩展和索引健康检查，然后才做召回调优。否则很容易在一个入口调好了，另一个入口仍然走旧逻辑。

## 2. 工程决策与原因

### 2.1 统一 RAG 查询入口

对应提交：`ffc909d Unify RAG query pipeline`

原先 `scripts/query.py`、`scripts/app.py`、`scripts/evaluate.py` 和 evaluator 中有重复的检索、父块扩展、术语表、生成拼装逻辑。重复入口会带来两个风险：

- 调参或修 bug 只影响一个入口，评估结果和实际使用结果不一致。
- 后续新增诊断、配置或缓存时，需要在多个脚本里重复改动。

因此新增 `src/pipeline.py`，把加载 retriever、查询、父块扩展、术语表补充、生成等路径统一为共享 `RAGPipeline`。CLI、Web app 和评估脚本改为调用同一 pipeline。

工程经验：

- RAG 项目最怕“评估路径”和“线上路径”分叉。只要分叉，评估数字就会逐渐失真。
- 统一入口不是重构洁癖，而是后续优化可验证的基础设施。

验证：

- 新增 `tests/test_pipeline.py`。
- 调整 hybrid retriever 相关测试，确保共享 pipeline 下行为稳定。

### 2.2 父块扩展以命中位置为中心

对应提交：`15029b9 Center parent context expansion`

检索命中 child chunk 后，系统会扩展到 parent chunk，并可拼接相邻父块。原逻辑更容易从文档开头扩展，导致真正命中的上下文被稀释。对于 Rust Book 这种长章节文档，错位上下文会直接影响答案忠实度。

改动后，父块扩展以命中的 parent 为中心，按左右相邻块补上下文，并受 `max_chars` 控制。

工程经验：

- RAG 的“召回到了”不等于“给 LLM 的上下文对了”。chunk 命中后的上下文重组同样是召回质量的一部分。
- 对长文档按文档开头扩展，常常会制造看似相关但实际偏题的上下文。

验证：

- 新增 `test_expand_to_parents_centers_on_hit_parent`，证明扩展包含命中块附近内容，并排除距离过远的前置块。

### 2.3 增加索引完整性检查

对应提交：`2df5a9d Add index integrity checker`

索引产物包括 `children.json`、`all_chunks.json`、`faiss.index`、`bm25.pkl`、glossary 子索引。只要其中一个过期或数量不一致，查询可能还能跑，但结果会悄悄变坏。

因此增加 `scripts/check_index.py`，检查索引文件存在性、chunk 数量、FAISS 向量数量、BM25 文档数、父子块引用、metadata 基本字段等。

工程经验：

- RAG 系统的“构建成功”不等于“索引一致”。索引健康检查应该成为每次重建后的固定动作。
- 对二进制索引和 JSON 元数据分离存储的系统，数量一致性检查收益很高。

验证：

- 新增 `tests/test_check_index.py`，覆盖正常索引、缺失文件、数量不一致等情况。

### 2.4 缓存术语表检索器

对应提交：`9eb2ce5 Cache glossary retrievers`

术语表用于补充 Rust 概念解释。若每次查询都重复构建 glossary 检索器，会增加启动和查询成本，也会让性能诊断更混乱。

因此在 `src/glossary.py` 中缓存 glossary retriever，避免重复加载。

工程经验：

- 对 RAG 来说，术语表这类辅助检索器通常是只读静态资源，适合缓存。
- 性能优化优先做“确定不会改变语义”的缓存，比直接换模型或加并发更稳。

验证：

- 新增 `tests/test_glossary.py`，验证缓存复用和空输入等边界行为。

### 2.5 外置运行配置

对应提交：`17d993d Externalize runtime configuration`

运行配置散落在脚本和模块里，会导致本地、评估、Web app 的参数不一致。项目中涉及模型路径、索引目录、host/port、生成参数、检索参数等，适合统一配置读取。

因此新增 `src/config.py` 和 `.env.example`，让运行参数可以通过环境变量覆盖，同时保留默认值。

工程经验：

- 当项目进入“可重复实验”阶段，配置必须从代码里拿出来。
- `.env.example` 是协作成本最低的配置文档，比口头说明可靠。

验证：

- 新增 `tests/test_config.py`，验证默认值、环境变量覆盖和类型转换。

## 3. 针对召回不准的三步改进

用户明确提出“召回不准”后，我没有直接调大 top_k 或换模型，而是先补可观测性，再做低风险排序改进。

### 3.1 增加检索诊断报告

对应提交：`d808054 Add retrieval diagnostics report`

新增 `scripts/diagnose_retrieval.py`，对单条 query 输出：

- BM25 top results
- Vector top results
- RRF document-level breakdown
- source、heading、morphology、expected term 命中情况

这个工具的价值在于把“召回不准”拆成可定位问题。例如 `rust有哪些关键字` 的诊断显示：向量侧能较好命中关键字附录，但 BM25 和基础 RRF 容易把泛化章节排上来。

工程经验：

- 没有诊断工具时，召回优化基本是在猜。
- RAG 调优要能回答三个问题：候选有没有进池、在哪个检索器进池、融合后为什么掉了。

验证：

- 新增 `tests/test_diagnose_retrieval.py`。
- 用真实查询 `rust有哪些关键字` 做 smoke test，确认诊断脚本能揭示 BM25、向量、RRF 的排序差异。

### 3.2 向 BM25 tokenizer 注入 Rust 和 Glossary 术语

对应提交：`5b86a74 Inject Rust terms into BM25 tokenizer`

中文 Rust 文档里有大量混合术语：`trait`、`lifetime`、`fn`、`match`、`borrow checker`、`所有权`、`生命周期` 等。默认中文分词对这些技术词并不总是友好。BM25 若不能稳定保留这些词，关键词类查询就会弱。

改动包括：

- 在 BM25 tokenizer 初始化时注入 Rust 关键词和常见术语。
- 注入 glossary 术语。
- 对带空格的英文短语，在文本中出现时额外追加 phrase token。

工程经验：

- 技术文档 RAG 不能完全依赖通用分词。领域词表是 BM25 成败的关键部分。
- 向 tokenizer 注入术语属于低风险改动，因为它提升的是稀疏检索对领域词的可见性。

验证：

- 扩展 `tests/test_bm25_store.py`。
- 全量测试通过。
- 之后已执行重建索引，使 tokenizer 改动进入 `vectorstore/bm25.pkl`。

### 3.3 给 RRF 增加轻量 metadata boost

对应提交：`1abf449 Boost RRF with metadata signals`

项目 chunk metadata 中已经有 `headings` 和 `morphology`。这些信息很适合做轻量 rerank：

- 查询命中 heading 时，小幅加分。
- `有哪些 / 列表 / 关键字 / 保留字` 等列表意图，偏好 `structured`。
- `代码 / 示例 / 如何 / 函数` 等代码意图，偏好 `code_unit`。
- `区别 / 对比 / 比较` 等比较意图，偏好 `narrative` 或 `mixed`。

加分幅度刻意保持很小：

- heading：`0.003`
- structured list：`0.002`
- code intent：`0.002`
- comparison：`0.001`

这样它不会取代向量和 BM25，只在候选基础分接近时利用文档结构做决策。

工程经验：

- RRF 是很好的第一阶段融合，但它不知道文档结构。heading 和 morphology 是“免费信号”，应该利用。
- boost 不宜过大。RAG 排序的主信号仍应来自检索器，metadata 只做 tie-breaker 或轻量纠偏。
- 中文 query 没有空格，heading 匹配不能只靠 whitespace split，所以加入常见中文 marker 子串匹配。

验证：

- 扩展 `tests/test_hybrid_retriever.py`，覆盖列表意图、代码意图和中文 heading 命中。
- 局部测试 `14 passed`。
- 全量测试 `109 passed`。

## 4. 索引重建与真实效果验证

在完成 tokenizer 和 RRF boost 后，发现 `vectorstore/bm25.pkl` 时间戳早于 tokenizer 改动。也就是说，代码已经改好，但真实 BM25 索引仍是旧词表构建出来的。

因此执行：

```powershell
$env:PYTHONUTF8='1'
S:\condaEnvs\py312\python.exe scripts/build_index.py
```

第一次直接运行时，Windows GBK 控制台无法输出 emoji 日志，触发 `UnicodeEncodeError`。设置 `PYTHONUTF8=1` 后构建成功。

重建结果：

- 来源：6 个 source
- 总 chunks：6440
- 子块索引：4589
- 父块：1851
- BM25 docs：4589
- Glossary 术语：374
- `vectorstore/bm25.pkl` 更新时间：2026-07-07 16:46

真实查询复测：

- `rust有哪些关键字`
  - top1：`reference > 关键字 > 严格关键字`
- `如何写Rust函数示例`
  - top2：`ch03-03-how-functions-work`
- `Rust所有权和借用有什么区别`
  - top1：`认识所有权`
  - top2：`RefCell<T> 在运行时检查借用规则`
  - top3：`借用`
- `trait是什么`
  - top1：`reference > 术语表 > 本地 trait`
- `生命周期是什么`
  - top1：`rust-by-example > 生命周期`

消融对比显示，metadata boost 对关键字类查询效果明显：

- boost off 时，`rust有哪些关键字` top3 更偏向 `通用编程概念`、`原始标识符` 等泛相关页面。
- boost on 时，关键字附录和 reference keywords 能进入 top1/top2。

## 5. 测试和提交纪律

这轮工作保持了“一项改动一个测试集合，一个提交”的节奏：

- `ffc909d`：统一 pipeline，并新增 pipeline 测试。
- `15029b9`：父块中心扩展，并新增扩展测试。
- `2df5a9d`：索引完整性检查，并新增 checker 测试。
- `9eb2ce5`：glossary 缓存，并新增缓存测试。
- `17d993d`：配置外置，并新增 config 测试。
- `d808054`：检索诊断报告，并新增诊断测试。
- `5b86a74`：BM25 tokenizer 术语注入，并新增 tokenizer 测试。
- `1abf449`：RRF metadata boost，并新增 hybrid retriever 测试。

最近一次完整测试结果：

```text
109 passed, 4 warnings
```

保留的 warnings 是既有测试标记和 pytest cache 权限问题，不影响本轮改动结论。

## 6. 关键工程经验

### 6.1 先统一入口，再优化指标

如果 CLI、Web、评估各走一套逻辑，任何优化都可能只是局部有效。统一 pipeline 是后续所有测试、诊断、调参可信的前提。

### 6.2 召回问题要拆层观察

“答案不准”可能来自：

- query rewrite 不合适
- BM25 没分出领域词
- 向量召回偏泛化
- RRF 融合把强信号冲淡
- child 命中后 parent 扩展错位
- glossary 补充过强或过弱
- LLM 在上下文中抽取失败

诊断脚本让这些问题可以逐层定位。

### 6.3 领域 RAG 要认真对待稀疏检索

Rust 中文文档里混合了中文解释、英文关键字、代码符号和专有概念。向量模型能解决语义相似，但精确术语和关键字查询仍然需要 BM25。词表注入是这类系统的基础优化。

### 6.4 利用已有 metadata，而不是急着上重模型

`headings` 和 `morphology` 已经包含了很强的结构信号。轻量 boost 成本低、可解释、可测试，适合作为 cross-encoder 之前的一层排序修正。

### 6.5 索引产物必须纳入工程流程

代码改了 tokenizer，不重建索引就没有真实效果。之后每次影响 chunk、tokenizer、embedding 或 metadata 的改动，都应该明确是否需要重建索引，并运行索引完整性检查。

### 6.6 Windows 本地环境要固定 UTF-8

构建脚本日志中包含 emoji，Windows 默认 GBK 控制台会报 `UnicodeEncodeError`。建议在启动脚本或文档中固定：

```powershell
$env:PYTHONUTF8='1'
```

或把构建日志改为纯 ASCII/中文无 emoji，以降低环境摩擦。

## 7. 当前成果

当前项目相较接手时，已经具备：

- 单一 RAG pipeline，减少入口漂移。
- 更合理的父块上下文扩展。
- 可自动检查的索引完整性工具。
- 可复用的 glossary retriever 缓存。
- 外置运行配置。
- 单条查询级的检索诊断工具。
- 面向 Rust 领域词的 BM25 tokenizer。
- 利用 heading 和 morphology 的轻量 RRF boost。
- 重建后的最新索引产物。
- 全量单元测试保持通过。

从真实查询看，关键字、生命周期、函数示例、所有权/借用、trait 等典型 Rust 问题的 top results 已经更稳定。

## 8. 后续建议

### 8.1 把 metadata boost 写入诊断报告

当前 `diagnose_retrieval.py` 的 RRF breakdown 仍显示基础 RRF 分数，没有拆出 metadata boost。建议下一步让报告显示：

- base RRF score
- heading boost
- morphology boost
- final score

这样诊断输出会和真实 `HybridRetriever.search()` 完全一致。

### 8.2 重建索引后自动运行 check_index

建议在 `scripts/build_index.py` 完成后自动调用或提示运行：

```powershell
python scripts/check_index.py
```

避免索引文件更新了但数量或 metadata 不一致。

### 8.3 增加一组固定 golden queries

把这次用到的查询固化成轻量回归集：

- `rust有哪些关键字`
- `如何写Rust函数示例`
- `Rust所有权和借用有什么区别`
- `trait是什么`
- `生命周期是什么`

每条定义期望 topN URL 或 expected terms。以后调 tokenizer、chunker、RRF 参数时，先跑这组，防止局部优化伤害核心查询。

### 8.4 继续改善 BM25 对精确附录页的排序

重建后 BM25 已有改善，但 `rust有哪些关键字` 中 BM25 仍把一些泛相关页面排在关键字附录前。后续可以考虑：

- heading token 加权进入 BM25 文档。
- 对 appendix/reference 这类结构页增强标题权重。
- 对 `有哪些/列表` 查询增加 heading/structured 的候选召回，而不只在 RRF 阶段 boost。

### 8.5 明确索引产物是否进入 Git

当前重建索引后 `git status` 没显示 `vectorstore/` 变更，说明索引产物可能被 ignore 或没有纳入版本管理。需要明确策略：

- 若索引可快速重建：不进 Git，只保留构建脚本和数据源。
- 若索引构建成本高：用 release artifact 或外部存储，不建议直接压进普通 Git 提交。

## 9. 总结

这轮工作的核心不是堆功能，而是把 RAG 项目从“能跑”推进到“能诊断、能验证、能稳定迭代”。统一 pipeline 降低了入口漂移，诊断脚本让召回问题可观察，tokenizer 和 metadata boost 则针对 Rust 文档的真实结构做了小而有效的改进。

后续最值得继续投入的方向，是把诊断报告和真实最终排序完全对齐，并建立 golden query 回归集。这样每次优化都能马上回答一个朴素但关键的问题：它到底让哪些问题变好了，又有没有让哪些问题变差。
