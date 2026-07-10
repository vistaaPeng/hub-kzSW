# 技术架构说明

> 电动自行车国家标准 + 道路交通法律 RAG 问答系统 — 方案、选型与设计原理
> 参照 `rag_annual_report` 原生版实现，**不含 LangChain 版**。

---

## 一、项目定位

将一套企业级 RAG（检索增强生成）流水线应用到「电动自行车相关国家标准 + 道路交通法律」领域的问答。
数据为本地 PDF（无需联网下载），元数据以 **文档号 / 标题 / 类型** 为核心。

语料（`data/raw_pdf/`）：

| doc_id | 文件 | 类型 |
|--------|------|------|
| `GB17761-2024` | 电动自行车安全技术规范 | 标准 |
| `GB43854-2024` | 电动自行车用锂离子蓄电池安全技术规范 | 标准 |
| `LAW-中华人民共和国道路交通安全法` | 中华人民共和国道路交通安全法 | 法律 |

> `GB+811-2022.pdf`（头盔标准）为 CID 字体乱码、无文本层且无图片，任何文本提取器均无法读取，已在 `build_manifest.py` 中排除（需 OCR 才能纳入）。

后端：阿里云 **DashScope**，`text-embedding-v3`（向量）+ `qwen-plus`（生成），单一 `DASHSCOPE_API_KEY`。

---

## 二、整体流水线

```
本地 PDF
  ▼ build_manifest.py   扫描 raw_pdf → manifest.json（doc_id/doc_title/doc_type）
  ▼ parse_pdf.py        PDF → 结构化 blocks（标题/条款/表格 + 页码 + 章节路径）
  ▼ chunk_documents.py  blocks → chunks（fixed / semantic / hierarchical）
  ▼ build_index.py      chunks → DashScope embedding → FAISS 索引 + meta
  ▼ rag_pipeline.py     查询 →(改写)→ 向量+BM25 → RRF →(Rerank)→ 阈值 → qwen-plus 生成
  ▼ serve.py            FastAPI：/query、/query/debug、/health、/ 可视化
  ▼ evaluation/         RAGAS 四指标 + 消融实验（3 策略 × 3 检索）
```

---

## 三、各环节技术选型

### 3.1 数据清单（build_manifest.py）
替代原版的巨潮下载脚本。扫描本地 `raw_pdf/`，从文件名正则推断：
- `doc_id`：`GB\s*[+]?\s*T?\s*(\d+)[-—.](\d+)` → `GB{号}-{年}`；无 GB 号且含“法”→ `LAW-{文件名}`。
- `doc_type`：含“法”→`法律`，否则`标准`。
- `doc_title`：清理后的文件名。
- 无法提取文本的文件在 `EXCLUDE_FILES` 中排除。

### 3.2 PDF 解析（parse_pdf.py）
组合策略：pdfplumber（表格）+ PyMuPDF/fitz（文字+字体）+ 可选 OCR（未装 tesseract 时降级为占位符）。

针对标准/法律做的适配：
- **标题/条款识别**：匹配 `第X章/第X节/第X条`、条款号 `4.1 / 4.1.2`、顶层编号 `5 要求`、`附录A / A.1` 等。
- **动态正文字号阈值**：先全文统计出「正文主字号」，仅当某行字号明显大于正文（≥ body+1.5）才按字体判为标题——解决了法律 PDF 正文本身是大字号（15.9pt）导致「整篇都被判成标题」的问题。
- **噪声过滤**：标准号页脚 `GB xxxxx—yyyy`、独立页码、全角/半角破折号页码 `－3－`、`ICS/CCS` 封面代号。
- **乱码检测**：`is_garbled()` 用汉字占比判断 CID 乱码页，走 OCR 降级。
- 章节路径栈 `_update_section` 按编号层级维护层次。

块类型：`title` / `table`（转 Markdown）/ `text`，每块保留 `page_num`、`section_path`、`is_ocr`。

### 3.3 文档分块（chunk_documents.py）
三种策略，`STRATEGY` 变量切换：
- `fixed`：500 字符定长、overlap 50（baseline）。
- `semantic`（默认）：遇标题强制切、表格单独成块、文字累积到 800 字符。
- `hierarchical`：父块（~2000）+ 子块（~400），Small-to-Big 检索。

每 chunk 元数据：`doc_id / doc_title / doc_type / page_num / section / block_types / is_ocr / strategy / source_file`（层级分块另含 `parent_id / parent_content`）。

实际产出：semantic 218 / fixed 158 / hierarchical 216 chunks。

### 3.4 Embedding（build_index.py）
- 模型 `text-embedding-v3`，维度 1024，批次上限 10（DashScope 硬限制）。
- 向量 L2 归一化 → 内积等价余弦相似度。
- **中文路径修复**：FAISS 的 C++ `write_index/read_index` 无法处理含中文的路径（本项目路径含“检索增强生成”），改为 `faiss.serialize_index → bytes` 由 Python 写文件、读时 `bytes → faiss.deserialize_index`。

### 3.5 向量库
FAISS `IndexFlatIP`（精确内积检索）。产出 `vectorstore/faiss_index.bin` + `faiss_meta.json`；消融另存 `faiss_fixed/`、`faiss_hierarchical/`。

### 3.6 检索策略（rag_pipeline.py）
- **向量检索**：查询向量 + FAISS，可按 `doc_id / doc_type` 过滤。
- **BM25**：jieba 分词 + `BM25Okapi`，擅长精确匹配术语/条款号/数值。
- **RRF 融合**：`score = Σ 1/(k+rank)`，k=60。
- **CrossEncoder Rerank**（可选）：`bge-reranker-base`，未装 sentence-transformers/模型时自动降级。
- **阈值拒答**：top vec_score < 0.25 且无过滤时拒答。

### 3.7 LLM 生成
`qwen-plus`，temperature 0.1，top-4 chunk 入上下文。系统提示约束：只据参考资料回答、标注来源编号、条款/数值精确、资料不足则拒答。查询改写用 `qwen-turbo`（可选）。

来源标签格式：`[i] {doc_title}（{doc_id}） · {section} · 第{page}页`。

---

## 四、评估体系

### 4.1 评测题集（questions.json，15 题）
| 类型 | 题数 | 考察 |
|------|------|------|
| simple_fact | 5 | 术语/定义/条款直接检索 |
| precise_number | 4 | 限值/参数（车速、质量、时限、罚则）——BM25 优势 |
| cross_doc_compare | 3 | 跨文件综合 |
| should_refuse | 3 | 幻觉控制（品牌/价格/出行建议） |

### 4.2 RAGAS 四指标（evaluate.py）
Faithfulness / Answer Relevancy / Context Precision / Context Recall。
打分 LLM 与 embedding **统一用 DashScope**（qwen-plus + text-embedding-v3），单一 key。
> 依赖注意：RAGAS 与 langchain 版本强耦合，本项目锁定 `ragas==0.2.15` + `langchain 0.3.x`（见 requirements.txt）；新版 `langchain-community` 会破坏 ragas 导入。

### 4.3 消融实验（compare_strategies.py）
3 分块策略 × 3 检索方式，指标 Hit Rate@4 / MRR（不依赖 LLM，快）。命中判定基于 `doc_id ∈ target_docs`。

---

## 五、目录结构

```
homework/
├── src/
│   ├── build_manifest.py     # 扫描本地 PDF 生成 manifest
│   ├── parse_pdf.py          # PDF → 结构化 blocks
│   ├── chunk_documents.py    # 三种分块策略
│   ├── build_index.py        # DashScope embedding + FAISS
│   ├── rag_pipeline.py       # 向量+BM25+RRF+Rerank+LLM
│   ├── serve.py              # FastAPI 服务
│   └── static/index.html     # 可视化页面
├── evaluation/
│   ├── questions.json        # 15 道评测题
│   ├── evaluate.py           # RAGAS 四指标
│   └── compare_strategies.py # 消融实验
├── data/
│   ├── raw_pdf/              # 原始 PDF
│   ├── manifest.json         # 文档清单
│   ├── parsed/               # 解析 JSON
│   └── chunks/               # 分块 JSON（all_{strategy}.json）
├── vectorstore/             # FAISS 索引 + meta（含 faiss_fixed/、faiss_hierarchical/）
├── requirements.txt
├── ARCHITECTURE.md          # 本文档
└── USAGE_GUIDE.md
```

---

## 六、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| 法律 PDF 整篇被判为标题 | 正文本身是 15.9pt，`fontsize>=14` 绝对阈值全命中 | 动态检测正文主字号，用相对阈值 body+1.5 |
| GB811 提取全是乱码 | CID 字体无 ToUnicode 映射、无图片 | `is_garbled` 检测并排除该文件 |
| FAISS 写/读索引报 No such file | C++ fopen 不支持含中文的路径 | serialize/deserialize + Python 文件 IO |
| RAGAS 导入报 vertexai 缺失 | ragas 0.4 与新版 langchain-community 不兼容 | 锁定 ragas 0.2.15 + langchain 0.3.x |
| 页码 `－3－` 未过滤 | 噪声正则只含半角 `—` | 增加全角破折号页码模式 |
