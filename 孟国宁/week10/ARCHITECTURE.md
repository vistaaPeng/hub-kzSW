# 技术架构说明

> 学术文献 RAG 问答系统 — 整体方案、选型决策与设计原理

---

## 一、项目定位

本项目以"学术文献智能问答"为场景，构建一套接近企业级落地标准的 RAG（检索增强生成）系统。数据来源于用户上传的学术论文 PDF（计算机科学、人工智能等领域），支持多篇文献联合检索和问答。

项目同时提供两套实现：

| 实现版本 | 定位 | 关键差异 |
|----------|------|---------|
| **原生版**（`src/`） | 企业级生产参考 | 手动控制每个环节，混合检索 + Rerank |
| **LangChain 版**（`src_langchain/`） | 框架快速原型参考 | 用框架抽象链路，聚焦 LCEL 链路设计 |

---

## 二、整体流水线

```
文献 PDF（用户导入到 data/raw_pdf/）
    │
    ▼ download_literature.py
文献导入（从源目录复制/扫描 raw_pdf）
    │
    ▼ parse_pdf.py
PDF 解析（文字 + 表格 + OCR + 章节结构）
    │
    ▼ chunk_documents.py
文档分块（三种策略可切换）
    │
    ▼ build_index.py
向量化 + 索引构建（DashScope API / 本地 BGE）
    │
    ▼ rag_pipeline.py
问答流水线：查询 → 检索 → 重排 → 生成
    │
    ▼ evaluation/
评估（RAGAS 四项指标 + 消融实验）
```

---

## 三、各环节技术选型

### 3.1 数据获取

**用户本地导入**（`download_literature.py`）

功能：
- 从用户指定的源目录复制 PDF 到 `data/raw_pdf/`
- 自动解析文件名提取元信息（作者、年份、标题）
- 生成 `data/manifest.json` 索引清单

### 3.2 PDF 解析

**组合策略**：pdfplumber（表格）+ PyMuPDF/fitz（文字+字体信息）+ pytesseract（OCR，可选）

| 库 | 职责 | 选型原因 |
|----|------|---------|
| `pdfplumber` | 表格提取 | 基于规则的表格算法，对论文表格识别更准确 |
| `PyMuPDF(fitz)` | 文字+字体元数据 | 提供每个 span 的字体大小和加粗信息，用于识别标题层级 |
| `pytesseract` | 扫描页 OCR | 部分论文原件为扫描件，需 OCR 降级处理 |

解析输出的三种块类型：
- `title`：识别依据是字体大于 13pt 或加粗且行不长，或匹配编号模式（1. / 1.1 / 1.1.1）
- `table`：直接转为 Markdown 格式
- `text`：正常文本段落

### 3.3 文档分块（Chunking）

提供三种策略：

#### 策略 A：固定大小分块（`fixed`）
- 简单可预测，但无视句子/段落边界

#### 策略 B：语义分块（`semantic`）—— 默认
- 遇标题强制切块，段落合并不超过 800 字符

#### 策略 C：层级分块（`hierarchical`）
- 父块（~1500字符）用于 LLM 上下文，子块（~300字符）用于精确匹配

### 3.4 Embedding 模型

**原生版**：DashScope text-embedding-v3（API，维度 1024）
**LangChain 版**：本地 BAAI/bge-small-zh-v1.5（维度 512，~90 MB）

### 3.5 向量库

**FAISS IndexFlatIP**（精确内积检索）
- 向量已 L2 归一化，内积等价于余弦相似度

### 3.6 检索策略

- **向量检索**：语义相似，理解同义词和近义表达
- **BM25 关键词检索**：精确匹配学术术语、模型名、参数数字
- **RRF 混合融合**：互补两路检索的盲区
- **CrossEncoder Rerank**（可选）：对候选集二次精排

### 3.7 LLM 生成

**DashScope qwen-plus**，temperature=0.1

系统提示核心约束：
1. 只从参考资料中回答，不编造数据
2. 引用时标注来源编号（如 `[1]`）
3. 资料不足时主动拒绝回答

### 3.8 LangChain LCEL 链

```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 四、评估体系

### 4.1 评测题集（20 题）

| 类型 | 题数 | 示例 |
|------|------|------|
| `simple_fact` | 5 | Transformer 的自注意力机制？ |
| `precise_number` | 4 | BERT Base 有多少参数？ |
| `cross_doc_compare` | 6 | Transformer vs BERT 架构区别 |
| `time_trend` | 3 | 从 Transformer 到 GPT 的发展趋势 |
| `should_refuse` | 2 | 作者联系方式？（应拒绝）|

### 4.2 RAGAS 四项指标

Faithfulness / Answer Relevancy / Context Precision / Context Recall

### 4.3 消融实验矩阵

```
分块策略（3）× 检索方式（3）= 9 种组合
```

---

## 五、目录结构

```
week10/
├── src/                         # 原生版（DashScope API）
│   ├── download_literature.py   # 文献导入
│   ├── parse_pdf.py             # PDF 解析
│   ├── chunk_documents.py       # 文档分块
│   ├── build_index.py           # 向量索引
│   ├── rag_pipeline.py          # 问答流水线
│   ├── serve.py                 # FastAPI 服务
│   ├── .env                     # API 密钥
│   └── static/index.html        # Web 界面
│
├── src_langchain/               # LangChain 版
│   ├── download_model.py        # BGE 模型下载
│   ├── build_index_lc.py        # PDF → FAISS
│   └── rag_chain_lc.py          # LCEL RAG 链
│
├── evaluation/
│   ├── questions.json           # 20 道测试题
│   ├── evaluate.py              # RAGAS 评估
│   └── compare_strategies.py    # 消融实验
│
├── data/
│   ├── raw_pdf/                 # 文献 PDF
│   ├── manifest.json
│   ├── parsed/                  # 解析后的 JSON
│   └── chunks/                  # 分块后的 JSON
│
├── vectorstore/                 # 向量索引
├── models/                      # 本地模型
├── requirements.txt
├── ARCHITECTURE.md
└── USAGE_GUIDE.md
```
