# 📚 刑法 RAG 问答系统

> 基于检索增强生成（RAG）技术的《中华人民共和国刑法》智能问答系统

## 项目简介

本项目将《中华人民共和国刑法》PDF 转化为可交互的智能问答系统。你可以用自然语言提问，系统会从刑法中检索相关法条，并由 AI 生成准确、有法条引用的回答。

### 核心功能

| 功能 | 说明 |
|------|------|
| PDF 解析 | 自动识别刑法"编→章→节→条→款→项"的层级结构 |
| 混合检索 | 向量语义检索 + BM25 关键词检索，双路互补，召回更精准 |
| AI 回答 | 基于检索到的法条生成回答，并标注引用来源 |
| 交互式问答 | 命令行交互，支持连续提问 |

---

## 快速开始

### 环境要求

- Python 3.9+
- 阿里云 DashScope API Key（免费申请）

### 安装依赖

```bash
cd D:\PythonStudy\yzh_study\严镇函\week10\项目
pip install -r requirements.txt
```

### 设置 API Key

```bash
set DASHSCOPE_API_KEY=sk-你的API密钥
```

---

## 使用流程

### 第 1 步：准备 PDF

将《中华人民共和国刑法》PDF 放入 `data/raw_pdf/` 或 `data/` 目录。

### 第 2 步：解析 PDF

```bash
python src/parse_pdf.py
```

解析法条结构，输出 → `data/parsed/中华人民共和国刑法.json`

### 第 3 步：分块 + 建索引

```bash
python src/chunk_documents.py
python src/build_index.py
```

输出 → `vectorstore/faiss_index.bin` + `vectorstore/faiss_meta.json`

### 第 4 步：开始问答

```bash
# 单次提问
python src/rag_pipeline.py --query "故意杀人罪判几年"

# 交互式问答
python src/rag_pipeline.py
```

---

## 项目结构

```
项目/
├── data/
│   ├── raw_pdf/         ← 原始刑法 PDF
│   ├── parsed/          ← 解析后的 JSON
│   └── chunks/          ← 分块后的 JSON
├── src/
│   ├── parse_pdf.py     ← PDF 解析
│   ├── chunk_documents.py ← 文档分块
│   ├── build_index.py   ← 向量化 + 建索引
│   └── rag_pipeline.py  ← RAG 问答流水线
├── vectorstore/         ← FAISS 索引 + 元数据
├── requirements.txt     ← 依赖清单
└── README.md            ← 本文件
```

---

## 原理说明

```
用户提问 → ① 向量检索(FAISS) → ② BM25 关键词检索
                ↓                    ↓
              ③ RRF 融合排名
                    ↓
              ④ LLM 生成回答 + 引用标注
                    ↓
              最终答案
```

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| PDF 解析 | PyMuPDF (fitz) |
| 向量模型 | DashScope text-embedding-v3 |
| 向量库 | FAISS (IndexFlatIP) |
| 关键词检索 | rank_bm25 + jieba |
| 大模型 | DashScope qwen-plus |