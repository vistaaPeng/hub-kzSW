# 文献 RAG 问答系统 — 使用指南

## 一、环境准备

### 1.1 安装依赖

```bash
cd 项目目录
pip install -r requirements.txt
```

### 1.2 配置 API 密钥

本项目需要阿里云 DashScope API Key。

```bash
export DASHSCOPE_API_KEY="sk-xxx"  # Linux/Mac
set DASHSCOPE_API_KEY=sk-xxx       # Windows
```

也可以直接编辑 `src/.env` 文件。

> 如需获取 API Key：https://dashscope.aliyun.com/

---

## 二、数据准备

### 2.1 导入文献 PDF

将学术论文 PDF 放入 `data/raw_pdf/` 目录，或从指定目录导入：

```bash
python src/download_literature.py --source D:/papers/
```

### 2.2 解析 PDF

```bash
python src/parse_pdf.py
```

### 2.3 文档分块

```bash
python src/chunk_documents.py
```

### 2.4 构建向量索引

```bash
python src/build_index.py
```

---

## 三、运行问答

### 3.1 交互式问答

```bash
python src/rag_pipeline.py
```

### 3.2 单次查询

```bash
python src/rag_pipeline.py --query "Transformer的注意力机制是什么"
```

### 3.3 高级选项

```bash
# 查询改写
python src/rag_pipeline.py --query "注意力机制是怎么回事" --query-rewrite

# 限定文献来源
python src/rag_pipeline.py --query "多头注意力" --source-file "Attention_Is_All_You_Need.pdf"

# 消融实验：关闭 BM25
python src/rag_pipeline.py --query "BERT预训练任务" --no-bm25
```

### 3.4 启动 Web 服务

```bash
cd src
uvicorn serve:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 `http://localhost:8000`。

---

## 四、LangChain 版本

```bash
# 下载模型
python src_langchain/download_model.py

# 构建索引
python src_langchain/build_index_lc.py

# 运行问答
python src_langchain/rag_chain_lc.py
python src_langchain/rag_chain_lc.py --query "自注意力机制" --with-sources
```

---

## 五、评估

```bash
# RAGAS 评估
python evaluation/evaluate.py --pipeline native

# 消融实验
python evaluation/compare_strategies.py
```

---

## 六、常见问题

**Q: 没有 DashScope API Key 怎么办？**
A: 可改用其他 OpenAI 兼容的 API，修改 `rag_pipeline.py` 中的 `DASHSCOPE_URL`。

**Q: 如何支持英文论文？**
A: 直接放入 PDF 即可，解析器和检索器均支持中英文。
