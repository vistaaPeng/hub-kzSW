# 基于PDF的知识问答系统项目报告

## 一、项目概述

本项目实现了一个基于RAG（Retrieval-Augmented Generation）技术的知识问答系统，能够从PDF文档中提取信息并回答用户问题。系统使用 **transformers** 库的BERT模型进行密集向量编码，结合BM25关键词检索实现混合检索。

### 1.1 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接口层                               │
│              CLI命令行 / Flask HTTP API                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG 流水线                               │
│   问题输入 → 混合检索(RRF) → 上下文构建 → LLM生成 → 答案输出     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│    向量检索模块     │         │     BM25检索模块     │
│  BERT + Cosine      │         │   BM25关键词匹配     │
│  (768维密集向量)    │         │                     │
└─────────────────────┘         └─────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        索引存储层                               │
│   dense_index.npy | dense_meta.json                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据处理层                               │
│   PDF解析 → 语义分块 → 索引构建                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 二、核心模块

### 2.1 PDF解析模块 (`src/parse_pdf.py`)

**功能**：从PDF文件中提取文本内容，识别标题和正文

**技术实现**：
- 使用 `PyPDF2` 库进行PDF文本提取
- 基于规则的标题识别：以"第"开头且包含"章"的行视为标题
- 支持大写英文标题识别
- 输出结构化的JSON格式数据

**处理流程**：
```python
PDF文件 → 逐页读取 → 按行分割 → 标题/正文分类 → JSON输出
```

### 2.2 文档分块模块 (`src/chunk_documents.py`)

**功能**：将解析后的文档分成适合检索的语义块

**分块策略**：
- 语义分块：以章节标题为边界进行分块
- 块大小控制：最大800字符
- 保留元数据：文档类型、标题、页码等

**输出格式**：
```json
{
  "chunk_id": "ai_0001",
  "content": "...",
  "metadata": {
    "doc_type": "ai",
    "title": "ai basics",
    "page_num": 1,
    "source_file": "ai_basics.pdf"
  }
}
```

### 2.3 索引构建模块 (`src/build_index.py`)

**功能**：使用BERT模型构建密集向量索引，支持语义检索

**技术实现**：
- 使用 `transformers` 库的 `AutoTokenizer` 和 `AutoModel`
- 模型：`bert-base-chinese`（768维输出）
- 取 `[CLS]` token的向量作为文本表示
- 批量编码提高效率（batch_size=8）
- 密集向量存储（numpy数组）

**编码流程**：
```python
文本 → Tokenize → BERT编码 → [CLS]向量 → 768维嵌入
```

**索引文件**：
| 文件 | 用途 |
|------|------|
| `dense_index.npy` | 密集向量矩阵（n × 768） |
| `dense_meta.json` | 块元数据 |

### 2.4 RAG流水线模块 (`src/rag_pipeline.py`)

**功能**：核心问答逻辑，结合BERT向量检索和LLM生成

**检索策略**：
1. **BERT向量检索**：问题编码后与索引向量计算余弦相似度
2. **BM25检索**：基于关键词的传统检索
3. **RRF融合**：Reciprocal Rank Fusion 融合两个检索结果

**融合算法**：
```
RRF(cid) = Σ(1 / (k + rank))  # k=60
```

**LLM集成**：
- 使用阿里云通义千问 `qwen-plus`
- 支持环境变量 `DASHSCOPE_API_KEY` 配置
- 参考资料引用标注

### 2.5 HTTP服务模块 (`src/serve.py`)

**功能**：提供RESTful API接口

**接口定义**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/query` | POST | 问答查询 |
| `/api/status` | GET | 服务状态 |

**请求示例**：
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能"}'
```

## 三、数据文件

### 3.1 原始PDF文档

系统包含4个领域的示例PDF文档：

| 文件名 | 主题 | 内容概述 |
|--------|------|----------|
| `ai_basics.pdf` | 人工智能基础 | AI定义、发展阶段、机器学习、深度学习、应用领域 |
| `climate_change.pdf` | 气候变化 | 全球变暖原因、影响、应对措施、碳中和目标 |
| `python_programming.pdf` | Python编程 | 语法基础、数据结构、面向对象、常用库 |
| `space_exploration.pdf` | 太空探索 | 航天历史、太阳系、火星探测、未来展望 |

### 3.2 数据目录结构

```
data/
├── raw_pdf/          # 原始PDF文件
│   ├── ai_basics.pdf
│   ├── climate_change.pdf
│   ├── python_programming.pdf
│   └── space_exploration.pdf
├── parsed/           # 解析后的JSON
│   ├── ai_basics.json
│   ├── climate_change.json
│   ├── python_programming.json
│   └── space_exploration.json
└── chunks/           # 分块结果
    └── all_semantic.json

vectorstore/          # 向量索引
├── dense_index.npy   # BERT密集向量索引
└── dense_meta.json   # 块元数据

model_cache/          # BERT模型缓存
└── bert-base-chinese/
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── pytorch_model.bin
```

## 四、运行方法

### 4.1 环境依赖

```
Python >= 3.8
transformers >= 4.0
torch >= 1.0
PyPDF2 >= 2.0
Flask >= 2.0
reportlab >= 3.6  (用于生成示例PDF)
```

### 4.2 启动步骤

**步骤1：解析PDF**
```bash
python src/parse_pdf.py
```

**步骤2：文档分块**
```bash
python src/chunk_documents.py
```

**步骤3：构建索引（使用BERT）**
```bash
python src/build_index.py
```

**步骤4：启动服务**
```bash
python src/serve.py
```

**步骤5：命令行查询**
```bash
python src/rag_pipeline.py --query "什么是人工智能"
```

### 4.3 配置LLM（可选）

如需使用大模型生成答案，设置环境变量：
```bash
set DASHSCOPE_API_KEY=your_api_key
```

## 五、系统特性

### 5.1 技术特点

1. **基于Transformer**：使用BERT模型进行语义编码，支持深层语义理解
2. **混合检索**：结合BERT向量检索和BM25关键词检索
3. **RRF融合**：提升检索准确性
4. **中文支持**：使用 `bert-base-chinese` 模型，支持中英文混合文本
5. **来源标注**：回答中引用具体来源

### 5.2 扩展性

- **添加新文档**：将PDF放入 `data/raw_pdf/`，重新运行解析和索引构建
- **更换模型**：可替换为其他预训练模型（如 `roberta-base-chinese`）
- **自定义LLM**：修改 `rag_pipeline.py` 中的 `call_llm` 函数

## 六、测试结果

### 6.1 示例问答

**问题**：什么是人工智能

**检索结果**：
- 向量召回：10条，最高分=437.110
- BM25召回：10条
- RRF融合后：17条
- 最终使用：4条上下文

**来源**：
1. ai basics · 第1页
2. ai basics · 第一章：什么是人工智能
3. ai basics · 第五章：人工智能的未来展望
4. python programming · 第1页

### 6.2 性能指标

| 指标 | 值 |
|------|-----|
| 文档数量 | 4 |
| 总块数 | 36 |
| 平均块长度 | 97字符 |
| 向量维度 | 768（BERT输出） |
| 检索速度 | < 500ms（含模型加载） |

## 七、总结

本项目成功实现了一个基于PDF的知识问答系统，具备以下特点：

1. **使用Transformer**：基于 `bert-base-chinese` 模型进行语义编码，支持深层语义理解
2. **完整的技术栈**：从PDF解析到问答服务的端到端解决方案
3. **模块化设计**：清晰的模块划分，易于扩展和维护
4. **混合检索策略**：结合BERT向量检索和BM25关键词检索，提高准确性
5. **中文友好**：支持中英文混合文档的处理

系统可作为学习RAG技术和Transformer应用的入门项目，也可作为企业知识库问答系统的基础框架。