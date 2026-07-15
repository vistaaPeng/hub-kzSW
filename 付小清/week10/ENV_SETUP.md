# 环境配置说明

本文档记录 work10 医学科普 RAG 项目的完整环境要求，从零开始搭建。

---

## 1. 基础环境

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows / macOS / Linux | 已在 Windows 10 验证 |
| Python | **3.10 ~ 3.12**（推荐 3.11） | 3.13 可用但部分包兼容性待验证 |
| 磁盘空间 | ~200 MB | 索引 + 依赖，无需下载大模型 |
| 网络 | 需要联网 | 调用 DashScope API |

### 检查 Python 版本

```bash
python --version
# 应输出 Python 3.10.x ~ 3.12.x
```

---

## 2. Python 依赖

### 依赖清单

| 包名 | 版本 | 用途 |
|------|------|------|
| openai | >= 1.30.0 | 调用 DashScope 兼容接口（Embedding + LLM） |
| faiss-cpu | >= 1.7.4 | 本地向量索引与检索 |
| numpy | >= 1.24.0 | 向量运算 |
| rank_bm25 | >= 0.2.2 | BM25 关键词检索 |
| jieba | >= 0.42.1 | 中文分词（BM25 依赖） |

### 安装命令

```bash
cd work10
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import faiss, numpy, openai, rank_bm25, jieba; print('依赖安装成功')"
```

### 与旧方案的区别

本项目**不使用**以下重量级依赖（避免下载 PyTorch ~122MB）：

- ~~sentence-transformers~~
- ~~torch~~
- ~~transformers~~

Embedding 全部通过 DashScope API 完成，安装快、磁盘占用小。

---

## 3. API Key 配置

### 3.1 获取 DashScope API Key

1. 注册阿里云账号：https://www.aliyun.com/
2. 开通 DashScope 服务：https://dashscope.console.aliyun.com/
3. 在「API-KEY 管理」中创建密钥
4. 新用户通常有 **100 万 token 免费额度**

### 3.2 设置环境变量

**Windows PowerShell（当前会话）：**

```powershell
$env:DASHSCOPE_API_KEY = "sk-你的密钥"
```

**Windows CMD：**

```cmd
set DASHSCOPE_API_KEY=sk-你的密钥
```

**Linux / macOS：**

```bash
export DASHSCOPE_API_KEY="sk-你的密钥"
```

**持久化（可选）：**

将上述命令写入系统环境变量，或复制 `.env.example` 为 `.env` 后手动加载。

### 3.3 验证 API Key

```bash
python -c "
import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
resp = client.chat.completions.create(
    model='qwen-turbo',
    messages=[{'role':'user','content':'你好'}],
    max_tokens=10
)
print('API 连接成功:', resp.choices[0].message.content)
"
```

---

## 4. 使用的模型

| 用途 | 模型 | 接口 | 单次费用 |
|------|------|------|---------|
| 文本向量化 | text-embedding-v3 | Embedding API | ~0.0007 元/千 token |
| 答案生成 | qwen-plus | Chat API | ~0.004 元/千 token |

### API 端点

```
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
```

使用 OpenAI 兼容格式，可用 `openai` Python SDK 直接调用。

---

## 5. 运行流程

### 步骤 1：构建知识库索引

```bash
python src/build_index.py
```

**做了什么：**
1. 读取 `data/docs/` 下 13 篇 Markdown 文档
2. 按标题和段落切分为文本块（约 50-80 个 chunk）
3. 调用 DashScope Embedding API 向量化
4. 构建 FAISS 索引，保存到 `vectorstore/`

**预期输出：**
```
加载文档: 00_disclaimer.md (xxx 字符)
加载文档: 01_common_cold.md (xxx 字符)
...
共生成 xx 个文本块
Embedding 进度: 1/x 批
FAISS 索引已保存 → vectorstore/faiss.index (xx 条)
索引构建完成！
```

**耗时：** 约 30 秒 ~ 1 分钟
**费用：** 约 0.01 元

### 步骤 2：命令行问答

```bash
# 单次提问
python src/qa.py --query "感冒有哪些主要症状"

# 仅检索（不调用 LLM，省钱）
python src/qa.py --query "头痛" --retrieve-only

# 交互模式
python src/qa.py
```

**耗时：** 单次约 2-5 秒
**费用：** 约 0.003-0.005 元/次

---

## 6. 费用汇总

| 操作 | 预估费用 | 说明 |
|------|---------|------|
| 首次构建索引 | ~0.01 元 | 仅需一次 |
| 单次问答 | ~0.003-0.005 元 | 1 次 Embedding + 1 次 LLM |
| 跑 demo 5 题 | ~0.02 元 | |
| 测试 20 题 | ~0.1 元 | |
| 课程作业全流程 | < 0.5 元 | 含反复调试 |

---

## 7. 常见问题

### Q: `ModuleNotFoundError: No module named 'faiss'`

```bash
pip install faiss-cpu
```

### Q: `请设置环境变量 DASHSCOPE_API_KEY`

确认已执行 `$env:DASHSCOPE_API_KEY = "sk-xxx"`，且 Key 有效。

### Q: `向量索引不存在`

先运行 `python src/build_index.py` 构建索引。

### Q: Embedding API 报错 `batch size is invalid`

text-embedding-v3 单批上限 10 条，`config.py` 中 `EMBED_BATCH_SIZE = 10` 已设置。

### Q: 回答"检索相关度较低"

知识库中没有相关内容，或问题表述与文档差异太大。可尝试换个问法。

### Q: 需要重新构建索引吗？

以下情况需要重新运行 `build_index.py`：
- 修改了 `data/docs/` 中的文档
- 更换了 Embedding 模型或维度
- 删除了 `vectorstore/` 目录

---

## 8. 目录与文件说明

```
work10/
├── data/
│   ├── docs/               # 知识库原始文档（可编辑）
│   └── chunks.json         # 分块结果（build_index 生成）
├── vectorstore/
│   ├── faiss.index         # FAISS 向量索引
│   └── meta.json           # 文本块元数据
├── src/
│   ├── config.py           # 模型、参数、Prompt 配置
│   ├── build_index.py      # 索引构建脚本
│   ├── rag.py              # RAG 核心逻辑
│   └── qa.py               # 命令行入口
├── requirements.txt        # pip 依赖
├── .env.example            # API Key 模板
├── ENV_SETUP.md            # 本文件
└── README.md               # 项目说明
```
