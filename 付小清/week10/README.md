# 第十周作业：医学科普 RAG 问答系统

基于自建医学知识库的检索增强生成（RAG）问答系统，支持命令行交互问答。

> **作业提交**：详见 [作业提交说明.md](作业提交说明.md)（含原理说明、运行方法、效果截图）

> **免责声明**：本系统为课程作业演示项目，回答仅供参考，不能替代医生诊断和治疗。

## 项目概述

| 项目 | 说明 |
|------|------|
| 主题 | 医学科普 — 常见疾病 + 症状自查指南 |
| 知识库 | 13 篇 Markdown 文档（7 种疾病 + 5 种症状指南 + 免责声明） |
| Embedding | DashScope text-embedding-v3（云端 API） |
| 向量库 | FAISS IndexFlatIP |
| LLM | DashScope qwen-plus |
| 交付形式 | 命令行问答 |

## 技术架构

```
用户提问
  ↓
DashScope Embedding（text-embedding-v3）
  ↓
向量检索 Top-10 + BM25 关键词检索 Top-10
  ↓
RRF 融合排名 → Top-4
  ↓
相关性阈值过滤
  ↓
qwen-plus 生成答案 + 来源标注 + 免责声明
```

## 快速开始

详细环境配置见 [ENV_SETUP.md](ENV_SETUP.md)。

```bash
# 1. 安装依赖（轻量，无需下载 PyTorch）
pip install -r requirements.txt

# 2. 配置 API Key
$env:DASHSCOPE_API_KEY = "sk-你的密钥"    # PowerShell

# 3. 构建索引
python src/build_index.py

# 4. 问答
python src/qa.py --query "感冒有哪些症状"
```

## 项目结构

```
work10/
├── data/docs/          # 医学知识库（13 篇 Markdown）
├── vectorstore/        # FAISS 索引（build_index.py 生成）
├── src/
│   ├── config.py       # 全局配置
│   ├── build_index.py  # 索引构建
│   ├── rag.py          # RAG 流水线
│   └── qa.py           # 命令行入口
├── requirements.txt             # Python 依赖
├── ENV_SETUP.md                 # 环境配置详细说明
├── 作业提交说明.md               # 作业提交文档（原理 + 运行 + 效果）
├── demo_output.txt              # 运行效果示例
├── run_demo.bat                 # 一键演示脚本
├── .env.example                 # API Key 配置示例
└── README.md
```

## 示例问题

```bash
python src/qa.py --query "高血压的诊断标准是什么"
python src/qa.py --query "突发剧烈胸痛应该怎么办"
python src/qa.py --query "咳嗽超过3周不好可能是什么原因"
python src/qa.py    # 交互模式，输入 demo 运行预设问题
```

## 费用估算

| 步骤 | 费用 |
|------|------|
| 索引构建（~50 个 chunk） | ~0.01 元 |
| 单次问答 | ~0.003-0.005 元 |
| 测试 20 题 | ~0.1 元 |

DashScope 新用户有免费额度，详见 [ENV_SETUP.md](ENV_SETUP.md)。
