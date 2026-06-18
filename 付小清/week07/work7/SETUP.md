# 训练前置准备工作指南

本文档记录在 `work7/` 跑 peoples_daily 序列标注训练前，需要完成的环境、依赖、数据和模型准备步骤。

---

## 一、环境要求

| 项目 | 最低要求 | 本机实测 |
|------|---------|---------|
| Python | 3.10+ | 3.12（conda 环境 `py312`） |
| PyTorch | 2.0+，带 CUDA | 2.5.1+cu121 |
| GPU 显存 | ≥ 6GB（batch=16） | GTX 1060 6GB ✓ |
| 磁盘空间 | ~500MB（模型 + 数据） | — |

### 激活 Python 环境

```powershell
# 使用 conda（推荐）
conda activate py312

# 或任意已安装 torch + transformers 的虚拟环境
```

验证 GPU 是否可用：

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 二、安装 Python 依赖

### 2.1 work7 必需库（最小集）

```powershell
pip install torch transformers seqeval pytorch-crf tqdm
```

| 库 | 用途 |
|----|------|
| `torch` | 深度学习框架 |
| `transformers` | 加载 BERT 模型与 tokenizer |
| `seqeval` | entity-level F1 评估 |
| `pytorch-crf` | CRF 层（仅 `--use_crf` 时需要） |
| `tqdm` | 训练进度条 |

### 2.2 从项目根目录一键安装（可选）

```powershell
cd E:\DeepLearning\week7\序列标注项目
pip install -r requirements.txt
```

> `requirements.txt` 还包含 `peft`、`openai` 等 LLM 相关库，work7 训练本身不需要，但不影响安装。

### 2.3 验证依赖

```powershell
python -c "import torch, transformers, seqeval, torchcrf, tqdm; print('全部依赖 OK')"
```

---

## 三、数据准备

### 3.1 数据位置

```
序列标注项目/data/peoples_daily/
├── train.json          # 20,864 条
├── validation.json     #  2,318 条
├── test.json           #  4,636 条（有标签）
└── label_names.json    # ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
```

### 3.2 若数据不存在，从 GitHub 下载

```powershell
cd E:\DeepLearning\week7\序列标注项目\src
python download_data.py
```

脚本会从 GitHub 下载人民日报 NER 的 CoNLL 原始文件，解析为 JSON 保存到 `data/peoples_daily/`。

数据来源（无需账号）：
- `https://github.com/OYE93/Chinese-NLP-Corpus` → NER/People's Daily/

若只需练习数据集、跳过 cluener：

```powershell
python download_data.py --skip_peoples_daily   # 不对，这会跳过 peoples_daily
# 正确做法：download_data.py 默认两个都下；若只要 peoples_daily，可单独运行后手动删除 cluener 部分
```

> 默认 `download_data.py` 会同时下载 cluener 和 peoples_daily，不影响 work7 使用。

### 3.3 验证数据

```powershell
python -c "import json; from pathlib import Path; d=Path(r'E:\DeepLearning\week7\序列标注项目\data\peoples_daily'); r=json.load(open(d/'train.json',encoding='utf-8')); print('训练集:', len(r), '条'); print('样例:', r[0]['tokens'][:6], r[0]['ner_tags'][:6])"
```

预期输出：`训练集: 20864 条`

---

## 四、预训练模型下载

work7 默认读取路径：

```
E:\DeepLearning\week7\pretrain_models\bert-base-chinese\
```

目录内需包含至少以下文件：

```
bert-base-chinese/
├── config.json
├── model.safetensors   # 或 pytorch_model.bin
├── vocab.txt
├── tokenizer.json      # ⚠ 必需，否则 word_ids() 报错
└── tokenizer_config.json
```

### 4.1 方式 A：ModelScope 下载（国内推荐）

```powershell
pip install modelscope

python -c "
from modelscope import snapshot_download
p = snapshot_download('AI-ModelScope/bert-base-chinese',
                      cache_dir=r'E:\DeepLearning\week7\pretrain_models')
print('下载完成:', p)
"
```

下载后模型位于：
`E:\DeepLearning\week7\pretrain_models\AI-ModelScope\bert-base-chinese\`

可将该目录作为 `--bert_path`，或复制关键文件到标准路径：

```powershell
$src = "E:\DeepLearning\week7\pretrain_models\AI-ModelScope\bert-base-chinese"
$dst = "E:\DeepLearning\week7\pretrain_models\bert-base-chinese"
New-Item -ItemType Directory -Force -Path $dst
Copy-Item "$src\*" $dst -Recurse -Force
```

### 4.2 方式 B：HuggingFace 下载

```powershell
# 国内建议设镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

python -c "
from transformers import BertModel, BertTokenizerFast
p = r'E:\DeepLearning\week7\pretrain_models\bert-base-chinese'
BertTokenizerFast.from_pretrained('bert-base-chinese').save_pretrained(p)
BertModel.from_pretrained('bert-base-chinese').save_pretrained(p)
print('保存到', p)
"
```

### 4.3 方式 C：训练时在线加载（需网络）

```powershell
python train.py --bert_path bert-base-chinese
```

首次运行会自动从 HuggingFace 缓存，无需本地目录。

### 4.4 验证模型

```powershell
python -c "
from transformers import BertTokenizerFast, BertModel
p = r'E:\DeepLearning\week7\pretrain_models\bert-base-chinese'
tok = BertTokenizerFast.from_pretrained(p)
mdl = BertModel.from_pretrained(p)
print('tokenizer:', type(tok).__name__)
print('模型参数量:', sum(x.numel() for x in mdl.parameters())/1e6, 'M')
"
```

预期：`BertTokenizerFast`，参数量约 `102.3 M`

> **踩坑记录**：若只有 `vocab.txt` 没有 `tokenizer.json`，会加载慢速 `BertTokenizer`，导致 `word_ids()` 不可用。work7 已改用 `BertTokenizerFast`，但模型目录仍须含 `tokenizer.json`。

---

## 五、目录结构确认

训练前确认如下结构完整：

```
week7/
└── 序列标注项目/
    ├── data/peoples_daily/          ← 数据
    ├── work7/                       ← 代码
    │   ├── train.py
    │   ├── evaluate.py
    │   └── ...
    └── pretrain_models/             ← 与 week7 同级
        └── bert-base-chinese/
            ├── config.json
            ├── model.safetensors
            └── tokenizer.json
```

---

## 六、开始训练

全部准备就绪后：

```powershell
conda activate py312
cd E:\DeepLearning\week7\序列标注项目\work7

# BERT + Linear（GTX 1060 建议 batch=16）
python train.py --batch_size 16 --bert_path "E:\DeepLearning\week7\pretrain_models\bert-base-chinese"

# BERT + CRF
python train.py --use_crf --batch_size 16 --bert_path "E:\DeepLearning\week7\pretrain_models\bert-base-chinese"

# 测试集评估
python evaluate.py --split test --batch_size 16 --bert_path "E:\DeepLearning\week7\pretrain_models\bert-base-chinese"
python evaluate.py --use_crf --split test --batch_size 16 --bert_path "E:\DeepLearning\week7\pretrain_models\bert-base-chinese"

# 汇总对比
python compare_results.py
```

---

## 七、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: seqeval` | 未安装评估库 | `pip install seqeval` |
| `ModuleNotFoundError: torchcrf` | 未安装 CRF 库 | `pip install pytorch-crf` |
| `word_ids() is not available` | 缺少 `tokenizer.json` | 用 ModelScope/HF 完整下载，或改用 `BertTokenizerFast` |
| CUDA OOM 显存不足 | batch 太大 | `--batch_size 8` 或 `--batch_size 16` |
| 找不到 checkpoint | 未训练或路径错误 | 先运行 `train.py`，检查 `work7/outputs/checkpoints/` |
| 模型下载慢/失败 | 网络问题 | 使用 ModelScope 或 `$env:HF_ENDPOINT="https://hf-mirror.com"` |

---

## 八、准备清单（Checklist）

- [ ] Python 3.10+ 环境已激活
- [ ] `torch`、`transformers`、`seqeval`、`pytorch-crf`、`tqdm` 已安装
- [ ] CUDA 可用（或接受 CPU 训练，极慢）
- [ ] `data/peoples_daily/` 四个 JSON 文件存在
- [ ] `bert-base-chinese` 模型含 `model.safetensors` + `tokenizer.json`
- [ ] 进入 `work7/` 目录，可以运行 `python train.py`

全部打勾即可开始训练。实验结果见 [`RESULTS.md`](RESULTS.md)。
