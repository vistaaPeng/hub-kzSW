# 训练前置准备工作指南

本文档记录在 `work8/` 跑 LCQMC / BQ Corpus 文本匹配实验前需要完成的环境、依赖、数据和模型准备。

---

## 一、环境要求

| 项目 | 最低要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 与 work7 相同 |
| PyTorch | 2.0+，带 CUDA | CPU 可跑但 LCQMC 极慢 |
| GPU 显存 | ≥ 6GB（batch=16） | GTX 1060 6GB 可用 |
| 磁盘空间 | ~2GB（模型 + 日志） | LCQMC 训练日志可能较大 |

```powershell
conda activate py312

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 二、安装依赖

### work8 最小依赖

```powershell
pip install torch transformers scikit-learn tqdm
```

### 从项目根目录安装（可选）

```powershell
cd E:\DeepLearning\week7\文本匹配项目
pip install -r requirements.txt
```

验证：

```powershell
python -c "import torch, transformers, sklearn, tqdm; print('全部依赖 OK')"
```

---

## 三、数据准备

数据位于上级目录，work8 直接读取：

```
文本匹配项目/data/
├── lcqmc/
│   ├── train.jsonl       # 238,766 条
│   ├── validation.jsonl  #   8,802 条
│   └── test.jsonl        #  12,500 条
└── bq_corpus/
    ├── train.jsonl       #  68,960 条
    ├── validation.jsonl  #   8,620 条
    └── test.jsonl        #   8,620 条
```

若数据不存在：

```powershell
cd E:\DeepLearning\week7\文本匹配项目\src
python download_data.py
```

验证：

```powershell
python -c "from pathlib import Path; d=Path(r'E:\DeepLearning\week7\文本匹配项目\data\lcqmc\train.jsonl'); print('LCQMC train 行数:', sum(1 for _ in open(d,encoding='utf-8')))"
```

---

## 四、预训练模型

默认路径（与 work7 共用）：

```
E:\DeepLearning\week7\pretrain_models\bert-base-chinese\
```

需包含：`config.json`、`model.safetensors`（或 `pytorch_model.bin`）、`vocab.txt`、`tokenizer.json`。

若尚未下载，参考 [`序列标注项目/work7/SETUP.md`](../序列标注项目/work7/SETUP.md) 第四节 ModelScope 或 HuggingFace 方式。

验证：

```powershell
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained(r'E:\DeepLearning\week7\pretrain_models\bert-base-chinese'); print('BERT OK')"
```

---

## 五、冒烟测试（建议）

先用 **BQ Corpus**、**1 epoch** 验证流程能跑通：

```powershell
cd E:\DeepLearning\week7\文本匹配项目\work8

python train_biencoder.py --dataset bq_corpus --loss cosine --epochs 1 --batch_size 16
python evaluate.py --dataset bq_corpus --model_type biencoder --loss cosine --split validation
```

无报错即可开始完整实验。

---

## 六、耗时预估

| 实验 | 约计（4 层 BERT，3 epoch，RTX 4060 级 GPU） |
|------|---------------------------------------------|
| BQ × 3 方法 | 1~2 小时 |
| LCQMC × 3 方法 | 6~12 小时（train 24 万条） |

建议：先完成 BQ 全套并写好报告框架，再挂 LCQMC 长跑。

---

## 七、准备清单

- [ ] Python 环境已激活，CUDA 可用
- [ ] `torch`、`transformers`、`scikit-learn`、`tqdm` 已安装
- [ ] `data/lcqmc/` 与 `data/bq_corpus/` 六个 jsonl 文件存在
- [ ] `pretrain_models/bert-base-chinese/` 模型完整
- [ ] BQ 冒烟测试通过

全部打勾后，按 [`README.md`](README.md) 运行完整实验，结果填入 [`RESULTS.md`](RESULTS.md)。
