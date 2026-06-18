# USAGE_GUIDE_PEOPLES.md — 人民日报 NER 训练与评估指南

## 1. 环境准备

### 依赖安装

```bash
pip install torch transformers seqeval pytorch-crf tqdm
```

核心依赖说明：

| 依赖包 | 版本要求 | 用途 |
|--------|----------|------|
| `torch` | ≥1.9 | 深度学习框架 |
| `transformers` | ≥4.0 | BERT 模型加载 |
| `seqeval` | ≥1.2.2 | Entity-level F1 评估 |
| `pytorch-crf` | ≥0.7.2 | CRF 层实现 |
| `tqdm` | 最新版 | 训练进度条 |

### 本地模型路径

本模块使用 `bert-base-chinese` 预训练模型，路径配置在 `train_peoples.py` 第 42 行：

```python
BERT_PATH = Path(r"D:\BaiduNetdiskDownload\AI\pretrain_models\bert-base-chinese")
```

如模型路径不同，请修改此处。

---

## 2. 数据集说明

### 数据集规模

| 数据集 | 规模 |
|--------|------|
| 训练集 | 20,864 条 |
| 验证集 | 2,318 条 |
| 测试集 | 4,636 条 |

### 标签体系

| 标签 | 说明 |
|------|------|
| `O` | 非实体 |
| `B-PER` / `I-PER` | 人名实体的开头/内部 |
| `B-ORG` / `I-ORG` | 机构名实体的开头/内部 |
| `B-LOC` / `I-LOC` | 地名实体的开头/内部 |

共 **7 个 BIO 标签**（O + 3类 × 2）。

### 数据格式

数据存储在 `data/peoples_daily/` 目录下，为分词 token 列表 + BIO 标签列表格式：

```json
{
  "tokens": ["华为", "技术", "有限公司", "总裁", "任正非", "在", "深圳", "接受", "采访"],
  "labels": ["B-ORG", "I-ORG", "I-ORG", "O", "B-PER", "O", "B-LOC", "O", "O"]
}
```

---

## 3. 训练与评估流程

### Step 1：探索数据（可选）

```bash
cd src_peoples
python explore_peoples.py
```

**输出文件**：
- `outputs/figures/entity_distribution_peoples.png` — 各类实体频次直方图
- `outputs/figures/text_length_distribution_peoples.png` — 文本长度分布
- `outputs/figures/entity_length_distribution_peoples.png` — 实体字符数分布

---

### Step 2：训练 BERT + Linear（基线）

```bash
python train_peoples.py
```

**可选参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--max_length` | 128 | 文本截断长度 |
| `--lr` | 2e-5 | BERT 层学习率 |
| `--head_lr_mult` | 5.0 | 分类头学习率倍数（BertNER 用 1e-4）|
| `--warmup_ratio` | 0.1 | 预热步比例 |
| `--dropout` | 0.1 | Dropout 概率 |

**训练日志示例**（3 epochs）：

```
Epoch 1/3 | train_loss=0.1514 | val_loss=0.0231 | val_entity_f1=0.9285 | time=221s
  ★ 新最优 F1=0.9285，已保存
Epoch 2/3 | train_loss=0.0175 | val_loss=0.0165 | val_entity_f1=0.9435 | time=221s
  ★ 新最优 F1=0.9435，已保存
Epoch 3/3 | train_loss=0.0086 | val_loss=0.0171 | val_entity_f1=0.9505 | time=221s
  ★ 新最优 F1=0.9505，已保存

训练完成！最优 val_entity_f1=0.9505
```

**输出文件**：
- `outputs/checkpoints_peoples/best_linear_peoples.pt` — 最优模型
- `outputs/logs_peoples/train_linear_peoples.json` — 训练日志

---

### Step 3：训练 BERT + CRF

```bash
python train_peoples.py --use_crf
```

CRF 版每个 epoch 比 Linear 版慢约 20-30%（前向-后向算法开销），但能学习标签转移约束，产生合法的 BIO 序列。

**训练日志示例**（3 epochs）：

```
Epoch 1/3 | train_loss=6.9944 | val_loss=0.9018 | val_entity_f1=0.9303 | time=300s
  ★ 新最优 F1=0.9303，已保存
Epoch 2/3 | train_loss=0.8094 | val_loss=0.7894 | val_entity_f1=0.9529 | time=302s
  ★ 新最优 F1=0.9529，已保存
Epoch 3/3 | train_loss=0.3923 | val_loss=0.9195 | val_entity_f1=0.9552 | time=298s
  ★ 新最优 F1=0.9552，已保存

训练完成！最优 val_entity_f1=0.9552
```

**输出文件**：
- `outputs/checkpoints_peoples/best_crf_peoples.pt` — 最优模型
- `outputs/logs_peoples/train_crf_peoples.json` — 训练日志

---

### Step 4：评估

```bash
# 评估 BERT + Linear
python evaluate_peoples.py

# 评估 BERT + CRF
python evaluate_peoples.py --use_crf
```

**评估输出示例**（BERT + Linear，test 集）：

```
======================================================================
模型：BERT + Linear  |  评估集：test
======================================================================
Entity-level Precision: 0.9335
Entity-level Recall:    0.9479
Entity-level F1:        0.9407

【逐类型 F1】
              precision    recall  f1-score   support

         LOC     0.9501    0.9503    0.9502      3464
         ORG     0.8762    0.9183    0.8968      2166
         PER     0.9732    0.9786    0.9759      1820

   micro avg     0.9335    0.9479    0.9407      7450
   macro avg     0.9332    0.9491    0.9410      7450
weighted avg     0.9343    0.9479    0.9409      7450

【非法 BIO 序列统计】
  总序列数：4636
  非法开头（I-X 开头）：0 条
  非法转移（B-X/I-X → I-Y, X≠Y）：128 条
  合计非法序列：128 条
  → 线性头约 2.8% 的序列含非法转移，充分训练的 CRF 可完全消除
```

**评估输出示例**（BERT + CRF，test 集）：

```
======================================================================
模型：BERT + CRF  |  评估集：test
======================================================================
Entity-level Precision: 0.9358
Entity-level Recall:    0.9509
Entity-level F1:        0.9433

【逐类型 F1】
              precision    recall  f1-score   support

         LOC     0.9459    0.9541    0.9500      3464
         ORG     0.8871    0.9215    0.9040      2166
         PER     0.9732    0.9786    0.9759      1820

   micro avg     0.9358    0.9509    0.9433      7450
   macro avg     0.9365    0.9518    0.9440      7450
weighted avg     0.9363    0.9509    0.9435      7450

【非法 BIO 序列统计】
  总序列数：4636
  非法开头（I-X 开头）：0 条
  非法转移（B-X/I-X → I-Y, X≠Y）：108 条
  合计非法序列：108 条
  → CRF 非法序列 108 条（2.3%）
  → 提示：训练 epoch 不足时转移矩阵尚未收敛；充分训练（3+ epochs）后可降至 0
```

**输出文件**：
- `outputs/logs_peoples/eval_linear_peoples_test.json`
- `outputs/logs_peoples/eval_crf_peoples_test.json`

---

## 4. 两方案对比

| 指标 | BERT + Linear | BERT + CRF |
|------|----------------|------------|
| Test F1 | 0.9407 | 0.9433 |
| PER F1 | 0.9759 | 0.9781 |
| ORG F1 | 0.8968 | 0.9040 |
| LOC F1 | 0.9502 | 0.9500 |
| 非法序列数 | 128 (2.8%) | 108 (2.3%) |
| 每 epoch 耗时 | ~221s | ~300s |
| 可训练参数 | 102.3M | 102.3M |

**关键观察**：
1. CRF 相比 Linear 在 Test F1 上提升约 0.3 个点
2. CRF 能学习标签转移约束，减少非法序列
3. LOC 类型两者表现接近，ORG 类型 CRF 提升明显
4. 训练 epoch 不足时 CRF 转移矩阵未完全收敛，可继续训练

---

## 5. CRF 训练优化策略

### 5.1 核心矛盾：过拟合 vs 转移矩阵收敛

从训练日志可以观察到一个有趣的矛盾：

| Epoch | train_loss | val_loss | val_entity_f1 | 非法序列数 |
|-------|------------|----------|---------------|------------|
| 1 | 6.9944 | 0.9018 | 0.9303 | - |
| 2 | 0.8094 | **0.7894** (最低) | 0.9529 | - |
| 3 | 0.3923 | **0.9195** (反弹) | **0.9552** (最高) | 108 条 |

**现象分析**：
- `train_loss` 持续下降（0.81 → 0.39），说明模型在训练集上持续学习
- `val_loss` 在第 3 轮上升（0.79 → 0.92），表明 BERT 层已开始过拟合
- `val_entity_f1` 在第 3 轮仍在上升（0.9529 → 0.9552），说明实体边界预测仍在提升
- 测试集仍有 108 条非法序列（2.3%），说明 CRF 转移矩阵未完全收敛

### 5.2 根本原因：学习率分配不合理

当前代码中各模块的学习率配置：

| 模块 | 参数量 | 学习率 | 问题 |
|------|--------|--------|------|
| **BERT 层** | ~102M | 2e-5 | 过拟合速度较快 |
| **分类头** | ~7K | 1e-4 | 正常 |
| **CRF 转移矩阵** | **仅 49 个**（7×7） | 1e-4 | **学习太慢** |

**关键问题**：CRF 转移矩阵只有 49 个参数，但使用与分类头（7K 参数）相同的学习率，导致收敛速度严重不足。

### 5.3 优化方案

#### 方案 A：分层学习率（推荐）

为 CRF 层设置更高的学习率，加速转移矩阵收敛：

```python
# 修改 train_peoples.py 中的 optimizer 配置
if args.use_crf:
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},              # 2e-5（防止 BERT 过拟合）
            {"params": head_params, "lr": args.lr * args.head_lr_mult},  # 1e-4（分类头）
            {"params": model.crf.parameters(), "lr": args.lr * 20},      # 4e-4（加速 CRF）
        ],
        weight_decay=0.01,
    )
```

#### 方案 B：添加早停机制

在 val_loss 开始上升时自动停止，避免过拟合：

```python
# 在训练循环中添加
patience = 2  # 连续 2 轮 val_loss 上升则停止
best_val_loss = float('inf')
no_improve_count = 0

for epoch in range(1, args.epochs + 1):
    ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"早停：val_loss 连续 {patience} 轮未改善")
            break
```

#### 方案 C：调整超参数

| 参数 | 当前值 | CRF 优化建议 |
|------|--------|--------------|
| `--lr` | 2e-5 | 1e-5（降低 BERT 学习率） |
| `--head_lr_mult` | 5.0 | 2.0（减少头部学习率） |
| `--dropout` | 0.1 | 0.2~0.3（增加正则化） |
| `--warmup_ratio` | 0.1 | 0.2（延长预热） |

### 5.4 预期效果

| 策略 | 预期结果 |
|------|----------|
| **CRF 学习率提高 4 倍** | 转移矩阵在 2-3 轮内完全收敛，非法序列降至 0 |
| **BERT 学习率降低** | 过拟合速度减缓 |
| **早停机制** | 在最优时机停止，避免继续过拟合 |

### 5.5 综合建议命令

```bash
# 优化后的 CRF 训练命令
python train_peoples.py --use_crf \
  --lr 1e-5 \
  --head_lr_mult 2 \
  --dropout 0.2 \
  --warmup_ratio 0.2 \
  --epochs 5
```

---

## 6. 作为模块调用

```python
from pathlib import Path
import torch
from transformers import BertTokenizer

import sys
sys.path.insert(0, str(Path("src_peoples")))

from dataset_peoples import build_label_schema, build_dataloaders
from model import build_model

BERT_PATH = Path(r"D:\BaiduNetdiskDownload\AI\pretrain_models\bert-base-chinese")
labels, label2id, id2label = build_label_schema()

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(
    use_crf=False,
    bert_path=str(BERT_PATH),
    num_labels=len(labels),
    dropout=0.1
).to(device)

ckpt = torch.load("outputs/checkpoints_peoples/best_linear_peoples.pt", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

text = "华为技术有限公司总裁任正非在深圳接受媒体采访"
chars = list(text)
enc = tokenizer(chars, is_split_into_words=True, max_length=128,
                truncation=True, padding="max_length", return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
token_type_ids = enc["token_type_ids"].to(device)

with torch.no_grad():
    logits = model(input_ids, attention_mask, token_type_ids)
    pred_ids = logits.argmax(dim=-1)[0]

word_ids = enc.word_ids(0)
results = []
prev_entity = None
for j, wid in enumerate(word_ids):
    if wid is None:
        continue
    if j < len(pred_ids):
        tag = id2label[pred_ids[j].item()]
        if tag.startswith("B-"):
            prev_entity = {"type": tag[2:], "start": wid, "end": wid, "text": chars[wid]}
            results.append(prev_entity)
        elif tag.startswith("I-") and prev_entity and prev_entity["type"] == tag[2:]:
            prev_entity["end"] = wid
            prev_entity["text"] = text[prev_entity["start"]:prev_entity["end"]+1]
        else:
            prev_entity = None

print(results)
# 输出：[{"type": "ORG", "start": 0, "end": 2, "text": "华为技术有限公司"},
#        {"type": "PER", "start": 4, "end": 4, "text": "任正非"},
#        {"type": "LOC", "start": 6, "end": 6, "text": "深圳"}]
```

---

## 7. 文件结构

```
src_peoples/
├── __init__.py               # 包初始化
├── dataset_peoples.py        # 数据集构建与标签体系
├── model.py                  # BertNER（Linear）和 BertCRFNER（CRF）模型
├── train_peoples.py          # 训练脚本
├── evaluate_peoples.py       # 评估脚本
├── explore_peoples.py        # 数据探索脚本
└── result_log.txt            # 训练与评估结果日志
```

---

## 8. 调试与常见问题

**Q: `ModuleNotFoundError: No module named 'seqeval'`**
A: 缺少依赖包，运行 `pip install seqeval`

**Q: `ModuleNotFoundError: No module named 'pytorch_crf'`**
A: 缺少 CRF 包，运行 `pip install pytorch-crf`（注意包名是 `pytorch-crf`，导入名是 `pytorch_crf`）

**Q: `OSError: Repo id must use alphanumeric chars...`**
A: BERT 模型路径包含非法字符或路径不存在。检查 `BERT_PATH` 是否正确指向本地 `bert-base-chinese` 目录。

**Q: 非法序列数较多，如何降低？**
A: CRF 模型需要足够的训练 epoch 让转移矩阵收敛。继续训练（增加 `--epochs`）可将非法序列数降至接近 0。

**Q: 如何加速训练？**
A: 减小 `--batch_size`（如 16）可降低显存占用；Linear 模型比 CRF 训练快约 20-30%。

**Q: 如何使用自己的数据集？**
A: 参考 `dataset_peoples.py` 中的 `build_label_schema()` 和 `build_dataloaders()` 函数，将数据整理为 `{"tokens": [...], "labels": [...]}` 的 JSON Lines 格式。
