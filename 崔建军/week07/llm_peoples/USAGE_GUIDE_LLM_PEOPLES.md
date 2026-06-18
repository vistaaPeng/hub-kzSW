# USAGE_GUIDE_LLM_PEOPLES.md

# 人民日报 NER — Qwen2.5-0.5B + LoRA 使用指南

## 目录

1. [环境准备](#1-环境准备)
2. [数据集说明](#2-数据集说明)
3. [模型架构](#3-模型架构)
4. [训练与评估流程](#4-训练与评估流程)
5. [性能对比](#5-性能对比)
6. [关键参数说明](#6-关键参数说明)
7. [文件结构](#7-文件结构)
8. [调试与常见问题](#8-调试与常见问题)

---

## 1. 环境准备

### 1.1 依赖安装

```bash
pip install torch transformers peft qwen-community seqeval
```

| 包名 | 版本要求 | 说明 |
|------|----------|------|
| torch | ≥2.0 | PyTorch 框架 |
| transformers | ≥4.x | Hugging Face 模型库 |
| peft | ≥0.7 | LoRA 微调工具 |
| qwen-community | 最新 | Qwen2 模型支持 |
| seqeval | 最新 | 序列标注评估 |

### 1.2 本地模型路径

模型文件位于 `D:\BaiduNetdiskDownload\AI\pretrain_models\Qwen2.5-0.5B-Instruct`，代码中已配置：

```python
# dataset_qwen.py
MODEL_NAME = Path(r"D:\BaiduNetdiskDownload\AI\pretrain_models\Qwen2.5-0.5B-Instruct")
```

### 1.3 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 6GB 显存 | 8GB+ 显存 |
| 内存 | 16GB | 32GB |
| 硬盘 | 10GB | 20GB |

---

## 2. 数据集说明

### 2.1 数据规模

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| 训练集 | 20,864 | 模型学习 |
| 验证集 | 2,318 | 调参与早停 |
| 测试集 | 4,636 | 最终评估 |

### 2.2 BIO 标签体系

| 标签 | 实体类型 | 示例 |
|------|----------|------|
| O | 非实体 | - |
| B-PER | 人名开始 | 任 |
| I-PER | 人名延续 | 正、非 |
| B-ORG | 机构开始 | 华 |
| I-ORG | 机构延续 | 为、技、术 |
| B-LOC | 地名开始 | 北 |
| I-LOC | 地名延续 | 京 |

### 2.3 数据格式（CoNLL）

```
任 B-PER
正 I-PER
非 I-PER
在 O
深 B-LOC
圳 I-LOC
华 B-ORG
为 I-ORG
有 I-ORG
限 I-ORG
公 I-ORG
司 I-ORG
```

---

## 3. 模型架构

### 3.1 Qwen2.5-0.5B + LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen2.5-0.5B-Instruct | 630.7M 参数 |
| LoRA target_modules | q_proj, v_proj | Attention 的 Q/V 层 |
| lora_rank | 8 | LoRA 矩阵维度 |
| lora_alpha | 16 | LoRA 缩放因子 |
| lora_dropout | 0.05 | Dropout 概率 |
| 可训练参数 | ~0.55M | 仅占 0.09% |

### 3.2 参数规模对比

| 模型 | 总参数量 | 可训练参数 | 占比 |
|------|----------|-----------|------|
| BERT-base（Linear） | 102.3M | 102.3M | 100% |
| BERT-base（CRF） | 102.3M | 102.3M | 100% |
| **Qwen2.5-0.5B（LoRA）** | 630.7M | **~0.55M** | **0.09%** |

### 3.3 学习率配置

| 模块 | 学习率 | 说明 |
|------|--------|------|
| Qwen 原始参数 | 冻结 | 不更新 |
| LoRA 参数 | 1e-4 | 较高以加速收敛 |
| 分类头参数 | 1e-4 | 与 LoRA 相同 |

---

## 4. 训练与评估流程

### Step 1：数据探索（可选）

```bash
cd d:\BaiduNetdiskDownload\AI\code\hub-kzSW\崔建军\week07\llm_peoples
python -c "from dataset_qwen import load_data; train,_=load_data('train'); print(f'训练集: {len(train)} 条')"
```

### Step 2：训练 LoRA 模型

```bash
python train_qwen.py --epochs 5 --lr 1e-4 --batch_size 4
```

**训练输出示例**（3 epochs）：

```
设备：cuda
BIO 标签数：7（O + 6 个实体标签）
实体类型：['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
数据集规模：训练=20864，验证=2318
模型：Qwen2.5-0.5B + LoRA
  参数总量：630.7M
  可训练参数：0.55M (0.09%)

训练步数：5216，预热步数：521

开始训练（Qwen2.5-0.5B + LoRA）— peoples_daily 数据集...
Epoch 1/3 | train_loss=0.3657 | val_loss=0.3015 | val_entity_f1=0.5905 | time=642s
  ★ 新最优 F1=0.5905，已保存 → outputs/checkpoints_qwen/best_qwen_lora_peoples.pt
Epoch 2/3 | train_loss=0.3657 | val_loss=0.3015 | val_entity_f1=0.5905 | time=642s
  ★ 新最优 F1=0.5905，已保存 → outputs/checkpoints_qwen/best_qwen_lora_peoples.pt
Epoch 3/3 | train_loss=0.2591 | val_loss=0.2417 | val_entity_f1=0.6435 | time=819s
  ★ 新最优 F1=0.6435，已保存 → outputs/checkpoints_qwen/best_qwen_lora_peoples.pt

训练完成！最优 val_entity_f1=0.6435
  Checkpoint: D:\BaiduNetdiskDownload\AI\code\hub-kzSW\崔建军\week07\outputs\checkpoints_qwen\best_qwen_lora_peoples.pt
  训练日志:   D:\BaiduNetdiskDownload\AI\code\hub-kzSW\崔建军\week07\outputs\logs_qwen\train_qwen_lora_peoples.json

下一步：python evaluate_qwen.py
```

### Step 3：评估模型

```bash
python evaluate_qwen.py
python evaluate_qwen.py --split validation  # 评估验证集
```

**评估输出示例**（3 epochs 后）：

```
模型：Qwen2.5-0.5B + LoRA  |  评估集：test
======================================================================
Entity-level Precision: 0.5883
Entity-level Recall:    0.6768
Entity-level F1:        0.6294

【逐类型 F1】
  weighted avg     0.6796    0.5883    0.6239     12989

【非法 BIO 序列统计】
  总序列数：4636
  非法开头（I-X 开头）：0 条
  非法转移（B-X/I-X → I-Y, X≠Y）：708 条
  合计非法序列：708 条
  → 非法序列占比：15.3%
```

### Step 4：对比分析

训练完成后，对比三种模型方案：

| 模型 | Test F1 | Val F1 | 可训练参数 | 训练时间 |
|------|---------|--------|-----------|----------|
| BERT + Linear | 0.9407 | 0.9505 | 102.3M | ~15min |
| BERT + CRF | 0.9433 | 0.9552 | 102.3M | ~25min |
| Qwen + LoRA | **0.6294** | **0.6435** | ~0.55M | ~35min/3epochs |

---

## 5. 性能对比

### 5.1 BERT vs Qwen+LoRA

| 指标 | BERT+Linear | BERT+CRF | Qwen+LoRA |
|------|-------------|----------|-----------|
| **Test F1** | 0.9407 | 0.9457 | **0.6294** |
| **Val F1** | 0.9505 | 0.9552 | **0.6435** |
| PER F1 | 0.9759 | 0.9751 | N/A |
| ORG F1 | 0.8968 | 0.9110 | N/A |
| LOC F1 | 0.9502 | 0.9520 | N/A |
| 可训练参数 | 102.3M | 102.3M | ~0.55M |
| 训练时间/epoch | ~5min | ~8min | ~10-13min |

**Qwen+LoRA 训练结果（3 epochs）**：
- Train Loss: 0.2591 → 持续下降，模型仍在学习
- Val Loss: 0.2417 → 验证集损失较低
- Val F1: 0.6435 → 验证集性能持续上升
- Test F1: 0.6294 → 测试集性能
- 非法序列占比: 15.3%（708/4636）

### 5.2 关键发现

1. **LoRA 参数效率高**：仅用 0.55M 参数（占总参数的 0.09%）实现了一定的 NER 能力
2. **性能差距明显**：Qwen+LoRA 的 Test F1（0.6294）明显低于 BERT+Linear（0.9407）和 BERT+CRF（0.9457）
3. **Decoder-only 架构限制**：Qwen 作为生成式模型，NER 不是其原生任务，性能受限于单向注意力机制
4. **训练仍在收敛**：3 epochs 后 Train Loss 仍在下降，继续训练可能提升性能

---

## 6. 关键参数说明

### 6.1 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--lr` | 1e-4 | LoRA 学习率 |
| `--batch_size` | 4 | 批次大小（显存不足时减小） |
| `--max_seq_len` | 256 | 最大序列长度 |
| `--warmup_ratio` | 0.1 | 预热步数比例 |
| `--weight_decay` | 0.01 | 权重衰减 |

### 6.2 LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_rank` | 8 | LoRA 矩阵秩，越大表达能力越强但参数量增加 |
| `--lora_alpha` | 16 | LoRA 缩放因子，通常设为 rank 的 2 倍 |
| `--lora_dropout` | 0.05 | LoRA 层的 Dropout |

### 6.3 常见问题处理

| 问题 | 解决方案 |
|------|----------|
| CUDA OOM | 减小 `--batch_size` 到 2 或 1 |
| loss 不下降 | 检查数据标注是否正确 |
| F1 过低 | 增加 `--epochs` 或调整 `--lr` |

---

## 7. 文件结构

```
llm_peoples/
├── __init__.py           # 包初始化
├── dataset_qwen.py       # 数据集加载与预处理
├── model_qwen.py         # Qwen2.5 + LoRA 模型定义
├── train_qwen.py         # 训练脚本
├── evaluate_qwen.py      # 评估脚本
└── outputs/              # 输出目录（自动创建）
    ├── checkpoints_qwen/ # 模型 checkpoint
    │   └── best_qwen_lora_peoples.pt
    └── logs_qwen/       # 训练/评估日志
        ├── train_qwen_lora_peoples.json
        └── eval_qwen_lora_peoples_test.json
```

---

## 8. 调试与常见问题

### Q1: CUDA out of memory

**问题**：训练时显存不足。

**解决方案**：
```bash
# 减小批次大小
python train_qwen.py --batch_size 2
# 或
python train_qwen.py --batch_size 1
```

### Q2: LoRA 不收敛

**问题**：val_loss 持续上升或波动。

**解决方案**：
- 减小学习率：`--lr 5e-5`
- 增加预热比例：`--warmup_ratio 0.2`
- 检查数据标注是否正确

### Q3: 如何选择最优模型？

**建议**：
1. 观察 `val_entity_f1`，选择最高的 epoch 对应模型
2. 不要单纯看 `val_loss`，因为它和 F1 不完全正相关
3. 最终以 test F1 为准

### Q4: Entity-level F1 vs Token-level F1

| 指标 | 评估方式 | 关注点 |
|------|----------|--------|
| **Entity-level F1** | 完整实体边界匹配 | 实体是否完整被识别 |
| **Token-level F1** | 每个 token 单独评估 | 逐字预测准确性 |

本项目使用 **Entity-level F1** 作为主要评估指标。

### Q5: 如何加速训练？

1. 增加批次大小（需要更多显存）
2. 减小序列长度 `--max_seq_len 128`
3. 使用梯度累积 `--gradient_accumulation 4`

---

## 参考命令汇总

```bash
cd d:\BaiduNetdiskDownload\AI\code\hub-kzSW\崔建军\week07\llm_peoples

# 基础训练（3 epochs）
python train_qwen.py

# 长时间训练（5 epochs，更好的收敛）
python train_qwen.py --epochs 5 --lr 1e-4 --batch_size 4

# 显存不足时
python train_qwen.py --epochs 5 --batch_size 2

# 评估测试集
python evaluate_qwen.py

# 评估验证集
python evaluate_qwen.py --split validation

# 加载特定 checkpoint 评估
python evaluate_qwen.py --checkpoint outputs/checkpoints_qwen/best_qwen_lora_peoples.pt
```
