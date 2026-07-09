# Week 6 文本分类实验记录

## 一、项目概述

基于 `bert-base-chinese` 对 CLUE TNEWS 新闻标题数据集进行 15 分类。项目采用模块化设计，包含数据管道、模型定义、训练、评估、推理五个核心模块。

> 完整的模块关系说明见同目录下的 `BERT文本分类说明.md`。

---

## 二、数据分布

训练集 53,360 条，验证集 10,000 条，类别分布**极不均衡**：

| 类别 | 训练集 | 验证集 | 占比 |
|------|--------|--------|------|
| 科技 | 5,955 | 1,089 | 11.2% |
| 财经 | 5,200 | 956 | 9.7% |
| 娱乐 | 4,976 | 910 | 9.3% |
| 国际 | 4,851 | 905 | 9.1% |
| 文化 | 4,081 | 736 | 7.6% |
| 体育 | 3,991 | 767 | 7.5% |
| 旅游 | 3,437 | 646 | 6.4% |
| 军事 | 3,368 | 693 | 6.3% |
| 电竞 | 3,390 | 659 | 6.4% |
| 教育 | 3,437 | 646 | 6.4% |
| 汽车 | 2,886 | 494 | 5.4% |
| 农业 | 2,886 | 494 | 5.4% |
| 房产 | 2,107 | 378 | 3.9% |
| 故事 | 1,111 | 215 | 2.1% |
| 证券 | 257 | 45 | 0.5% |

---

## 三、BERT Fine-tuning 实验

### 3.1 训练1：CLS Pooling（基线）

```bash
python src/train.py --pool cls
```

**训练日志**：

| Epoch | Train Loss | Train Acc | Val Acc | Val Macro F1 |
|-------|-----------|-----------|---------|--------------|
| 1 | 1.442 | 51.0% | 55.97% | 0.5407 |
| 2 | 1.054 | 61.3% | **56.76%** | **0.5571** |
| 3 | 0.832 | 69.4% | 56.73% | 0.5509 |

> 第 2 epoch 达到最优，第 3 epoch 开始过拟合（Train Acc 持续上升，Val Acc 停滞）。

**评估**：
```bash
python src/evaluate.py --pool cls
```

![[confusion_matrix_cls.png]]

**批量推理**：
```bash
python src/predict.py --pool cls --input_file data/val.json --output_file outputs/val_predictions_cls.json
```
准确率：**5628/10000 = 56.28%**

**单条推理**：
```bash
python src/predict.py --pool cls --text "苹果发布了最新的 iPhone 17，搭载 A19 芯片"
```
![[Pasted image 20260528233602.png]]

---

### 3.2 训练2：Mean Pooling + 加权 Loss（处理类别不均衡）

```bash
python src/train.py --pool mean --use_class_weight
```

**训练日志**：

| Epoch | Train Loss | Train Acc | Val Acc | Val Macro F1 |
|-------|-----------|-----------|---------|--------------|
| 1 | 1.451 | 49.9% | 54.49% | 0.5342 |
| 2 | 1.029 | 60.1% | 55.64% | 0.5481 |
| 3 | 0.794 | 67.3% | **55.97%** | **0.5532** |

> Weighted 版本效果略低于 CLS 基线（55.97% vs 56.76%），`class_weight` 线性加权对过拟合的缓解有限。

**评估**：

![[Pasted image 20260528234015.png]]
![[confusion_matrix_mean.png]]

**单条推理**（需指定 weighted checkpoint）：
```bash
python src/predict.py --pool mean --text "苹果发布了最新的 iPhone 17，搭载 A19 芯片" --ckpt_path ./outputs/checkpoints/best_mean_weighted.pt
```
![[Pasted image 20260528233623.png]]

**批量推理**：
```bash
python src/predict.py --pool mean --input_file data/val.json --output_file outputs/val_predictions.json --ckpt_path outputs/checkpoints/best_mean_weighted.pt
```

---

### 3.3 关键发现：过拟合严重

三种配置的对比：

| 配置 | Train Acc (E3) | Val Acc (最优) | 差距 |
|------|----------------|----------------|------|
| cls | 69.4% | 56.76% | **12.6%** |
| cls_weighted | 66.9% | 55.93% | **11.0%** |
| mean_weighted | 67.3% | 55.97% | **11.3%** |
| roberta_base_std (cls) | 67.1% | **58.86%** | **8.2%** |

**问题诊断**：训练集准确率持续上升，验证集却卡在 56% 不动——模型在"死记硬背"训练数据，但泛化能力没提升。

---

### 3.4 训练4：参数调优（增大 Dropout + 降低学习率 + 增加 Epochs）

```bash
python src/train.py --pool cls --dropout 0.3 --epochs 10 --lr 1e-5 --max_length 128
```

**预期改进方向**：
- `dropout 0.3` → 抑制过拟合
- `lr 1e-5` → 更温和的微调
- `epochs 10` + Early Stopping → 找到真正的最优解
- `max_length 128` → 减少截断损失

> 代码还在运行中，待补充结果。

---

### 3.5 训练5：使用 `nlp_roberta_backbone_base_std`（CLS Pooling）

将基线预训练模型从 `bert-base-chinese` 替换为 `nlp_roberta_backbone_base_std`，观察相同训练配置下的表现差异。

```bash
# 训练（bert_path 已配置为 nlp_roberta_backbone_base_std 路径）
python src/train.py --pool cls
```

**训练日志**：

| Epoch | Train Loss | Train Acc | Val Acc | Val Macro F1 |
|-------|-----------|-----------|---------|--------------|
| 1 | 1.505 | 51.3% | 57.55% | 0.5396 |
| 2 | 1.087 | 61.5% | 58.64% | 0.5746 |
| 3 | 0.927 | 67.1% | **58.86%** | **0.5775** |

> 相比 `bert-base-chinese` 基线（56.76%），`nlp_roberta_backbone_base_std` 提升到 **58.86%**，提升约 **2.1 个百分点**；同时训练/验证差距从 12.6% 缩小到 8.2%，过拟合有所缓解。

**评估**（基于已保存的 `outputs/val_predictions.json`）：

| 指标 | 数值 |
|------|------|
| Accuracy | **58.86%** |
| Macro F1 | **0.5775** |

```bash
python src/evaluate.py --pool cls --ckpt_path outputs/checkpoints/best_cls_roberta_base.pt
```

![[confusion_matrix_cls_roberta_base.png]]

**批量推理**：

```bash
python src/predict.py \
    --pool cls \
    --ckpt_path outputs/checkpoints/best_cls_roberta_base.pt \
    --input_file data/val.json \
    --output_file outputs/val_predictions.json
```

准确率：**5886/10000 = 58.86%**

**单条推理示例**：

```bash
python src/predict.py \
    --pool cls \
    --ckpt_path outputs/checkpoints/best_cls_roberta_base.pt \
    --text "苹果发布了最新的 iPhone 17，搭载 A19 芯片"
```

---

### 3.5.1 疑问与反思：RoBERTa 没有 [CLS]，为什么得分更高？

> 对比 `bert-base-chinese` 和 `nlp_roberta_backbone_base_std` 的训练日志时产生了一个疑问：RoBERTa 并没有像 BERT 那样用 NSP 任务训练 [CLS]，为什么用 `pool=cls` 反而比 BERT 高？

#### 1. 代码里的 “CLS 池化” 到底取什么？

看 `src/model.py` 的 `_pool` 函数：

```python
if self.pool == "cls":
    return last_hidden[:, 0, :]   # 取序列第 0 个位置的 hidden state
```

它取的不是 `[CLS]` 这个词的“预训练语义”，而是**序列第 0 个位置的向量**。

- 对 BERT 来说，位置 0 是 `[CLS]`；
- 对 RoBERTa 来说，位置 0 是 `<s>`（句子起始符，和 BERT 的 `[CLS]` 位置等价，只是名字不同）。

所以虽然 RoBERTa 没有 NSP 任务，但在 fine-tuning 阶段，梯度会教会模型把分类需要的信息聚合到第 0 个位置。

#### 2. 为什么 RoBERTa 得分更高？

核心原因不是 [CLS] 用得好，而是**预训练得到的底层表示更强**：

| 方面 | bert-base-chinese | nlp_roberta_backbone_base_std |
|---|---|---|
| 预训练任务 | MLM + NSP | 通常只有 MLM（RoBERTa 标准配置） |
| 训练方式 | 静态 mask、较小 batch | 动态 mask、更大 batch、更多数据、更长训练 |
| 位置 0 的预训练 | NSP 让 [CLS] 带句间关系信息 | 没有 NSP，但 MLM 学到的上下文表示更丰富 |

带来的实际效果：

1. **基础语义表示更强**：RoBERTa 的 hidden state 质量更高，即使只取位置 0，也足以支撑更好的分类。
2. **泛化更好**：RoBERTa 的 train_acc 更低（67.1% vs 69.4%），但 val_acc 更高，说明它没有“死记硬背”训练集。
3. **中文下游任务更适配**：`nlp_roberta_backbone_base_std` 在大规模中文语料上训练，对 TNEWS 这类中文短文本分类更有优势。

#### 3. 小结

RoBERTa 得分更高，**不是因为它用了 [CLS]，而是因为它底层的中文语义表示更强、泛化更好**。BERT 的 `[CLS]` 有 NSP 预训练优势，但 RoBERTa 更强的 MLM 预训练 + 更优的训练策略弥补了这一点，并在下游任务上表现更好。

> 后续可以验证：对 RoBERTa 用 `pool=mean` 再训一次。因为它没有 NSP，均值池化理论上更公平——不依赖位置 0 是否被预训练成句子向量，而是综合所有 token 的信息。

---

## 四、LLM Zero-Shot 分类对比实验

使用本地 `Qwen2-0.5B-Instruct` 做 zero-shot 分类，与 BERT fine-tune 对比。

```bash
python src_llm/classify_llm.py --num_samples 100
```

### 4.1 各类别表现

| 类别 | 样本数 | 正确 | 错误 | 无法解析 | 准确率 |
|------|--------|------|------|----------|--------|
| **体育** | 13 | 10 | 2 | 1 | **76.9%** |
| **娱乐** | 7 | 5 | 0 | 2 | **71.4%** |
| **教育** | 11 | 6 | 4 | 1 | **54.5%** |
| **文化** | 4 | 2 | 0 | 2 | **50.0%** |
| 科技 | 9 | 4 | 4 | 1 | 44.4% |
| 军事 | 8 | 3 | 3 | 2 | 37.5% |
| 旅游 | 11 | 3 | 5 | 3 | 27.3% |
| 汽车 | 7 | 2 | 3 | 2 | 28.6% |
| 财经 | 7 | 2 | 5 | 0 | 28.6% |
| **国际** | 14 | 0 | 3 | **11** | **0.0%** |
| 农业 | 1 | 0 | 0 | 1 | 0.0% |
| 房产 | 2 | 0 | 0 | 2 | 0.0% |
| 故事 | 1 | 0 | 0 | 1 | 0.0% |
| 电竞 | 4 | 0 | 3 | 1 | 0.0% |
| 证券 | 1 | 0 | 1 | 0 | 0.0% |

**整体：37% 准确率，30% 无法解析**（BERT fine-tune 约 57~62%）

### 4.2 表现规律分析

**表现好的类别**（关键词独特、领域边界清晰）：
- **体育**：球队名、运动员名、赛事名是强信号，预训练语料中高频出现
- **娱乐**：明星/影视词汇明确，模型容易识别
- **教育**：大学名称、考试制度与教育类别强相关

**表现差的类别**：
- **国际（0%，78.6% 无法解析）**：模型输出"政治"/"外交"而非"国际"，类别名本身不够具体
- **财经 vs 科技（互相混淆）**：现代企业新闻常同时涉及财经和科技，边界模糊
- **电竞**：模型输出"游戏"而非"电竞"，预训练语料中"游戏"更通用

---

## 五、LLM Few-Shot 实验

每类提供 2 个示例，共 30 条 few-shot 示例嵌入 prompt，与 zero-shot 做对比。

```bash
python src_llm/classify_llm_fewshot.py --num_samples 100
```

**结果**：

| 指标 | Zero-Shot | Few-Shot (2条/类) | 变化 |
|------|-----------|-------------------|------|
| **准确率** | 37.0% | 37.0% | **0%** |
| **无法解析** | 30 条 | 30 条 | 0 |
| **变好样本** | — | 0 条 | — |
| **变差样本** | — | 0 条 | — |
| **不变样本** | — | **100 条** | — |

**结论**：0.5B 参数的小模型几乎没有 in-context learning 能力，few-shot 示例对它只是"噪音"而非"信号"。

**原因分析**：
1. 模型容量太小（0.5B），无法从示例中抽象分类规则
2. 30 条示例导致 prompt 过长，注意力被稀释
3. 小模型对 `user → assistant` 对话格式的 few-shot 不敏感

---

## 六、改进方案总结

针对 BERT fine-tuning 准确率瓶颈（~60%），按效果/投入比排序：

### 6.1 训练策略（最快见效）

| 改进项 | 当前值 | 建议值 | 预期效果 |
|--------|--------|--------|----------|
| Dropout | 0.1 | **0.3** | 抑制过拟合 |
| Max Length | 64 | **128** | 减少截断 |
| Epochs | 3 | **10 + Early Stopping** | 找到真正最优 |
| 学习率 | 2e-5 | **1e-5** | 更温和微调 |

命令：
```bash
python src/train.py --pool cls --dropout 0.3 --epochs 10 --lr 1e-5 --max_length 128
```

### 6.2 Loss 函数优化

**Focal Loss** 替代 `class_weight` 的线性加权：

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha[targets] * focal_term * ce_loss
        return loss.mean()
```

### 6.3 换更强的预训练模型

| 模型 | 参数量 | 预期提升 |
|------|--------|----------|
| `bert-base-chinese` | 102M | 基线 ~57% |
| `chinese-roberta-wwm-ext` | 102M | +2~3% |
| `macbert-base-chinese` | 102M | +2~3% |
| `bert-large-chinese` | 335M | +3~5% |

### 6.4 模型结构改进

- **多层特征融合**：取 BERT 最后 4 层加权求和，而非仅用最后一层
- **Attention-based Pooling**：让模型学习每个 token 的重要性权重

### 6.5 数据层面

- **数据增强**：对少数类（证券 0.5%、房产 3.9%、故事 2.1%）做同义词替换/回译
- **标签修正**：人工抽检验证集，修正明显错误的标注（如"小产权房"标为"农业"）
- **动态采样**：`WeightedRandomSampler` 让少数类每轮被抽到概率更高

### 6.6 集成策略

- **模型集成**：3 个不同 seed 的模型投票，稳定提升 1~2%
- **TTA**：测试时文本变换后取平均预测

---

## 七、踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No such file or directory: train.py` | 文件在 `src/` 下，用户在根目录执行 | `python src/train.py` |
| `OSError: Repo id must use alphanumeric chars` | `evaluate.py`/`predict.py` 默认 BERT 路径指向项目外 | 修正 `BERT_PATH = ROOT / "bert-base-chinese" / "bert-base-chinese"` |
| `FileNotFoundError: ../data/val.json` | 相对路径用 `../` 指向了项目外 | 用 `./data/val.json` 或 `data/val.json` |
| `FileNotFoundError: best_mean_weighted` | checkpoint 路径少写 `.pt` 后缀 | `--ckpt_path outputs/checkpoints/best_mean_weighted.pt` |
| `UnicodeEncodeError: 'gbk' codec` | Windows 终端无法编码 `✓`/`✗` | 替换为 `OK`/`NG`/`NA` |
| ModelScope 模型路径转义 | 缓存目录名中 `.` 被转义为 `___` | `Qwen2___5-0___5B-Instruct` |
| Few-shot 无效果 | 0.5B 模型无 in-context learning 能力 | 需换 7B+ 模型或放弃 few-shot |

---

## 八、待完成

- [ ] 训练4 结果补充（dropout 0.3 + lr 1e-5 + epochs 10）
- [ ] Focal Loss 实验
- [x] `nlp_roberta_backbone_base_std` 模型对比（58.86%，优于 bert-base-chinese 2.1%）
- [ ] `chinese-roberta-wwm-ext` 模型对比
- [ ] 数据增强实验
- [ ] 模型集成实验
