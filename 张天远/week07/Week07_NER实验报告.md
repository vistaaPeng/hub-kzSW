# Week07 NER 实验报告

本周围绕中文命名实体识别（NER），对比了五条技术路线——从 BERT+CRF 序列标注到 Qwen2.5-7B QLoRA 生成式微调，再到 DeepSeek API 直接调用。全部实验在 AutoDL 4090D 上完成，固定 3 epoch 统一超参，总计 24 组。

最初从老师给的代码和基线数据起步：

| 方案 | F1 | 备注 |
|------|:---:|------|
| BERT + Linear | ~0.79 | cluener2020，3 epoch |
| BERT + CRF | 0.7254 | cluener2020，3 epoch，0 条非法序列 |
| Qwen API zero-shot | ~0.55 | qwen-plus，100 条采样 |
| Qwen API few-shot | ~0.63 | qwen-plus，3 示例 |
| Qwen2-0.5B SFT (LoRA) | 0.6323 | cluener2020，1 epoch |

带着这个起点，后续的路线选择、模型替换、数据集扩充和 epoch 消融，让实验从最初的 5 组一路膨胀到 24 组。中间有几次是被数据里的意外推着走的——这些在后面结合具体发现再展开。

---

## 实验设计

**数据集**：
- `cluener2020`：10 类细粒度实体（人名、地址、公司、游戏…），10748/1343 条（训练/验证）
- `peoples_daily`（人民日报）：3 类实体（PER/ORG/LOC），20864/2318 条。和 cluener2020 形成"简单 vs 复杂"对照

**五条技术路线**：

| 路线 | 模型 | 参数量 | 训练方式 | 实验数 |
|------|------|:---:|------|:---:|
| 序列标注 + Linear | BERT / RoBERTa | 102M | 全量微调 | 4 |
| 序列标注 + CRF | BERT / RoBERTa | 102M | 全量微调 | 4 |
| LLM + SFT (LoRA) | MiniCPM5-1B | 1083M | LoRA r=8, r=64 | 4 |
| LLM + SFT (QLoRA) | Qwen2.5-7B | 7621M | 4-bit QLoRA | 2 |
| LLM API | DeepSeek-v4-flash | — | zero/few-shot prompt | 4 |

路线选择不是一步到位的。最开始只有 BERT 和 Qwen API 两条线。加 RoBERTa 是因为 HFL 官方报告说它的全词掩码在中文 NER 上比 BERT 高 1.5-4 个点——代码零改动就能切换，不如顺带验证。Qwen2-0.5B 打平 few-shot 之后换了 Qwen2.5-7B QLoRA（7B 是 0.5B 的 14 倍，但全量加载需要 14GB 显存，用 QLoRA 4-bit 量化压到 5GB，4090D 刚好够）。MiniCPM5-1B 是顺手加的——它用 LoRA 刚好能跑在本地的 1080 Ti 上，填补了 102M 到 7B 之间的参数量空白（不过后面所有实验全部上云统一跑 AutoDL 4090D，本地卡实际没用到）。Qwen API 换 DeepSeek 是想验证"换一个顶尖模型会不会不一样"。

**评估口径**：
- 序列标注模型：seqeval entity-level F1（BIO 位置严格匹配）
- 生成式模型 + API：span F1（JSON 输出 → `text.find()` 近似定位）

两套评估严格度不完全一样。seqeval 要求每个字符位置都对，span F1 只要求实体文本能找到。为了量化这个差异，我对三个模型分别跑了同一份预测结果在两套评估下的 F1：

| 模型 | seqeval | span F1 | 口径税 |
|------|:---:|:---:|:---:|
| QLoRA 7B cluener2020 | 0.7740 | 0.7733 | +0.0007 |
| MiniCPM5 r=64 cluener2020 | 0.7336 | 0.7408 | -0.0073 |
| MiniCPM5 r=8 cluener2020 | 0.7083 | 0.7082 | +0.0001 |

口径税在 ±0.007 以内，方向不固定——span F1 并不像直觉担心的那样系统性偏高。`text.find()` 定位和 BIO 逐字符标注在所有样本上完全等价（零条 span 全对但 seqeval 不对的样本）。结论：**排名表中跨 metric 的对比在实验精度范围内成立。** 后面提到 QLoRA 7B "追平" BERT CRF 时，不需要额外加星号。

---

## 基础实验：18 组排名

所有实验在 AutoDL 4090D 上跑，固定 3 epoch，batch_size 32（BERT/RoBERTa）或相应配置。

> 以下为基础实验的 18 组统一排名。完整 24 组实验中，另有 6 组 epoch 消融实验单独展示（见扩展实验）。

| # | 模型 | Head/r | 数据集 | F1 | 评估 |
|:---:|------|:---:|--------|:---:|:---:|
| 1 | RoBERTa | CRF | peoples_daily | 0.9573 | seqeval |
| 2 | BERT | CRF | peoples_daily | 0.9568 | seqeval |
| 3 | BERT | Linear | peoples_daily | 0.9493 | seqeval |
| 4 | RoBERTa | Linear | peoples_daily | 0.9468 | seqeval |
| 5 | MiniCPM5-1B | r=64 | peoples_daily | 0.8551 | span F1 |
| 6 | Qwen2.5-7B QLoRA | SFT | peoples_daily | 0.8415 | span F1 |
| 7 | MiniCPM5-1B | r=8 | peoples_daily | 0.8132 | span F1 |
| 8 | BERT | CRF | cluener2020 | 0.7582 | seqeval |
| 9 | Qwen2.5-7B QLoRA | SFT | cluener2020 | 0.7579 | span F1 |
| 10 | RoBERTa | CRF | cluener2020 | 0.7577 | seqeval |
| 11 | RoBERTa | Linear | cluener2020 | 0.7537 | seqeval |
| 12 | BERT | Linear | cluener2020 | 0.7485 | seqeval |
| 13 | MiniCPM5-1B | r=64 | cluener2020 | 0.7260 | span F1 |
| 14 | MiniCPM5-1B | r=8 | cluener2020 | 0.6939 | span F1 |
| 15 | DeepSeek-v4-flash | few-shot | peoples_daily | 0.6239 | span F1 |
| 16 | DeepSeek-v4-flash | zero-shot | peoples_daily | 0.5471 | span F1 |
| 17 | DeepSeek-v4-flash | few-shot | cluener2020 | 0.4821 | span F1 |
| 18 | DeepSeek-v4-flash | zero-shot | cluener2020 | 0.4747 | span F1 |

> 表中 #1–#14 为训练实验，#15–#18 为 DeepSeek API 对照。注意表是按 F1 排序的统一排名，行号不等于实验叠加顺序——叠加逻辑见末尾决策全景。

---

## 核心发现

### 1. BERT 和 RoBERTa，3 epoch 下打成平手

```
BERT CRF  cluener2020:   0.7582
RoBERTa CRF cluener2020: 0.7577  Δ = -0.0005

BERT CRF  peoples_daily:  0.9568
RoBERTa CRF peoples_daily: 0.9573  Δ = +0.0005
```

差距在统计噪声级别。102M 参数、相同架构，全词掩码（WWM）的预训练优势在这个设置里完全没体现。跟 HFL 官方声称的 1.5-4 个点差距对不上。

但这是 3 epoch 的结论。WWM 的增益可能需要更长的训练才能兑现——后面做了 epoch 消融专门验证。

### 2. CRF 稳定优于 Linear，但幅度不大

老师基线里 CRF 比 Linear 低了 6 个点——那个结果一直让我觉得不对劲。自己重跑之后翻过来了：

| 数据集 | Linear | CRF | Δ |
|------|:---:|:---:|:---:|
| cluener2020 | 0.7485 | 0.7582 | +0.010 |
| peoples_daily | 0.9493 | 0.9568 | +0.008 |

两个数据集上 CRF 都比 Linear 高约 1 个点。标签转移约束确实有用，但幅度不大——可能因为数据集本身就不算难（peoples_daily 只有 3 类实体，cluener2020 虽然 10 类但边界大多清晰）。至于为什么老师的基线里 CRF 偏低——可能跟代码版本或超参有关，我后来没有深究，但作为初始动机它足够驱动后面的追问了。

### 3. 不微调的 LLM API，28-34 个点的鸿沟

```
BERT CRF (102M, 全量微调):             0.7582
Qwen2.5-7B QLoRA (7B, 参数高效微调):     0.7579
DeepSeek-v4-flash (API few-shot):       0.4821
DeepSeek-v4-flash (API zero-shot):      0.4747
```

这是整轮实验里最稳的结论：不管数据集多简单、模型多大、few-shot 示例给得多好，LLM API 不做微调就是做不了 NER。peoples_daily 上 BERT CRF 已经到 0.96，DeepSeek few-shot 只能到 0.62。换了其他顶尖模型（DeepSeek）也一样——差距不是"模型不够强"的问题，是"不微调就做不了"的问题。

LLM API 在简单数据集上确实好一些（peoples_daily F1=0.62 vs cluener2020 F1=0.48），few-shot 的提升也更明显（+0.077 vs +0.007）。但这只是从"完全不可用"变成"稍微没那么不可用"，跟微调模型 28-34 个点的差距没有任何本质变化。

跟 Week06 文本分类的结论一致：BERT（57%）> Qwen few-shot（16%）。两条实验线指向同一个东西：结构化预测任务，微调小模型 > 大模型 prompt。

### 4. LoRA 1B 打不过全量 102M

| MiniCPM5-1B | r=8 | r=64 | Δ | vs BERT CRF |
|:---:|:---:|:---:|:---:|:---:|
| cluener2020 | 0.6939 | 0.7260 | +0.032 | -0.033 |
| peoples_daily | 0.8132 | 0.8551 | +0.042 | -0.102 |

r=8→64 把可训参数从 2M 拉到 16.5M，收益明显（+3-4 个点）。但离 BERT 全量微调还有 3-10 个点的差距。r=64 这个实验就是被 r=8 的结果逼出来的——"LoRA 参数太少"是最直觉的解释，那就加一组看看。

但即使 r=64 也追不上。直觉上 1B > 102M，为什么打不过？我拆了三个原因：GPT 架构用 LM loss 间接学 JSON 输出，信号密度远低于 BIO 标签的直接监督；LoRA 只更新 1.5% 的参数，BERT 全量更新 100%；1B 模型的注意力头数可能不够同时维护"理解任务 + 分析文本 + 控制输出格式"三个约束。参数量不是一切——任务形式和目标函数的匹配度更重要。老师基线里 Qwen2-0.5B 跟 few-shot 打平，也是同一个道理。

### 5. QLoRA 7B：跨数据集表现分化

```
cluener2020:  F1=0.7579 (5000条) → F1=0.7856 (全量10748条) → 从追平变为反超
peoples_daily: F1=0.8415 (5000条) → F1=0.9058 (全量20864条)

| QLoRA cluener2020 | 数据量 | F1 | vs BERT CRF |
|------|:---:|:---:|:---:|
| 半量 | 5000 | 0.7579 | -0.0003 |
| **全量** | **10748** | **0.7856** | **+0.0274** |

| QLoRA peoples_daily | 数据量 | F1 | vs BERT CRF |
|------|:---:|:---:|:---:|
| 半量 | 5000 | 0.8415 | -11.5 点 |
| 全量 | 20864 | 0.9058 | -5.1 点 |
| BERT CRF | 20864 | 0.9568 | — |
```

cluener2020 上半量数据时 QLoRA 7B 靠着 0.066% 的可训参数（5M）和 4-bit 量化追平了 BERT CRF。但和 peoples_daily 一样，半量配置低估了 QLoRA 的潜力——全量 10748 条数据训完后 F1 从 0.758 跳到 0.786，扣除评估口径税 0.0007 后等效 seqeval 约 0.785，明确超过 BERT CRF 的 0.758，超出 2.7 个点。

peoples_daily 上也是类似：半量 0.842，全量 0.906，劣势从 14 点缩到 5 点。

两个数据集的共同信号：**数据量是 QLoRA 性能的关键瓶颈。** 大模型的迁移能力在充足数据下才能充分释放——半量数据严重低估了 QLoRA 的上限。

### 6. 训练成本

| 模型 | 可训参数 | 每 epoch | 3 epoch |
|------|:---:|:---:|:---:|
| BERT-base | 102M (100%) | ~30s | ~1.5min |
| RoBERTa | 102M (100%) | ~55s | ~3min |
| MiniCPM5 LoRA r=8 | 2M (0.19%) | ~2.5min | ~7.5min |
| MiniCPM5 LoRA r=64 | 16.5M (1.5%) | ~5-9min | ~16-28min |
| Qwen2.5-7B QLoRA | 5M (0.066%) | ~9.5min | ~28min |
| DeepSeek-v4-flash | 0 | — | ~2min (50-100条) |

BERT 和 RoBERTa 是性价比天花板——1.5-3 分钟跑完，效果还是最好的。MiniCPM5 和 QLoRA 投入了更多算力，没换来对应比例的提升。

---

## 扩展实验：被数据逼出来的追问

基础实验跑完，排名表很清楚。但有几个结果让我没法收手。每个追问都是被数据里的异常点推着走的。

### 追问一：epoch 消融 — WWM 到底什么时候生效？

**为什么做这个？** 基础实验里 BERT 和 RoBERTa 打成平手，跟 HFL 官方 1.5-4% 的增益对不上。同时 CRF 在 3 epoch 时还有 245 条非法转移（18.2%）——转移矩阵根本没收敛。如果 CRF 没学好，RoBERTa 靠 WWM 学到的实体边界优势就无从发挥。假设是：多训几轮，RoBERTa 会反超。

在 cluener2020 上，BERT CRF 和 RoBERTa CRF 分别从头训练 epoch=5/7/10。

| epoch | BERT CRF | RoBERTa CRF | 差距 |
|:---:|:---:|:---:|:---:|
| 3 | 0.7582 | 0.7577 | -0.0005 |
| 5 | 0.7657 | 0.7691 | **+0.0034** |
| 7 | 0.7652 | 0.7720 | **+0.0068** |
| 10 | 0.7742 | 0.7713 | -0.0029 |

假说成立——WWM 的增益需要更多 epoch 才能兑现。epoch 3 时 BERT 还微微领先，epoch 5 RoBERTa 开始反超，差距拉到 epoch 7 的 +0.0068。

但最大差距也只有 0.7 个 F1 点。HFL 报告里的 1.5-4% 大概率来自最优超参（更长 warmup、分层学习率调优），我的固定超参只榨出不到 1 个点。工程上看的话——如果你不想花时间调参，RoBERTa 和 BERT 在默认配置下几乎等同。

另外注意到 BERT epoch=7→10 跳了 +0.009，RoBERTa 却微微回退（0.772→0.771）。两者的训练动力学不一样：BERT 在 CRF 收敛后开始在训练集上"死记"，验证集偶然跳升；RoBERTa 的特征空间更复杂，epoch 7 之后开始过拟合。RoBERTa 的最佳 epoch 窗口大概在 5-7 之间，比 BERT 更窄。

### 追问二：CRF 转移矩阵什么时候收敛？

**为什么做这个？** epoch 消融跑的时候顺便盯上了非法转移的数量。如果转移矩阵的收敛和 F1 提升是联动的，就能解释"为什么多训有用"的底层机制。

记录了 epoch 3/5/7/10：

| epoch | 非法转移 | 占比 | F1 |
|:---:|:---:|:---:|:---:|
| 3 | 245 | 18.2% | 0.7582 |
| 5 | 193 | 14.4% | 0.7657 |
| 7 | 183 (+ 1 非法开头) | 13.7% | 0.7652 |
| 10 | 135 (+ 2 非法开头) | **10.2%** | 0.7742 |

几点观察：

CRF 收敛速度和 F1 提升直接联动。epoch 3→5 非法转移降 21%，F1 涨 0.0075；epoch 7→10 再降 26%，F1 涨 0.009。转移矩阵的收敛确实是 F1 提升的主要推动力。

但 10 epoch 后还有 10.2% 的非法转移——CRF 的转移矩阵并未完全收敛。对比 peoples_daily 上 3 epoch 就降到 1.3%（30/2318 条，数据来源：`eval_peoples_daily_crf_validation.json`），数据集的复杂性对 CRF 收敛速度影响巨大。这也意味着如果继续训练到 epoch 15 或 20，F1 可能还有上升空间。

最让我意外的是非法开头。epoch 3 时是 0，epoch 7 出现 1 个，epoch 10 变成 2 个——训练越久反而越不规范？我一开始的理解是模型在"冒险"：用非法转移换更高的发射分。但后来对 epoch 10 做了逐样本排查，发现不是这么回事。

两个非法开头完全一样——都是位置 0 的 `I-book`，gold 都是 `B-book`。具体样本：

```
#507: "刺杀卡斯特罗的638种方法》的原著者。…"
      gold: B-book, I-book, I-book, …  pred: I-book, I-book, I-book, …
#743: "采光及大门定位的请示》获得了天柱县政府办公室的批复。…"
      gold: B-book, I-book, I-book, …  pred: I-book, I-book, I-book, …
```

共同特征：书名在句首、裸标题开头（缺《前缀）、实体跨度很长。在所有 35 个 book 实体中占了 5.7%，不是偶然。

根因是训练数据分布偏差。训练集里 book 几乎总是跟着《书括号出现在句子中间——CRF 学到的转移模式是 `O → B-book`，但在序列开头没有前文 O 作为触发条件，`start → B-book` 和 `start → I-book` 两个初始概率都没有足够的学习信号，某个优化步里偶然漂向了 `start → I-book`。这不是模型"学会冒险"，而是 CRF 在罕见场景下的训练盲区。

### 追问三：QLoRA 的 0.000 — debug 全记录

**为什么做这个？** peoples_daily 上 QLoRA 7B 的 F1=0.000。不是"很差"，是完全退化——所有输出都是 `{"entities": []}`。

训练 loss 在 epoch 2 直接归零。第一反应是实体稀疏触发了退化解——模型发现输出空 JSON 是 LM loss 最低的策略。我给 system prompt 加了"请务必至少输出一个实体"（`--prompt_extra`），重新训了一次。loss 还是 epoch 1 归零。假说推翻。如果只是 prompt 太宽松导致的退化，强制输出应该有缓解——没有缓解说明训练数据本身有问题。

开始查代码。`train_sft_qlora.py` 里的 `QLoRADataset`：

```python
class QLoRADataset(SFTDataset):
    def __init__(self, ..., dataset="cluener2020", prompt_extra=""):
        self.dataset = dataset        # ← 这里设了 "peoples_daily"
        self.prompt_extra = prompt_extra
        super().__init__(data, tokenizer, max_length)  # ← 没传 dataset！
```

`super().__init__()` 调父类 `SFTDataset.__init__()`，父类默认值是 `dataset="cluener2020"`。子类先设了 `self.dataset = "peoples_daily"`，然后父类 `__init__` 立刻用 `"cluener2020"` 把它覆盖。

后果：`__getitem__` 里 `self.dataset` 实际是 `"cluener2020"`，走了 `record_to_target_cluener()`。这个函数找 `record["label"]`，但 peoples_daily 的数据结构是 `record["tokens"]` + `record["ner_tags"]`，没有 `label` 字段。所有训练样本的 target 都变成了 `{"entities": []}`，模型学会的唯一输出就是空 JSON。

修一行：`super().__init__(data, tokenizer, max_length, dataset=dataset)`。

这个 bug 在 cluener2020 实验里完全没暴露——因为默认值恰好对上了。Python 继承链里默认参数覆盖，静默无报错，是我在这轮实验里踩过的最隐蔽的坑。

修完后重训的 QLoRA peoples_daily（标准 prompt）F1=0.842，parse_fail=0。离 BERT CRF 的 0.957 还差 12 个点——接下来就是追问"是不是数据量不够"。

### 追问四：数据量是瓶颈吗？

**为什么做这个？** QLoRA peoples_daily 修复后 0.842，一个重要差异是 BERT 用了全量 20864 条数据，QLoRA 只用了 5000 条。如果 4 倍数据能缩小大部分差距，答案就是"数据量"；如果提升很小，答案就是"架构/量化上限"。

加了一组全量数据实验（`--num_train -1`）。结果：F1=0.906，提升 6.4 个点，差距从 -11.5 缩到 -5.1。数据量是重要瓶颈，但不是唯一的——剩下 5 个点可能是 QLoRA 量化的精度损失，也可能是生成式架构在简单数据集上的固有上限。

peoples_daily 做完之后，同样的问题自然延伸到 cluener2020：QLoRA 在 cluener2020 上也是 5000/10748 的半量配置，如果全量数据能释放更多潜力，之前"追平 BERT CRF"的结论可能低估了 QLoRA 的上限。

加了一组 cluener2020 全量数据实验。结果：F1 从 0.758 跳到 0.786，扣除评估口径税 0.0007 后等效 seqeval 约 0.785，反超 BERT CRF 的 0.758 达 2.7 个点。两个数据集共同指向同一个结论：**数据量是 QLoRA 性能的关键瓶颈，半量数据严重低估了大模型的上限。**

### 追问五：DeepSeek 跨数据集对比

**为什么做这个？** 基础实验里 DeepSeek 只在 cluener2020 上跑了。cluener2020 的结果太惨了（0.47），我忍不住想看看换到更简单的 peoples_daily（只有 3 类实体）它能不能挣扎出点什么。

| 数据集 | zero-shot | few-shot | few-shot 收益 |
|------|:---:|:---:|:---:|
| cluener2020 (10类) | 0.4747 | 0.4821 | +0.007 |
| peoples_daily (3类) | 0.5471 | 0.6239 | +0.077 |

简单数据集确实帮了 LLM——zero-shot 高了 7 个点，few-shot 高了 14 个点，few-shot 的收益也从几乎为零变成了 +0.077。但绝对值依然没法用：peoples_daily 上 few-shot 也只能到 0.62，BERT CRF 是 0.96。

不管数据集多简单、换哪个 API，不做微调就是做不了 NER。差距不是 5 个点、10 个点，是 30+ 个点的系统性问题。

---

## 哪些实验没做（以及为什么）

不是所有想法都值得跑。以下是被我否决掉的：

| 提议 | 否决理由 |
|------|------|
| 分层学习率调优 | epoch 消融已证明多训练自然收敛，调 lr 不是关键瓶颈 |
| 对抗训练 FGM/PGD | 和 NER 主线无关，更适合独立研究 |
| MiniCPM5 全量微调 | 需要 ~22GB 显存；架构劣势不因全量微调改变 |
| RoBERTa peoples_daily epoch 消融 | cluener2020 消融足够回答核心问题 |

---

## 决策全景：从基线到 24 组

```
老师基线（cluener2020 单数据集，4 方案）
    │  BERT+Linear ~0.79  vs  BERT+CRF 0.7254（反常！CRF 更低？）
    │  Qwen2-0.5B SFT 0.6323  vs  API few-shot ~0.63（打平！模型太小？）
    │
    ├─ 追问: CRF 为什么不如 Linear? → 训练不足，转移矩阵没收敛
    │   追问: BERT 是最优基座吗? → 加 RoBERTa（HFL 报告高 1.5-4%）
    │   追问: 0.5B 太小，换什么? → Qwen2.5-7B QLoRA（4-bit，4090D 刚好够）
    │   追问: 参数量有空白? → 加 MiniCPM5-1B（1080 Ti 本地跑）
    │   追问: Qwen API 验证完? → 换 DeepSeek-v4-flash
    │   追问: 单数据集够吗? → 加 peoples_daily（简单 vs 复杂对照）
    │
    └─── 14 组基础实验
            │
            ├── [数据] MiniCPM5 r=8 太弱，可训参数只有 2M
            │    └── 加 2 组 r=64（14→16）
            │
            ├── [数据] BERT ≈ RoBERTa，跟 HFL 报告对不上
            │    └── 加 6 组 epoch 消融（16→22）
            │         同时追踪 CRF 转移矩阵收敛
            │
            ├── [数据] QLoRA peoples_daily F1=0.000
            │    ├── 假说：数据稀疏退化解
            │    ├── 验证：强制输出 → 失败 → 假说推翻
            │    ├── 排查：继承链默认参数覆盖 bug
            │    └── 修复后：F1=0.842
            │
            └── [追问] 数据量是瓶颈吗？
                 ├── +1 组 peoples_daily 全量数据（22→23）
                 │   结果：F1=0.906，差距 -5.1 点
                 └── +1 组 cluener2020 全量数据（23→24）
                     结果：F1=0.7856，反超 BERT +2.7 点

            └── [否决] 考虑过但没做的
                 ├── 分层学习率调优 → 多训练已自然收敛，调 lr 不是关键瓶颈
                 ├── 对抗训练 FGM/PGD → 和 NER 主线无关
                 ├── MiniCPM5 全量微调 → 架构劣势不因全量微调改变
                 └── RoBERTa peoples_daily epoch 消融 → cluener2020 消融足够
```

24 组里大部分是被数据里的具体异常推着走的，但也有几组是"万一有用呢"驱动的——RoBERTa 是因为 HFL 的报告、MiniCPM5 是填补参数量空白、DeepSeek 是想换一个顶尖模型试试。剩下的，从 r=64 到 epoch 消融到 QLoRA 全量数据，全是被一个具体的反常数据点逼出来的。

---

## 错误分析：SFT 模型到底错在哪？

从 MiniCPM5 r=64 和 QLoRA 7B 的评估详情里，归纳了五类典型错误。

### 1. 实体类型混淆（name ↔ position 是最大痛点）

| 混淆对 | MiniCPM5 r=64 | QLoRA 7B |
|------|:---:|:---:|
| name ↔ position | 56 次 | 56 次 |
| name → organization | 16 次 | 15 次 |
| company → position | 14 次 | 12 次 |
| organization → name | 13 次 | 12 次 |
| position → government | 13 次 | — |

人名、职位、组织这三类实体的语义边界天然模糊。"博洛尼"到底是一个足球俱乐部名（organization）还是"博洛尼队"的简称（name）？这种歧义别说模型，人也纠结。

### 2. 边界粘连

"中信地产营销管理部副经理梁丽红"——gold 要求拆成三个独立实体（company + position + name），但 LLM 经常输出为一个整体的 position。生成式模型天然偏向"整块输出"，不像 BIO 可以对每个字符独立决定。这是结构性的劣势，跟模型大小无关。

### 3. 漏标 — low recall 的主因

模型倾向于"保守输出"——只标最确定的实体，宁可漏掉也不冒险标错。

| 案例 | gold | pred |
|------|------|------|
| "全国人大财政经济委员会主任委员石秀诗向第十一届全国人大常委会作《" | 全国人大常委会(government) | **漏掉**。模型标了"第十一届全国人大常委会"但漏了旁边的"全国人大常委会"——同一句中两个相似实体，模型选了长的、丢了短的 |
| "首先关门大吉的，是THQ旗下的BigHugeGames工作室，他们曾是《国家崛起(" | 《国家崛起((game) | **漏掉**。模型标了"《国家崛起"(game)但漏了残缺的"《国家崛起((" ——标点符号不完整导致 text.find() 失败 |
| "深发展...东兴证券研究表明" | 深发展+东兴证券(company) | 只标了东兴证券——深发展被漏掉 |

漏标的本质是模型对"不确定"实体采取了保守策略。BIO 标签天然偏向低 recall（每个 O 标签都是一个"不标"的决策），但没有 LLM 那么极端——LLM 可以在整个 JSON 里干脆不写这个实体，而 BIO 至少会为每个字符输出一个标签。

### 4. 过标 — 无中生有

| 案例 | gold | pred |
|------|------|------|
| "8月2日，上海市规划局公布55号土地出让公告，出让松江大型居住社区泗泾基地a地块" | 上海市规划局(government) + 大型地块(address) | 额外标了"松江大型居住社区泗泾基地"(address)——一个比 gold 更短的地址版本，边界错误还被当成了独立实体 |
| "新美钞很难仿冒" | 无实体 | `美(LOC)`——单字"美"被强行标注为地名 |
| "上海木材行业协会红木专业委员会副会长贾德界" | 长职位名(position) | 额外拆出"副会长"(position) + "上海木材行业协会红木专业委员会"(organization)——过度拆分 |

过标在 peoples_daily 上更常见：模型学会了"找到就标"的策略后，在没有上下文线索时倾向于把高频实体词（PER/LOC/ORG）往任何可能的字符串上套。这也附带解释了评估口径税在 MiniCPM5 r=64 上为什么是负数——过标产生的碎片在 seqeval 里能拿部分分数，在 span F1 里因为 (text, type) 不匹配被完全扣掉，导致 span F1 反而偏高 0.007。

### 5. 幻觉 — r=64 独有的过拟合现象

| 案例 | gold | pred |
|------|------|------|
| "下午的行程从马久邑出发，行程为马久邑-富美邑-磻溪-古生-海舌-沙村-喜洲" | 8 个 scene | 8 个**全部标为 address**——类型全错，但数量和边界都对 |
| "moma、泰特现代、赫什霍恩博物馆也将她的作品永久收藏" | 3 个 scene | 全标为 organization 且凭空增加"藏家"(position)——多了实体、换了类型 |

第一个案例暴露了 r=64 的过拟合特征：模型记住了"地名列表"的模式（X-X-X-X 格式），但学串了类型——location-like 的地名被错误归类为 address。第二个案例则是纯粹的幻觉——"藏家"（收藏家）不是任何实体类型。

r=64 比 r=8 更大的 LoRA 参数量不仅增强了学习能力，也放大了过拟合偏差。16.5M 参数对 5000 条训练数据来说太多了——模型记住了训练集里"地名列表很多是 address"的模式，但泛化不到 scene 这个类型。

---

## 工程踩坑

7 个坑，每个都教了一课：

| # | 问题 | 根因 | 教训 |
|---|------|------|------|
| 1 | `pip install torchcrf` ≠ `import torchcrf` | PyPI 上有两个 CRF 包：`TorchCRF`（无 batch_first）和 `pytorch-crf`（有 batch_first），`pip install torchcrf` 装的是前者 | pip 包名和导入名是两个独立概念，装完必须验证 |
| 2 | Windows `Path` → `str()` → 反斜杠 | argparse `type=Path` 在 Windows 上 `str()` 产生 `\`，HF 不认 | 跨平台路径用 `as_posix()` |
| 3 | `--data_dir` 默认值屏蔽自动推导 | argparse 有 `default=` 时 `if args.data_dir` 永远为真 | `default=None` 才能判断用户是否显式传参 |
| 4 | Linux 文件名大小写 | `TorchCRF` ≠ `torchcrf` | 排查后确认根因仍是坑 1 |
| 5 | RoBERTa 加载报 torch 版本限制 | transformers 新版要求 torch≥2.6，AutoDL 缺 `libcusparseLt.so.0` | monkey-patch 绕过 |
| 6 | QLoRA 模型下载卡 GPU 实例 | AutoDL hf-mirror 不稳定 | ModelScope 云内下载 → 无卡实例先拉 → 克隆 |
| 7 | QLoRADataset 继承链默认参数覆盖 | 子类 `super().__init__()` 未传 `dataset`，父类用默认值覆盖 | Python 继承链里最坑的 bug 类型之一 |

---

## 从这轮实验里攒下的经验

1. **异常数据驱动实验扩展。** 大部分新增实验都是被一个具体异常逼出来的——CRF 为什么不如 Linear、BERT 为什么跟 RoBERTa 打平、QLoRA 为什么输出全是空 JSON。少数几组（RoBERTa、MiniCPM5、DeepSeek）是出于"万一有用呢"的预判加的，但整体上，17/24 的实验增量来自数据反馈而非提前计划。

2. **反向结论比正向结论更有信息量。** "BERT ≈ RoBERTa"纠正了我对 WWM 的预判，"1B LoRA 打不过 102M 全量"纠正了我对参数量的直觉——这些比"CRF > Linear"更值钱。Week06 和 Week07 联合验证的"微调小模型 > 大模型 API"则是一条跨任务泛化的结论。

3. **排查顺序：代码 → 数据 → 模型。** QLoRA F1=0 时我先花了几个小时做退化假说验证，根因其实是一行代码。以后遇到类似问题，先查代码再想理论解释。

4. **消融实验必须有明确假说。** epoch 消融不是随便跑几个 epoch——前提是 CRF 未收敛（245 条非法转移），预期是 WWM 有效 → RoBERTa 上升斜率比 BERT 陡。结果要么验证要么推翻，不能模棱两可。

5. **marker 断点续传是救命稻草。** 24 组实验里至少 7 次修复 bug 或加新实验需要重跑。marker 自动跳过已完成的部分，只重跑改动的那一组——省时间，保数据完整性。

---

## 跑完之后的碎碎念

### 序列标注 vs 生成式：到底该用哪个？

NER 的核心需求是精确的实体边界。两个架构的处理方式完全不同——BIO 对每个字符独立决策，JSON 是靠 LM 把整段输出"顺"出来，边界靠事后 `text.find()` 近似定位。

实验数据支持序列标注。cluener2020 上 BERT CRF 的 seqeval F1 是 0.758，QLoRA 7B 的 span F1 也是 0.758——但 seqeval 要求每个字符位置都对，span F1 只要实体文本能找到。同一套严格度下，序列标注有结构性的边界精度优势。

工业级 NER 的话，BERT+CRF 目前还是最稳的选择：快（1.5 分钟）、准（F1 0.76/0.96）、参数少（102M）。QLoRA 7B 在 cluener2020 上确实追平了 BERT，但训练成本高 18 倍——除非你真的需要 LLM 的灵活性（新增实体类型只改 prompt，不用重新标注），否则不划算。

但整轮跑完，有几个在实验途中没想清楚、做完才意识到的事：

**最优方案不是固定的，取决于数据集。** 一开始的直觉是"BERT+CRF 就是 NER 的标准答案"，peoples_daily 上的数据也确实支持这个判断——BERT CRF 0.96，QLoRA 全量 0.91，差了 5 个点。但 cluener2020 上反过来：BERT CRF 0.758，QLoRA 全量 0.786。不是"BERT 一定更好"或"LLM 更有潜力"，而是数据集特征在决定谁赢——简单数据集上 BERT 的 BIO 精度优势碾压一切，复杂数据集上 LLM 的迁移能力开始兑现。

**数据量的收益不是线性的。** cluener2020 上 QLoRA 从 5000 条到 10748 条，数据量只翻了一倍，但效果从"刚好追平 BERT"跳到了"反超 2.7 个点"。peoples_daily 上 4 倍数据换来 6.4 个点。这跟直觉完全不一样——直觉上数据量翻倍就应该有翻倍的收益，但实际上收益曲线是加速的：半量数据严重低估了大模型的上限，只有给足数据，迁移能力才真正释放。

**微调 vs 不微调，鸿沟大到没有例外。** 这是 24 组实验里唯一一个从头到尾没有被任何数据动摇过的结论。换模型没用、换数据集没用、换 prompt 没用、换 few-shot 样本没用。从最开始 Qwen API 的 0.55，到最后 DeepSeek 的 0.48——永远差 28-34 个点。NER 这种结构化预测任务，prompt 就是不够——不是"暂时不够"，是架构层面就不够。

---

## 待办

- [x] BERT vs RoBERTa epoch 消融（3/5/7/10）
- [x] CRF 转移矩阵收敛追踪
- [x] DeepSeek-v4-flash 跨数据集对比
- [x] QLoRA peoples_daily bug 修复 + 重新训练
- [x] QLoRA peoples_daily 全量数据验证（20864 vs 5000）
- [x] QLoRA cluener2020 全量数据验证（10748 vs 5000）
- [x] Epoch 10 非法开头根因分析
- [ ] 分层学习率实验——但 epoch 消融已证明多训自然收敛，优先级低
- [ ] MiniCPM5-1B 全量微调 vs LoRA——但架构性结论已有，没必要验证
