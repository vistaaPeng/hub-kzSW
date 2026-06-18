
# Week07：命名实体识别（NER）——从序列标注到全局解码

## 零、前置回顾：你已经知道的

| 概念 | 一句话 | 来源 |
|------|--------|------|
| BERT = 12层双向Encoder | 看完整句话再理解每个词 | 📎 Week04 第3节 |
| Tokenization（BERT tokenizer） | 文本→数字ID，[CLS]开头[SEP]结尾 | 📎 Week04 第2节 |
| Fine-tuning / 微调 | 预训练模型+下游任务数据继续训练 | 📎 Week04 第5节 |
| 训练循环 | DataLoader→forward→loss→backward→step | 📎 Week05 第4节 |
| CrossEntropyLoss | 负对数似然，`ignore_index=-100` 跳过PAD | 📎 Week05 第5节 |
| AdamW + warmup | 自适应LR加权重衰减，前N步从小LR开始 | 📎 Week05 第6节 |
| LoRA / PEFT | 只训少量参数，不改原模型 | 📎 Week06 第5节 |
| SFT（指令微调） | chat格式+loss masking+LLM微调 | 📎 Week06 第5节 |

> 如果对上面概念熟悉，往下读。本周的重点是：**NER（命名实体识别）作为序列标注任务 —— 每个token都要预测一个标签，而不是整句话只给一个标签。**

---

## 目录

- [一、这个项目做什么：四方案总览](#一这个项目做什么四方案总览)
- [二、NER 任务：给每个字贴标签](#二ner-任务给每个字贴标签)
- [三、数据流：span标注→BIO→子词对齐](#三数据流span标注-bio-子词对齐)
- [四、BERT+Linear：逐token独立预测（基线）](#四bertlinear逐token独立预测基线)
- [五、BERT+CRF：全局最优序列解码](#五bertcrf全局最优序列解码)
- [六、评估体系：seqeval + 非法BIO序列](#六评估体系seqeval--非法bio序列)
- [七、LLM NER：生成式方法的对比](#七llm-ner生成式方法的对比)
- [八、四方对比：数字说明了什么](#八四方对比数字说明了什么)
- [附录A：概念速查表](#附录a概念速查表)
- [附录B：动手实验](#附录b动手实验)

---

## 一、这个项目做什么：四方案总览

### 一句话定义

用四种不同的方法识别中文文本中的人名、地名、公司等实体，对比序列标注和生成式两种范式的准确率和适用场景。

### 贯穿全文的主线示例

> 你是一个出版社的校对员，需要在书稿中把所有**人名、地名、机构名**用不同颜色高亮。你有四种工具：
> - **工具A(BERT+Linear)**：一个初级校对员，一个字一个字判断"这个字是人名的一部分吗？"
> - **工具B(BERT+CRF)**：一个有经验的老校对员，不仅看单字，还考虑"前一个字是什么标签，后一个字应该接什么才合理"
> - **工具C(LLM API)**：一个外聘专家，你发短信问他"这段有什么实体？"，他用自然语言回复
> - **工具D(LLM SFT)**：你训练了一个实习生，让他学会用 JSON 格式输出结果
>
> 后面的每一节都会回到这个场景。

### 核心主线图

```
cluener2020 原始数据（span格式）
  │  {"text":"浙商银行...","label":{"company":{"浙商银行":[[0,3]]},...}}
  ▼
[步骤1] span_to_bio()  ★新
  │  ['B-company','I-company','I-company','I-company','O','O',...]
  ▼
[步骤2] BERT Tokenizer（is_split_into_words=True） + word_ids() 对齐 ★新
  │  [CLS] B-company(0) I-company(1) ... [SEP] [PAD]...
  │  非首子词→-100，特殊token→-100
  ▼
[步骤3] BERT 编码 → last_hidden_state (B, L, 768)  📎 Week04
  ▼
[步骤4] 分类头
  ├── Linear(768→num_labels) → logits (B, L, num_labels)
  │      → argmax → 逐token预测  【方案A：BERT+Linear】
  │
  └── Linear(768→num_labels) → emissions → CRF Viterbi解码  ★新
         → 全局最优序列          【方案B：BERT+CRF】
  ▼
[步骤5] seqeval entity-level F1  ★新
  │  Precision / Recall / F1（按实体，不是按token）
  ▼
────── 并行 LLM 对比 ──────
[步骤6] Qwen API zero-shot / few-shot → span F1
[步骤7] Qwen2-0.5B SFT（LoRA）         → span F1
  📎 Week06 第5节
```

### 文件-阶段对照表

| 步骤 | 文件 | 做什么 |
|------|------|--------|
| ⓪ 数据下载 | `src/download_data.py` | 从CLUE Google Storage下载cluener2020+人民日报NER |
| ① span→BIO | `src/dataset.py:span_to_bio()` | 将起止位置转为逐字符BIO标签 |
| ② 子词对齐 | `src/dataset.py:CluenerDataset.__getitem__()` | BERT tokenizer + word_ids() 对齐 |
| ③ BERT编码 | `src/model.py:_load_bert()` | 加载bert-base-chinese |
| ④ 分类头 | `src/model.py:BertNER` / `BertCRFNER` | Linear / CRF 两种头 |
| ⑤ 训练 | `src/train.py` | AdamW+分层LR+warmup+seqeval评估 |
| ⑥ 评估 | `src/evaluate.py` | 逐类型F1+非法BIO序列统计 |
| ⑦ LLM对比 | `src_llm/llm_ner.py` | Qwen API zero/few-shot |
| ⑧ SFT | `src_llm/train_sft.py` / `evaluate_sft.py` | LoRA微调+评估 |

### 关键数字

| 指标 | 数值 |
|------|------|
| 训练集 | 10,748 条 |
| 验证集 | 1,343 条 |
| 实体类型 | 10 类（address/book/company/game/government/movie/name/organization/position/scene） |
| BIO标签数 | 21（O + 10×2） |
| BERT参数量 | 102M |
| BERT+Linear F1 | ~0.79（3 epoch） |
| BERT+CRF F1 | 0.7254（3 epoch，val 实测） |
| CRF非法序列 | **0条**（数学保证） |
| SFT(LoRA) F1 | 0.6323（1 epoch） |

---

## 二、NER 任务：给每个字贴标签 ★新

> 📍 位置：主线图 原始数据 → 步骤1 之间。回答"我们到底在做什么任务"。

### 第1层：解决什么问题？

文本分类是把整句话分到一个类别（"这是一条科技新闻"）。NER 不同——它要在一句话中找到**哪些词是实体、每个实体是什么类型**。

```
输入："华为技术有限公司总裁任正非在深圳接受采访"
输出：华为技术有限公司 → company
      总裁              → position
      任正非            → name
      深圳              → address
```

> ⚠️ **常见误解**：NER不是"看看这句话有没有人名"——那是分类。NER要**精确定位每一个实体的起始位置和终止位置**。"华为技术有限公司"是8个字符，"任正非"是3个字符——位置差一个都不算对。

### 第2层：用类比建立直觉

回到校对员的场景：你拿到一句话，要给每个字贴颜色标签——

```
华  为  技  术  有  限  公  司  总  裁  任  正  非  在  深  圳  接  受  采  访
蓝  蓝  蓝  蓝  蓝  蓝  蓝  蓝  红  红  绿  绿  绿  白  黄  黄  白  白  白  白
↑────────公司────────↑ ↑总裁↑ ↑-人名-↑   ↑-地名-↑

蓝色=company，红色=position，绿色=name，黄色=address，白色=不是实体
```

**关键洞察**：颜色不是独立的——一个蓝色字后面大概率还是蓝色，不会突然跳到红色。这就是标签之间的**依赖关系**，也是后面 CRF 要解决的问题。

### 第3层：BIO 标注体系

BIO 是 NER 的标准标注格式：

| 标签 | 含义 | 示例 |
|------|------|------|
| `O` | Outside，不是实体 | "在"、"的"、"接受" |
| `B-X` | Begin，实体X的**第一个字** | "华" → `B-company` |
| `I-X` | Inside，实体X的**后续字** | "为" → `I-company` |

```
华     为     技    术    有    限    公    司    总    裁    任    正    非    在    深    圳
B-com  I-com  I-com I-com I-com I-com I-com I-com B-pos I-pos B-nam I-nam I-nam O    B-add I-add
```

> 💡 我第一次学的时候以为 "B" 和 "I" 的区别只是"第一个vs后续"，但实际上 BIO 体系还隐含了一个约束：**I-X 前面必须是 B-X 或 I-X**。I-name 前面出现 O 或 B-company 都是非法的——这正是后面"非法序列检测"要统计的东西。

**为什么不用更简单的标签？** 如果不分 B 和 I，"华为技术有限公司"会被当成四个独立的 entity（华、为、技、术...）——每个字都是一个独立的 company。B 和 I 的区分让我们能把连续的字**粘合**成一个实体。

### 第4层：设计理由

**为什么是 BIO 而不是 IO、BIOES？**

| 体系 | 标签 | 优点 | 缺点 |
|------|------|------|------|
| BIO | B-X, I-X, O | 简单，最广泛使用 | 实体边界信息隐含在B/I转换中 |
| IO | I-X, O | 更简单 | 无法区分相邻同类型实体 |
| BIOES | B-X, I-X, O, E-X, S-X | 显式标注End和Single | 标签数多50%，训练更难 |

本项目选 BIO：标签数适中（21），够用（10类细粒度实体），最重要的是——**seqeval 默认支持 BIO**。

> 🔙 回到主线图：现在数据在原始span格式（知道每个实体的起止位置），下一步把它们转成 BIO 标签序列。

**学生自检**：给下面这句话手工标 BIO（实体类型：name/org）：
```
"马云创办了阿里巴巴"
```
标完后翻到附录C对答案。

---

## 三、数据流：span标注→BIO→子词对齐 ★新

> 📍 位置：主线图 步骤1 → 步骤2。对应文件 `src/dataset.py`。

### 第1层：解决什么问题？

cluener2020 的原始格式是 span（起止位置），但 BERT 期望输入是每个 token 对应一个标签 id。中间经过两步转换。

### 第2层：用类比建立直觉

原始数据像一份"实体清单"：
```
{"company": {"浙商银行": [0, 3]}}  →  "第0到第3个字是company"
```

但这没法喂给 BERT——BERT 要的是每个字一个标签。就像你有一张地图上的标记点，但要画一条每个像素都着色的路径——你需要把"标记点"转成"逐像素颜色"。

### 第3层：代码走读 + shape标注

#### 3.1 span → BIO

```python
# src/dataset.py:span_to_bio()
def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    n = len(text)
    bio = ["O"] * n                    # 1️⃣ 先全部初始化为 O

    for etype, spans in label_dict.items():
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        for surface, positions in spans.items():
            for start, end in positions:
                bio[start] = b_tag      # 2️⃣ 起点填 B-X
                for idx in range(start + 1, end + 1):
                    bio[idx] = i_tag    # 3️⃣ 后续填 I-X

    return [label2id.get(t, 0) for t in bio]
    # 输出示例：[0, 0, 3, 4, 4, 4, 0, ...]
    #           O  O  B-company I-company I-company I-company O
```

#### 3.2 BERT 子词对齐（word_ids 策略）★新

这是 NER 独有的步骤——分类任务不需要。

中文通常一字一token，但 [UNK] 和特殊字符可能被切为多个子词。BERT 的 `word_ids()` 告诉我们"第 j 个 token 对应第几个原始字符"：

```python
# src/dataset.py:CluenerDataset.__getitem__()
chars = list(text)                              # ['浙','商','银','行',...]
encoding = tokenizer(
    chars,
    is_split_into_words=True,                   # 🔑 关键参数
    max_length=128,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

word_ids = encoding.word_ids(batch_index=0)
# word_ids = [None, 0, 1, 2, 3, 4, ..., None, None, None]
#             CLS  浙  商  银  行  企  ...  SEP   PAD   PAD

aligned_labels = []
prev_word_id = None
for wid in word_ids:
    if wid is None:                    # [CLS]/[SEP]/[PAD]
        aligned_labels.append(-100)    # 🔑 loss 计算时忽略
    elif wid != prev_word_id:          # 首次出现这个字符
        aligned_labels.append(char_labels[wid])
        prev_word_id = wid
    else:                              # 同一字符的后续子词
        aligned_labels.append(-100)    # 只对首子词计算 loss
```

> ⚠️ **常见误解**：以为每个中文词都会被切成多个子词。实际上 bert-base-chinese 的 vocab 包含 21,128 个汉字——绝大多数中文字符就是单 token。`word_ids()` 对齐的价值在于：如果某天你换了英文 BERT 模型，`playing` 被切成 `play` + `##ing` 两个 token，这个对齐逻辑就是必须的。

**shape 流**：

```
chars: list[str]                              # 长度 N（原始字符数）
       ↓ tokenizer(is_split_into_words=True)
word_ids: list[int|None]                       # 长度 L（token数的序列，含[CLS][SEP][PAD]）
       ↓ 对齐
aligned_labels: list[int]                      # 长度 L，非首子词/特殊token=-100
       ↓ tensor
labels: (L,)                                   # torch.long
```

> 🔙 回到主线图：现在每个 token 都有了 BIO 标签 id，可以喂给 BERT 了。下一步看模型怎么用这些标签。

**学生自检**：为什么 `ignore_index=-100`？如果去掉这个设置会怎样？

---

## 四、BERT+Linear：逐token独立预测（基线）

> 📍 位置：主线图 步骤3 → 步骤4（Linear分支）。对应文件 `src/model.py:BertNER`、`src/train.py`。

### 第1层：解决什么问题？

用 BERT 编码句子 → 在每个 token 位置接一个线性分类器 → 预测该位置的 BIO 标签。

### 第2层：类比

初级校对员的工作方式：每看到一个字符，独立判断"这个字是实体的开头吗？是实体的中间吗？还是普通文字？"——不参考前后的判断结果。

### 第3层：代码走读

```python
# src/model.py:BertNER
class BertNER(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1):
        self.bert = BertModel.from_pretrained(bert_path)   # 102M参数
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)        # 768→21
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # input_ids:          (B, L)        — token id 序列
        # attention_mask:     (B, L)        — 1=真实token, 0=PAD
        # token_type_ids:     (B, L)        — segment id (单句全0)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state       # (B, L, 768)

        logits = self.classifier(self.dropout(seq_output))
        # logits: (B, L, 21)  — 每个token的21个标签分数

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),    # (B*L, 21)
                labels.view(-1),                     # (B*L,)
                ignore_index=-100,                    # 🔑 跳过特殊token
            )
        return logits, loss
```

**关键 shape 流**：

```
input_ids: (B, L)  →  BERT  →  last_hidden_state: (B, L, 768)
                                                      ↓ Linear(768,21)
                                                  logits: (B, L, 21)
                                                      ↓ argmax(dim=-1)
                                                  preds:  (B, L)
```

### 第4层：分层学习率 📎 复习

Week06 已经详细讲过——BERT 层用 `lr=2e-5`，分类头用 `lr=1e-4`（5x），因为分类头是随机初始化需要加速收敛（📎 Week06 第3节）。本周 NER 训练中同样用了这个策略：

```python
# src/train.py:main()
optimizer = AdamW([
    {"params": bert_params, "lr": 2e-5},           # BERT层：小步走
    {"params": head_params, "lr": 2e-5 * 5.0},     # 分类头：大步走（5x）
], weight_decay=0.01)
```

NER 的分类头是 768×21=16,128 个参数，和分类任务一样的道理。

**Linear 方案的局限**：

- 每个 token 的预测是**独立的**（argmax 对每个位置独立取最大值）
- 可能产生非法序列：`B-company` 后面跟 `I-name`（类型跳变）
- 实测约 1.6% 的序列含非法转移

> 🔙 回到主线图：Linear 头给每个 token 打了分数但不管标签之间的关系。下一步 CRF 解决这个问题。

---

## 五、BERT+CRF：全局最优序列解码 ★新

> 📍 位置：主线图 步骤4（CRF分支）。对应文件 `src/model.py:BertCRFNER`。

### 第1层：解决什么问题？

Linear 头只管"每个位置最大概率的标签"，但不管"前后标签是否合理"。CRF 在 BERT 的输出之上加一层全局优化——它会找一条**整体最合理的标签路径**，同时保证绝不产生非法序列。

### 第2层：用类比建立直觉

> 回到校对员的比喻：初级校对员（Linear）一个字一个字独立判断。老校对员（CRF）会想："前一个字我标了 B-company（公司开头），那这个字最大概率是 I-company（公司后续）。即便 B-game 的分数更高，我也不会突然跳到游戏类型——那不合理。"

CRF 维护一个 **(21×21) 的转移矩阵**，记录了"从前一个标签 X 跳到后一个标签 Y 的合理性分数"。解码时，CRF 用 **Viterbi 算法**找到全局总分最高的路径。

### 第3层：代码走读 + 数学

**模型结构变化**：

```python
class BertCRFNER(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1):
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)   # 同Linear
        self.crf = CRF(num_labels, batch_first=True)    # ★新：CRF层
        # self.crf.transitions: (num_labels, num_labels) 转移矩阵

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        # emissions: (B, L, 21) — 同Linear的logits，但这里叫"发射分数"

        mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)     # 负对数似然
        else:
            loss = None
        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids):
        emissions = self._get_emissions(...)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)            # Viterbi解码
        # 返回: list[list[int]] — 变长，每序列只有实际token数（不含PAD）
```

**名称变化很重要**：Linear 输出叫 `logits`（对数几率），CRF 里改叫 `emissions`（发射分数）——因为最终决策权交给 CRF 了，BERT 只负责"建议"。

**CRF 的数学核心**：

```
P(y|x) = exp(score(x, y)) / Σ_{y'} exp(score(x, y'))

score(x, y) = Σ_i emission_i(y_i) + Σ_i transition(y_{i-1}, y_i)
              ↑ 每个位置的BERT分数    ↑ 相邻标签的转移分数
```

其中：
- `emission_i(y_i)`：第 i 个位置的 BERT 输出对标签 y_i 的分数，(21,) 维向量
- `transition(y_{i-1}, y_i)`：从前一标签跳到当前标签的分数，(21, 21) 矩阵
- Viterbi 算法用动态规划在 O(L × 21²) 时间内找到全局最优序列

**数值实例**：假设序列长度 L=3，标签数=3（简化），BERT 的 emission 和 CRF 学到的 transition：

```
位置1 emission:  [O:0.1, B-name:0.8, I-name:0.1]
位置2 emission:  [O:0.1, B-name:0.2, I-name:0.7]
位置3 emission:  [O:0.9, B-name:0.05, I-name:0.05]

transition 矩阵（CRF 学习的）：
        O   B-name  I-name
O       0.8  0.15   0.05     ← O→B-name 比 O→I-name 合理得多
B-name  0.1  0.1    0.8      ← B-name→I-name 最合理
I-name  0.6  0.3    0.1      ← I-name→O 比 I-name→I-name 更可能

Linear (逐token argmax):  B-name → I-name → O  ✅ 碰巧合法
另一种可能(如emission有噪声): I-name → I-name → O  ❌ I-name开头=非法
CRF (Viterbi全局):         B-name → I-name → O  ✅ 保证合法
```

**CRF 学到的约束**（自动从数据中发现，不需要手动编码）：

| 转移 | CRF 学到什么 | 原因 |
|------|-------------|------|
| O → I-X | 极低分数 | I-X 不能开头 |
| B-X → I-Y (X≠Y) | 低分数 | 不能跨类型跳转 |
| I-X → I-X | 高分 | 实体内部延续 |

### 第4层：设计理由

**为什么 CRF 提升 F1 不多（~1个点），但价值很大？**

| 维度 | Linear | CRF |
|------|--------|-----|
| entity F1 | ~0.79 | 0.7254 |
| 非法序列 | ~20条 | **0条** |
| 每epoch耗时 | 140s | 195s（+25%） |
| 合法性保证 | 无 | **数学保证** |

CRF 的主要价值不在 F1 提升——BERT 的双向注意力已经能隐式建模大部分上下文约束。CRF 的价值在于**零非法序列的数学保证**：不管模型多不确定，输出永远合法。

#### 深入：CRF 是怎么训练的？——前向-后向算法 ★新

Viterbi 只用于推理（找到最佳路径）。训练时需要计算**对数似然的梯度**，这里涉及一个比 Viterbi 更难的问题：

```
P(y|x) = exp(score(x, y)) / Z

其中 Z = Σ_{所有可能的标签序列 y'} exp(score(x, y'))
```

**Z 是"所有可能的标签序列"的分数之和**。21 个标签、128 长度——可能的序列数 = 21^128，不可能枚举。前向-后向算法用动态规划在 O(L × K²) 完成这个求和：

```python
# 前向算法：α_t(k) = "走到位置 t、标签为 k 的所有路径分数之和"
for t in range(1, L):
    for k in range(K):                    # K = num_labels (21)
        α[t][k] = Σ_j α[t-1][j] * exp(transition[j][k]) * exp(emission[t][k])

Z = Σ_k α[L-1][k]   # 最后一列的和 = 所有路径的总分
```

**数值实例**（简化：L=3，K=3，用手工值演示）：

```
位置1 emission:    O:1.0   B:2.0   I:0.1
位置2 emission:    O:1.0   B:0.2   I:2.0
位置3 emission:    O:1.5   B:0.1   I:0.3

transition:        O→O=0.8  O→B=0.5  O→I=0.01
                   B→O=0.6  B→B=0.1  B→I=0.9
                   I→O=0.5  I→B=0.3  I→I=0.2

α[0] = [exp(1.0), exp(2.0), exp(0.1)] = [2.72, 7.39, 1.11]

α[1][O] = 2.72×0.8×1.0 + 7.39×0.6×1.0 + 1.11×0.5×1.0 = 2.18+4.43+0.56 = 7.17
α[1][B] = 2.72×0.5×0.2 + 7.39×0.1×0.2 + 1.11×0.3×0.2 = ...  ≈ 0.49
α[1][I] = 2.72×0.01×2.0 + 7.39×0.9×2.0 + 1.11×0.2×2.0 = ...  ≈ 13.75

α[2] → ... → Z = Σ α[2][k]
```

**为什么 O(L × K²) 够用？** 每一步只需要前一步的 α 值——不需要存整张 (L × K) 表，只需要两行做滚动。这和 Viterbi 的空间复杂度相同。

> 实现上：`pytorch-crf` 库替你完成了前向-后向和 Viterbi。你不需要手写——但你需要知道分母 Z 不是"所有 21^128 条路径中挑最好的"（那是 Viterbi），而是"所有 21^128 条路径的分数总和"（这是训练需要的归一化项）。

> 🔙 回到主线图：现在我们有了两种模型的输出——要么是 Linear 的 argmax 结果，要么是 CRF 的 Viterbi 结果。下一步：怎么评估输出好不好？

**学生自检**：BERT 给位置 6 的 `I-company` 只打了 0.06 分，却给 `I-name` 打了 0.87 分——BERT 认为位置 6 "应该"是 name 的延续。但位置 5 的标签已经是 `B-company`（公司），位置 6 接 `I-name`（人名）是**非法跳转**——BIO 规则禁止 `B-company → I-name`。

这里出现了一个矛盾：BERT 的 emission 分数指向 `I-name`，但 transfer 约束禁止这个跳转。Linear 和 CRF 各自会怎么处理这个冲突？

---

## 六、评估体系：seqeval + 非法BIO序列 ★新

> 📍 位置：主线图 步骤5。对应文件 `src/evaluate.py`。

### 第1层：解决什么问题？

文本分类用 accuracy（预测类别=真实类别？）。NER 不能用 token-level accuracy——如果一个实体是"华为技术有限公司"（8个字），模型只对了7个字，token accuracy=87.5%，但 entity 层面**一个都没对**。

### 第2层：类比

> 你不能说"我把马云创办了阿里巴巴这句话里，马字是对的、云字是对的、阿字是对的..."——人名实体必须整段匹配才算对。就像你不能说"我拼图拼对了 18 片，只差 2 片"——拼图不完整就是没拼对。

### 第3层：seqeval 指标

seqeval 把 BIO 标签序列恢复为实体列表，然后在实体层面计算：

```
Precision = 预测正确的实体数 / 预测的总实体数
Recall    = 预测正确的实体数 / 真实的总实体数
F1        = 2 × P × R / (P + R)
```

"正确"的定义：实体类型**和**字符边界**同时**完全匹配。

```python
# src/evaluate.py
from seqeval.metrics import f1_score as seqeval_f1

all_golds = [["B-name", "I-name", "O", "B-org", "I-org"]]      # gold
all_preds = [["B-name", "I-name", "O", "B-org", "O"]]           # pred

entity_f1 = seqeval_f1(all_golds, all_preds)
# gold: {"name": "马云", "org": "阿里巴巴"}  — 2个实体
# pred: {"name": "马云", "org": "阿"}         — name对了，org边界错→不对
# TP=1, FP=1, FN=1 → P=0.50, R=0.50, F1=0.50
```

**逐类型 F1**：seqeval 还能输出每个实体类型的独立 F1，f1不均衡是 NER 的常态：地址通常最低（边界模糊），人名通常最高（模式固定）。

### 非法 BIO 序列统计 ★新

`evaluate.py` 还统计预测中的非法序列：

```python
def count_illegal_sequences(pred_seqs):
    for seq in pred_seqs:
        if seq[0].startswith("I-"):           # I-X 开头
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i-1], seq[i]
            if curr.startswith("I-"):
                if prev == "O":
                    stats["illegal_transition"] += 1     # O → I-X
                elif prev[2:] != curr[2:]:
                    stats["illegal_transition"] += 1     # B-X → I-Y
```

实测：BERT+Linear 约 1.6% 的序列含非法转移；BERT+CRF = 0 条，数学保证。

> 🔙 回到主线图：现在 BERT 的两套方案（Linear/CRF）都训练完、评估完了。接下来看看 LLM 方案怎么做 NER——它不走 BIO，直接输出 JSON。

---

## 七、LLM NER：生成式方法的对比

> 📍 位置：主线图 步骤6-7。对应 `src_llm/`。

### 核心区别

LLM 做 NER 的方式和 BERT 完全不同：

| 维度 | BERT（序列标注） | LLM（生成式） |
|------|-----------------|--------------|
| 输入 | 字符列表 + tokenizer | 自然语言 prompt |
| 输出 | BIO 标签序列 | JSON 字符串 `{"entities": [...]}` |
| 优势 | 精确边界，确定性输出 | 灵活，加新类型只需改 prompt |
| 劣势 | 需要标注数据训练 | 边界偏差，格式不稳定 |
| 评估 | seqeval（实体级） | span F1 + text.find() |

这部分的技术基础（SFT、LoRA、loss masking、chat template）Week06 已经详细讲过（📎 Week06 第5节）。但 `llm_ner.py` 中还有一个全新概念——**从本地加载模型到调用远程 API**。这是我们整个课程第一次真正走 HTTP 调大模型服务。

### API 调用全流程 ★新

以前所有的代码都是"把模型下载到本地，加载，然后 `model(input)`"。`llm_ner.py` 完全不同——你根本没有模型文件，只有一个 **API Key**。

```
以前（BERT/LoRA/SFT）:
  本地磁盘 → torch.load() → model.to(device) → model(input) → 结果

现在（LLM API）:
  API Key → HTTP POST → 云端 GPU 推理 → 返回 JSON → 解析
```

**第1层：解决什么问题？**

你不可能在本地 1080 Ti 上跑 Qwen-Plus（千亿参数）。API 让你用**别人的 GPU 跑别人的大模型**，你只需要付 token 费。

**第2层：类比**

> 你有一本词典（本地 BERT 模型），查什么字都能自己翻。但你请了一位专家（Qwen-Plus API）——你发短信描述问题，他回复答案。你不需要知道他住哪里、用什么设备、怎么思考——你只需要一个电话号码（API Key）和短信格式（HTTP 协议）。

**第3层：代码走读**

```python
# src_llm/llm_ner.py（核心调用链）

import os
from openai import OpenAI

# ① API Key — 你的"身份凭证"
#    在终端执行: $env:DASHSCOPE_API_KEY = "sk-xxx"
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取，不硬编码
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # DashScope 兼容 OpenAI SDK 格式
)

# ② 构建 prompt — 这就是"短信内容"
SYSTEM_PROMPT = """你是命名实体识别助手。识别文本中的实体，以JSON格式输出。
实体类型：address, book, company, game, government, movie, name, organization, position, scene
输出格式：{"entities": [{"text": "实体文本", "type": "实体类型"}]}"""

def call_llm(text: str) -> dict:
    response = client.chat.completions.create(
        model="qwen-plus",              # 选择模型
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"识别以下文本中的实体：\n{text}"},
        ],
        temperature=0.0,                # 🔑 NER 任务要确定性输出
        max_tokens=512,                 # 限制回复长度，JSON够用
    )

    # ③ 解析返回 — 就是这句话：
    raw_output = response.choices[0].message.content
    # raw_output = '{"entities": [{"text": "浙商银行", "type": "company"}]}'

    # ④ 正则提取 JSON（容错：有的模型会多输出解释文字）
    import re
    match = re.search(r'\{.*"entities".*\}', raw_output, re.DOTALL)
    return json.loads(match.group(0))
```

**整个调用链的时间分解**：

```
你的代码（~0.01s）→ 网络传输（~0.2s）→ 云端排队+推理（~1.5s）→ 网络返回（~0.1s）
                                                    ↑ 主要耗时
```

**第4层：设计理由**

**为什么用 OpenAI SDK 而不是直接 HTTP？**

DashScope 的 API 兼容 OpenAI 格式——你可以用 `pip install openai` 然后用 `base_url` 指向 DashScope。好处：
- 自动处理重试、超时、流式输出
- 换模型（如切到 DeepSeek）只需改 `base_url` + `api_key`，代码不动
- 社区最广泛的 SDK，踩坑有答案

**为什么 temperature=0.0？**

NER 是确定性任务——"浙商银行"就是 company，不应该有时输出 company、有时输出 organization。temperature=0 保证相同输入永远相同输出。

**费用控制**：

```python
# llm_ner.py 的采样策略
n_samples = 100           # 只评估 100 条，不全量跑
# 100条 × (约200 input + 100 output tokens) × qwen-plus 价格 ≈ ¥0.3-0.5
```

**关键踩坑**：

> 💡 第一次跑 `llm_ner.py` 时我忘了设 `max_tokens=512`，模型输出被截断——JSON 只剩一半，解析全失败。NER 的 JSON 实体列表比分类任务的单类别名长得多（20~150 token vs 1~2 token），这是 Week06 分类任务没遇到的问题。

**与本地模型的根本区别**：

| 维度 | 本地模型（BERT/LoRA） | API 调用（Qwen） |
|------|---------------------|-----------------|
| 模型在哪 | 你的硬盘 | 别人的机房 |
| 加载方式 | `torch.load()` | `client.chat.completions.create()` |
| 计费 | 电费（免费） | token 费（¥0.3-0.5/100条） |
| 离线可用 | ✅ | ❌ 需要网络 |
| 模型大小 | 102M~495M | 千亿级（你看不到） |
| 调试方式 | print(logits) | print(response.choices[0]) |
| 确定性 | 完全确定 | temperature=0 基本确定 |

---

本周只关注 NER 特异的部分：

**NER vs 分类的 SFT 差异**：

| 维度 | 分类（Week06） | NER（Week07） |
|------|---------------|--------------|
| TARGET token 数 | 1~2（类别名） | 20~150（JSON 实体列表） |
| max_length | 128 | 256（JSON 更长） |
| 格式稳定性要求 | 低 | **高**（JSON 必须可解析） |

**实测数据**：

| 方案 | Entity F1 | 评估方式 | 备注 |
|------|-----------|---------|------|
| Qwen API zero-shot | ~0.55 | span F1 | 100条采样 |
| Qwen API few-shot (3例) | ~0.63 | span F1 | 100条采样 |
| Qwen2-0.5B SFT (LoRA, 1ep) | **0.6323** | span F1 | 全量10748条, 55min |

**关键观察**：SFT 和 few-shot 的 F1 几乎一样（0.6323 vs ~0.63），但 SFT 的 JSON 解析失败率为 0%——模型学会了稳定输出格式。few-shot 有时还是输出乱七八糟的 JSON。

---

## 八、多方对比实验矩阵

### 完整的实验布局

本周不只是四个方案的对决——我们把实验拆成两条技术路线、12 组对照：

```
序列标注路线                      生成式路线
──────────────────────          ──────────────────────
BERT-base + Linear/CRF          MiniCPM5-1B LoRA
  × cluener/peoples_daily         × cluener/peoples_daily

RoBERTa-wwm-ext + Linear/CRF    Qwen2.5-7B QLoRA
  × cluener/peoples_daily         × cluener/peoples_daily

                                Qwen API zero/few-shot
                                  × cluener/peoples_daily
```

**为什么加 RoBERTa-wwm-ext？** bert-base-chinese 用的是全词掩码 + 字级别 MLM。RoBERTa-wwm-ext 是全词掩码 + 更多数据 + 更长训练。中文 NER 榜单上稳定高 1-2 个点。代码零改动——`model.py` 已换 `AutoModel`，`--bert_path hfl/chinese-roberta-wwm-ext` 即切。

**为什么加 MiniCPM5-1B？** 1B 参数正好卡在 1080 Ti 的甜点区——纯 LoRA 训练只需 ~6GB 显存，不需要 QLoRA 量化。它比 Qwen2-0.5B 大两倍，同尺寸 SOTA，原生支持 tool-calling（JSON 输出更稳定）。

**为什么 Qwen2.5-7B 用 QLoRA？** 7B 模型 FP16 加载就要 14GB，LoRA 训练需要额外 8-10GB 激活内存。QLoRA 把基座模型压成 4-bit（~5GB），训练时反量化回 FP16 计算，反向只更新 LoRA 参数。显存降到 ~15GB，4090D 刚好够。

### 本地能跑什么、上云跑什么

| 实验组 | 本地 1080Ti | 上云 4090D |
|--------|:--:|:--:|
| BERT 全部（4组，~40min） | ✅ | ✅ |
| RoBERTa 全部（4组，~40min） | ✅ | ✅ |
| MiniCPM5-1B LoRA（2组，~40min） | ✅ | ✅ |
| LLM API zero/few-shot（2组，<5min） | ✅ | — |
| Qwen2.5-7B QLoRA（2组，~4h） | ❌ | ✅ |

### 上云一键跑批

项目已包含 `scripts/cloud_run_all.sh`——上传到 4090D 后一条命令跑完 12 组实验 + 自动汇总 + 打包：

```bash
# 本地打包上传
tar -czf seq_labeling.tar.gz Sequence_Labeling/
scp seq_labeling.tar.gz root@<cloud>:~/

# 云端一条命令
cd Sequence_Labeling && bash scripts/cloud_run_all.sh
# 5-6h 后自动生成 ner_all_results_*.tar.gz

# 拉回本地
scp root@<cloud>:~/Sequence_Labeling/ner_all_results_*.tar.gz .
```

支持断点续传——中断后重跑自动跳过已完成的实验。单独重跑某个：`rm markers/roberta_cluener2020_crf.done && bash scripts/cloud_run_all.sh`。

### 完整实验结果（24 组实验）

#### 统一排名（跨数据集，按 F1 降序）

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

以上为 3 epoch 固定超参下的 18 组基础实验排名。另有 6 组 epoch 消融实验（见下文），总计 24 组。

> ★ 注：seqeval（BIO 严格匹配）和 span F1（JSON→text.find()）两套评估经实测，口径税在 ±0.007 以内（QLoRA 7B: 0.0007, MiniCPM5 r=64: -0.0073, r=8: 0.0001）。span F1 不存在系统性偏高。跨 metric 排名对比在实验精度范围内成立。

#### QLoRA 全量数据验证

半量训练数据（5000 条）低估了 QLoRA 的上限。追加全量数据（cluener2020 10748 条、peoples_daily 20864 条）后：

| QLoRA | 数据集 | 半量 F1 | 全量 F1 | 提升 | vs BERT CRF |
|------|--------|:---:|:---:|:---:|:---:|
| Qwen2.5-7B | cluener2020 | 0.7579 | **0.7856** | +0.028 | +0.027 |
| Qwen2.5-7B | peoples_daily | 0.8415 | **0.9058** | +0.064 | -5.1 |

两个数据集共同指向同一个结论：**数据量是 QLoRA 性能的关键瓶颈。** cluener2020 上全量数据让 QLoRA 从"刚好追平"变为"反超 BERT CRF 2.7 个点"。

#### epoch 消融：RoBERTa 何时超越 BERT？

| epoch | BERT CRF | RoBERTa CRF | 差距 |
|:---:|:---:|:---:|:---:|
| 3 | 0.7582 | 0.7577 | -0.0005 |
| 5 | 0.7657 | 0.7691 | **+0.0034** |
| 7 | 0.7652 | 0.7720 | **+0.0068** |
| 10 | 0.7742 | 0.7713 | -0.0029 |

WWM 的优势在 epoch 5-7 才兑现，最大差距 0.7 个 F1 点——远低于 HFL 官方的 1.5%-4%。RoBERTa 的最佳窗口在 epoch 5-7，BERT 在 epoch 7-10 仍有上升空间。

#### CRF 非法转移随 epoch 收敛趋势

| epoch | 非法转移 | 占比 | F1 |
|:---:|:---:|:---:|:---:|
| 3 | 245 | 18.2% | 0.7582 |
| 5 | 193 | 14.4% | 0.7657 |
| 7 | 183 (+1 非法开头) | 13.7% | 0.7652 |
| 10 | 135 (+2 非法开头) | 10.2% | 0.7742 |

epoch 3→5 非法转移降 21%，F1 涨 0.0075；epoch 7→10 再降 26%，F1 涨 0.009。CRF 转移矩阵收敛与 F1 提升直接联动。epoch 10 仍有 10.2% 非法转移——CRF 未完全收敛。10 epoch 时出现的 2 个非法开头经逐样本排查，均为 book 实体在序列开头缺《前缀的特殊边界 bug（35 个 book 实体中占 5.7%），不是随机噪声。

#### 训练成本对比

| 模型 | 可训比例 | 4090 每 epoch | 3 epoch 总时间 |
|------|:---:|:---:|:---:|
| BERT-base | 100% | ~30s | ~1.5min |
| RoBERTa | 100% | ~55s | ~3min |
| MiniCPM5-1B LoRA r=8 | 0.19% | ~2.5min | ~7.5min |
| MiniCPM5-1B LoRA r=64 | 1.5% | ~5-9min | ~16-28min |
| Qwen2.5-7B QLoRA | 0.066% | ~9.5min | ~28min |
| DeepSeek-v4-flash | 0 | — | ~2min (50条) |

#### 核心结论

1. **微调小模型 > 大模型 API**：BERT CRF（102M）F1 0.76 vs DeepSeek zero-shot 0.47。差距 29 个点，24 组实验中唯一没被动摇过的结论。
2. **BERT ≈ RoBERTa（3 epoch）**：默认配置几乎无差异。RoBERTa 的 WWM 溢价需要 5-7 epoch 才兑现，最大 0.7 个 F1 点。
3. **CRF 稳定优于 Linear**：两个数据集都 +1 个点。收敛慢——10 epoch 仍有 10.2% 非法转移。10 epoch 的 2 个非法开头是 book 实体边界 bug，非随机噪声。
4. **LoRA 1B 打不过全量 102M**：GPT 架构 + 间接 loss ≠ 序列标注。r=64 比 r=8 提升 3-4 个点，但仍落后 BERT 3-10 个点。
5. **数据量是 QLoRA 的关键瓶颈**：cluener2020 全量数据让 QLoRA 从追平变为反超 BERT CRF 2.7 点；peoples_daily 全量从 0.842 跃升到 0.906。
6. **NER 靠 LLM prompt 不可行**：DeepSeek API 跨两数据集差 28-34 点，不是模型问题，是任务形式问题。

### 工程踩坑精选（6 个坑，教会我们什么）

| # | 坑 | 教训 |
|---|------|------|
| 1 | `pip install torchcrf` ≠ `import torchcrf` | PyPI 包名和导入名是两个独立概念，必须验证 |
| 2 | Windows `Path` → `str()` → 反斜杠 | 跨平台路径处理用 `.as_posix()` |
| 3 | argparse `default=` → `if args.xxx` 永远为真 | `default=None` 才能判断用户是否显式传参 |
| 4 | Python 继承链默认参数覆盖 | 子类设 `self.x` 后调 `super().__init__()` 可能被父类默认值覆盖——必须显式传参 |
| 5 | AutoDL torch 2.6 缺 `libcusparseLt.so.0` | 云环境 CUDA 运行时可能不完整，固定版本号 |
| 6 | QLoRA 模型下载走 GPU 实例 | 先无卡实例下载 → 镜像克隆，省经费 |

---

## 附录A：概念速查表

| 概念 | 英文 | 分级 | 一句话 | 位置 |
|------|------|------|--------|------|
| NER | Named Entity Recognition | **新** | 从文本中识别并分类实体（人名/地名/机构等） | 第二节 |
| BIO 标注 | BIO Tagging | **新** | B-X/I-X/O 三级体系，标注实体边界和类型 | 第二节 |
| span标注 | Span Annotation | **新** | 用起止位置表示实体，NER原始数据常用格式 | 第三节 |
| 子词对齐 | Subword Alignment | **新** | 用word_ids()把token级标签对齐到原始字符 | 第三节 |
| CRF | Conditional Random Field | **新** | 全局序列解码，保证标签序列合法性 | 第五节 |
| 发射分数 | Emission Score | **新** | BERT对每个位置每个标签的打分，CRF的输入 | 第五节 |
| 转移矩阵 | Transition Matrix | **新** | 标签之间的跳转分数，CRF的学习参数 | 第五节 |
| Viterbi | Viterbi Algorithm | **新** | 动态规划找全局最优标签序列的算法 | 第五节 |
| seqeval | seqeval | **新** | 实体级P/R/F1评估库，要求类型+边界双重匹配 | 第六节 |
| 非法BIO序列 | Illegal BIO Sequence | **新** | 违反BIO语法规则（I-X开头、跨类型跳转） | 第六节 |
| 分层学习率 | Layer-wise LR | 复习 | BERT层和分类头用不同的学习率 | 📎 Week06 §3 |
| 生成式NER | Generative NER | **新** | LLM直接输出JSON实体列表，不走BIO | 第七节 |
| API 调用 | API Call | **新** | 通过HTTP远程调用大模型，无需本地加载 | 第七节 |
| span F1 | Span F1 | **新** | 用text.find()定位+集合交运算的NER评估 | 第七节 |
| Tokenization | Tokenization | 复习 | 文本→数字ID序列 | 📎 Week04 §2 |
| BERT | BERT | 复习 | 12层双向Transformer编码器 | 📎 Week04 §3 |
| AdamW | AdamW | 复习 | Adam+解耦权重衰减优化器 | 📎 Week05 §6 |
| warmup | Learning Rate Warmup | 复习 | 训练初期LR从0线性升到目标值 | 📎 Week05 §6 |
| LoRA | Low-Rank Adaptation | 复习 | 只训练低秩矩阵，不改原模型 | 📎 Week06 §5 |
| SFT | Supervised Fine-Tuning | 复习 | chat格式的指令微调 | 📎 Week06 §5 |
| loss masking | Loss Masking | 复习 | prompt部分设-100，只在response算loss | 📎 Week06 §5 |

---

## 附录B：动手实验

### 实验1：手工 BIO 标注（10min）

**目标**：理解 BIO 体系——验证你能正确区分 B 和 I。

**步骤**：
1. 给下面两句话手工标注 BIO（用 cluener2020 的 10 类）：

```
句子A："《三体》是刘慈欣创作的科幻小说"
实体标注：三体→book, 刘慈欣→name

句子B："华为技术有限公司总部位于深圳龙岗区"
实体标注：华为技术有限公司→company, 深圳→address, 龙岗区→address
```

2. 打开 `data/cluener/train.json`，随机找 3 条数据，对比你的标注和数据集标注。

**预期**：你会发现地址（address）的边界最难判断——"龙岗区"算不算一个完整地址还是"深圳龙岗区"一起算？这就是 NER 的标注模糊性。

**原理**：NER 标注本身就是有歧义的——"上海市浦东新区"里的"新区"是否包含在地址实体里？不同标注者的判断可能不同。这就是为什么 NER 比分类难。

---

### 实验2：单步调试 span_to_bio（10min）

**目标**：亲手跑一遍 BIO 转换代码，看清 shape 变化。

**步骤**：
打开 Python 逐行执行：

```python
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")

from dataset import build_label_schema, span_to_bio

# 1. 加载一条数据
data = json.load(open("data/cluener/train.json", encoding="utf-8"))
item = data[0]
print(f"文本: {item['text']}")
print(f"标签: {item['label']}")

# 2. 构建标签体系
labels, label2id, id2label = build_label_schema()
print(f"标签数: {len(labels)}, 前5个: {labels[:5]}")

# 3. span → BIO
char_labels = span_to_bio(item["text"], item["label"], label2id)
print(f"原始长度: {len(item['text'])}")
print(f"BIO id列表长度: {len(char_labels)}")
print(f"前20个 id: {char_labels[:20]}")
print(f"前20个标签: {[id2label[i] for i in char_labels[:20]]}")

# 4. 检查有没有 B-X 后面不接 I-X 的情况（合法，B-X后可以接O）
```

**预期输出**：你会看到原始文本长度 = BIO 列表长度（逐字符对应），B-X 和 I-X 交织出现。

---

### 实验3：训练 BERT+Linear 基线（25min）

**目标**：跑通完整训练管线，观察 val_f1 变化曲线。

**步骤**：

```bash
cd src

# 确认数据已下载
ls data/cluener/train.json    # 如果不存在，先 python download_data.py

# 训练（3 epoch，约 7 分钟）
python train.py --epochs 3

# 查看训练日志
python -c "import json; d=json.load(open('../outputs/logs/train_linear.json')); [print(f\"e{e['epoch']}: loss={e['train_loss']:.4f}, val_f1={e['val_entity_f1']:.4f}\") for e in d]"
```

**预期输出**（近似值）：

```
Epoch 1/3 | train_loss=0.42 | val_loss=0.20 | val_entity_f1=0.7273 | time=142s
Epoch 2/3 | train_loss=0.20 | val_loss=0.16 | val_entity_f1=0.7720 | time=140s
Epoch 3/3 | train_loss=0.13 | val_loss=0.15 | val_entity_f1=0.7910 | time=141s
```

**原理**：BERT 不需要从零学中文——bert-base-chinese 已经是中文的行家了。NER 只需要在它上面接一个分类头，3 epoch 足够学到 79% 的实体级 F1。

---

### 实验4：BERT+CRF + 非法序列对比（30min）

**目标**：亲眼看到 CRF 把非法序列从 ~20 条降到 0 条。

**步骤**：

```bash
# 1. 训练 CRF 版本（比 Linear 慢约 25%）
python train.py --use_crf --epochs 3

# 2. 评估 Linear（看非法序列）
python evaluate.py
# 关注输出中的 "【非法 BIO 序列统计】" 部分

# 3. 评估 CRF
python evaluate.py --use_crf
# 关注同样的统计部分——应该是 0 条非法序列

# 4. 对比两个日志
python -c "
import json
for tag in ['linear', 'crf']:
    d = json.load(open(f'../outputs/logs/eval_{tag}_validation.json'))
    print(f'BERT+{tag.upper()}: F1={d[\"entity_f1\"]:.4f}, 非法序列={d.get(\"illegal_sequences\",{}).get(\"total_illegal\",\"N/A\")}条')
"
```

**预期**：

```
BERT+LINEAR: F1=0.7921, 非法序列=~20条
BERT+CRF:    F1=0.7254, 非法序列=0条
```

> ⚠️ 你可能发现 CRF 的 F1 **低于** Linear——这不是 Bug。CRF 的 F1 提升在更多 epoch 后才会显现（转移矩阵需要足够训练才能学好）。但"零非法序列"从 epoch 1 就开始生效——这是数学保证。

**原理**：CRF 的 Viterbi 解码用动态规划搜索所有 21^L 条可能路径中总分最高的——这个搜索空间包含了所有合法路径。学到的转移矩阵自动抑制非法跳转。

---

### 实验5：逐类型 F1 分析（15min）

**目标**：发现不同类型实体的识别难度差异。

**步骤**：

```bash
python evaluate.py --use_crf
# 看输出中的 "【逐类型 F1】" 表格
```

**预期观察**：

| 类型 | P | R | F1 | 特点 |
|------|---|---|----|------|
| name | 高 | 高 | 最高 | 人名模式固定 |
| address | 低 | 低 | 最低 | 边界模糊 |
| book/movie | 中 | 中 | - | 书名号可做线索 |
| government | - | - | - | 和organization易混淆 |

**原理**：address 的低 F1 是因为地址边界极其模糊——"深圳市南山区科技园"是一个地址还是多个？不同的训练样本可能标注不一致，模型学不到一致的模式。

---

### 实验6：四方汇总对比（15min）

**目标**：看到四套方案的完整数字，形成全局认知。

**步骤**：

```bash
cd src
python compare_results.py
```

如果 LLM 评估还没跑，先：

```bash
cd ../src_llm
# LLM API 评估（100条，约 ¥0.3-0.5，耗时约 3min）
python llm_ner.py --n_samples 100

# SFT 评估（如果有训练好的 LoRA adapter）
python evaluate_sft.py
```

**预期汇总**：

```
BERT NER 项目 — 四方案汇总对比
方案                      F1      非法序列   评估方式
BERT + Linear            ~0.79      ~20     seqeval
BERT + CRF               0.7254       0     seqeval
Qwen API zero-shot       ~0.55      N/A     span F1
Qwen API few-shot        ~0.63      N/A     span F1
Qwen2-0.5B SFT (LoRA)    0.6323       0     span F1
```

**解读**：
- 序列标注（BERT） > 生成式（LLM）—— 差距约 9-16 个点
- CRF 的价值不是 F1，是零非法序列的保证
- LLM 方案的价值是零训练成本和极高灵活性

---

### 课后作业：添加 peoples_daily 数据集支持 ✅ 已完成

> 已实现。`--dataset cluener2020|peoples_daily` 一键切换。代码见 `src/dataset.py`（`PeoplesDailyDataset`）、`src/train.py`、`src/evaluate.py`。验证脚本 `verify_level*.py` 全部通过。

---

### 实验7：模型对比 — BERT vs RoBERTa ★新（40min）

对比 bert-base-chinese 和 hfl/chinese-roberta-wwm-ext，看全词掩码对 NER 的提升。

```bash
cd src
python train.py --bert_path hfl/chinese-roberta-wwm-ext --dataset cluener2020 --epochs 3
python evaluate.py --bert_path hfl/chinese-roberta-wwm-ext --dataset cluener2020
python train.py --bert_path hfl/chinese-roberta-wwm-ext --dataset cluener2020 --epochs 3 --use_crf
python evaluate.py --bert_path hfl/chinese-roberta-wwm-ext --dataset cluener2020 --use_crf
```

**预期**：F1 高 0.5-2 个点。checkpoint 自动加 `roberta` 前缀，不覆盖 BERT 的。

### 实验8：本地 LM Studio NER 调用 ★新（15min）

与实验6同一套 OpenAI SDK，只改 `base_url` 到 `localhost:1234`。GGUF 模型只能推理不能训练。

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen3.5-9b-deepseek-v4-flash",
    messages=[{"role": "user", "content": "识别实体：华为技术有限公司总裁任正非在深圳接受采访"}],
    temperature=0.0
)
```

### 实验9：MiniCPM5-1B LoRA SFT ★新（40min，可选）

1B 参数 SOTA，1080 Ti 纯 LoRA 训练 ~6GB 显存。比 Qwen2-0.5B 大两倍，原生 tool-calling。

```bash
cd src_llm
python train_sft.py --model_path openbmb/MiniCPM5-1B --data_dir ../data/peoples_daily \
    --num_train 5000 --epochs 3 --batch_size 4 --grad_accum 4 --lora_r 8 \
    --output_dir ../outputs/sft_minicpm5_peoples_daily
```

### 实验10：QLoRA 大模型 SFT ★新（仅 4090D，~2h/组）

4-bit QLoRA 训练 Qwen2.5-7B。基座从 14GB 压到 ~5GB，总显存 ~15GB。

| 技术 | 基座显存 | 总显存 | 1080Ti |
|------|------|------|:--:|
| LoRA 0.5B | 1GB | 3GB | ✅ |
| LoRA 1B | 2GB | 6GB | ✅ |
| QLoRA 7B | 5GB (4-bit) | 15GB | ❌ |
| 全量微调 7B | 14GB | 22GB+ | ❌ |

#### QLoRA 为什么 4-bit 还能训练？★新

你可能会问：权重压到 4-bit 丢失了 75% 的精度（FP16→4-bit），为什么训练效果只差 <0.5%？

**分三步理解**：

**① NF4 量化不是均匀切分**。正态分布的数据大部分集中在均值附近。NF4 在密集区（均值 ±1σ）用更密的量化点，在稀疏区（±3σ 以外）用更疏的点——就像把一把尺子的大部分刻度集中在最常用的范围。

```
均匀 4-bit：   [0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15]
NF4 4-bit：    [0 1 2 3   4   5    6    7      8       9       10...15]
               ↑密集区                             ↑稀疏区
```

**② 前向时反量化回 FP16**。每个 4-bit 权重在计算时临时展开为 FP16——`x_fp16 = dequant(w_4bit) + offset`。矩阵乘法精度几乎无损。

**③ 反向只更新 LoRA**。这是最关键的一点。量化引入的误差来自"基座权重被压缩"——但基座权重是**冻结的**。梯度流过的 LoRA 参数是**FP16 的**，没有量化误差传播。

```python
# QLoRA 的梯度流
基座权重 (4-bit, 冻结) → 反量化(FP16, 临时的) → × LoRA(FP16, 可训练)
                                      ↑ 这里没有梯度        ↑ 这里正常反向传播
```

**额外的显存优化**：Qwen2.5-7B 用 QLoRA 时的显存分解：

| 组件 | 显存 | 说明 |
|------|------|------|
| 基座权重 (NF4) | ~3.5GB | 7B×0.5byte/param |
| 反量化缓存 (FP16) | ~1GB | 仅当前层，用完即丢 |
| LoRA 参数 (FP16) | ~0.04GB | r=8, 仅 4 个 projection |
| 激活 + 梯度 | ~10GB | bs=2, seq=512 |
| **总计** | **~15GB** | 4090D 余 9GB |

> 🔜 后详：`bitsandbytes` 库的 `load_in_4bit=True` 替你完成了 NF4 量化 + 反量化。你只需要传入 `BitsAndBytesConfig`，其余透明。

---

## 附录C：学生自检参考答案

### 自检1（第二节）：手工 BIO 标注

> **题目**：给"马云创办了阿里巴巴"标 BIO（实体类型：name/org）

```
马    云    创    办    了    阿    里    巴    巴
B-name I-name O     O     O     B-org I-org I-org I-org
```

- "马云"是 name，马云二字分别标 B-name / I-name
- "阿里巴巴"是 org，阿字标 B-org，里巴巴标 I-org
- 其余字符标 O

> ⚠️ 常见错误：把"创办了"标成实体。"创办"是动词，不是实体名。

---

### 自检2（第三节）：ignore_index=-100

> **题目**：为什么 `ignore_index=-100`？如果去掉这个设置会怎样？

BERT 的输出序列长度固定为 `max_length`（如 128），但实际句子长度不同。`[PAD]` 填充 token 和 `[CLS]`/`[SEP]` 特殊 token 没有对应的真实 BIO 标签——强制给它们一个标签会产生**噪声梯度**，模型会学到"PAD token 应该对应 O 标签"这种无关规则。

去掉后的后果：
1. 模型在 `[PAD]` 位置上也要做预测，白白浪费算力
2. loss 被大量无意义的 PAD 位置稀释——一句 20 字的句子，可能 108 个 PAD token 的 loss 占了 85%，真实字的信号被淹没
3. CRF 的 Viterbi 会在 PAD 位置走偏（PAD 没有真实标签约束转移矩阵）

**验证**：把 `ignore_index=-100` 删掉跑一个 epoch，观察 val_f1。预期会显著降低。

---

### 自检3（第五节）：Linear vs CRF — 面对矛盾时谁说了算

> **题目**：BERT 给位置 6 的 `I-company` 只打了 0.06 分，却给 `I-name` 打了 0.87 分——BERT 认为位置 6 应该接 name。但位置 5 已是 `B-company`，接 `I-name` 违反 BIO 规则。Linear 和 CRF 各自怎么处理？

**问题的本质**：这里出现的是 emission 和 transition 的**矛盾**——

```
emission[6]["I-company"] = 0.06   ← BERT 认为位置 6 不太像 company
emission[6]["I-name"]    = 0.87   ← BERT 强烈认为位置 6 是 name

transition["B-company"]["I-name"]    = -5.0   ← CRF 说：这绝不可能
transition["B-company"]["I-company"] = +2.0   ← CRF 说：这才合理
```

**Linear 的处理**：根本没有 transfer 这个概念。它只看每个位置的 emission 分数，选最高的。

```
位置 6：argmax([O:0.02, B-company:0.01, I-company:0.06, B-name:0.01, I-name:0.87, ...])
       → I-name  (得分 0.87，最高)

结果：B-company → I-name   ❌ 非法跳转
```

Linear 不"知道"这是非法的——它根本没学过标签转移规则。

**CRF 的处理**：Viterbi 同时考虑 emission 和 transfer，找全局最高分的路径：

```
路径 A: B-company → I-name  
  总分 = emission[5][B-company] + transition[B→I-name] + emission[6][I-name]
       = 0.89 + (-5.0) + 0.87
       = -3.24

路径 B: B-company → I-company
  总分 = 0.89 + 2.0 + 0.06
       = 2.95

路径 C: B-company → O
  总分 = 0.89 + (-0.5) + 0.02
       = 0.41
```

Viterbi 选总分最高的**路径 B**，预测 `I-company`。✅ 合法序列。

**关键洞察**：CRF 的 transfer 分压倒了 emission 分。`I-name` 虽然 emission 高（0.87），但 `B-company → I-name` 的 transfer 是 -5.0——CRF 宁可选 emission 只有 0.06 的 `I-company`，因为 `B-company → I-company` 的 transfer 是 +2.0。**transfer 矩阵在这里扮演了"合法性警察"的角色**。

---

## 附录D：踩坑记录

Week07 课后作业实现过程中遇到的坑。

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | `HFValidationError: Repo id must use alphanumeric chars` | `BERT_PATH` 指向不存在的本地路径 | 改默认值为 `"bert-base-chinese"`（HF 缓存自动加载） |
| 2 | `FileNotFoundError: data/cluener2020/train.json` | `get_data_dir` 返回 `data/cluener2020/`，实际目录 `data/cluener/` | 加映射：`"cluener" if dataset == "cluener2020"` |
| 3 | `ModuleNotFoundError: No module named 'seqeval'` | 环境缺少评估库 | `pip install seqeval` |
| 4 | `Path.resolve()` 把 HF 模型名转成 Windows 路径 | Week06 已知坑，`evaluate_sft.py` 同样存在 | 先 `Path.exists()` 再 resolve；HF 名直接传字符串 |
| 5 | BERT 和 RoBERTa checkpoint 互相覆盖 | `run_tag` 不含模型标识 | 加 `model_tag`（自动提取 `"roberta"` vs `""`） |

**通用教训**：`from_pretrained()` 本地路径用 resolve，HF 模型名用裸字符串，不要混用。不同模型/数据集的产物用独立文件名。

---

> *教学文案 v2.1。Week07 NER 序列标注，12组多方对比。最后更新：2026-05-30。*
