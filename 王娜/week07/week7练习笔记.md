# Week7 序列标注（NER）练习笔记

> 本笔记汇总了 `cluener2020`（span格式）与 `peoples_daily`（BIO格式）两个数据集从下载、探索、训练到评估的完整流程，以及 BERT+Linear、BERT+CRF 和 LLM NER 的核心原理对比。

---

## 一、项目结构与数据集对比

```
序列标注项目/
├── data/
│   ├── cluener/           # CLUE benchmark，span格式，10类实体
│   └── peoples_daily/     # 人民日报，BIO格式，3类实体（PER/ORG/LOC）
├── src/
│   ├── download_data.py
│   ├── explore_data.py           # cluener 数据探索
│   ├── explore_data_practice.py  # peoples_daily 数据探索
│   ├── train.py                  # cluener 训练
│   ├── train_practice.py         # peoples_daily 训练
│   ├── train_practice_new.py     # peoples_daily 支持 resume
│   ├── evaluate.py               # cluener 评估
│   ├── evaluate_practice.py      # peoples_daily 评估
│   ├── evaluate_practice_new.py  # peoples_daily 评估（带 CRF 硬约束）
│   ├── dataset.py
│   └── model.py
├── src_llm/
│   ├── llm_ner.py                # cluener LLM NER（zero-shot vs few-shot）
│   └── llm_ner_new.py            # peoples_daily LLM NER（支持 kimi/deepseek）
└── outputs/
    ├── checkpoints/
    ├── figures/
    └── logs/
```

| 对比项 | cluener2020 | peoples_daily |
|--------|-------------|---------------|
| **格式** | span 标注：`{"name": {"张三": [[0,2]]}}` | BIO 标注：`{"tokens": [...], "ner_tags": [...]}` |
| **实体数** | 10 类（address/book/company/game/government/movie/name/organization/position/scene） | 3 类（PER人名 / ORG机构 / LOC地名） |
| **规模** | 训练 10748 / 验证 1343 / 测试 1345 | 训练 ~20864 / 验证 ~2318 / 测试 ~4636 |
| **来源** | Google Storage | GitHub Raw |

---

## 二、数据下载与常见问题

### 2.1 cluener2020
- 来源：`https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip`
- 国内可直接访问，通常不会失败

### 2.2 peoples_daily
- 来源：`https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/...`
- **常见错误**：`The read operation timed out`
- **原因**：GitHub Raw 在国内访问慢，60秒超时
- **解决**：
  - 方法1：终端设置代理后重试
  - 方法2：手动下载三个文件（train/dev/test）并放入 `data/peoples_daily/`
  - 方法3：修改脚本增加超时时间+重试机制

---

## 三、数据探索（explore_data）

### 3.1 核心统计指标

```python
# cluener: span → 需从 label 中提取实体
for etype, spans in label.items():
    for surface, positions in spans.items():
        for start, end in positions:
            entity_length = end - start + 1

# peoples_daily: BIO → 需从 ner_tags 中合并连续 B-/I- 标签
def extract_entities_from_bio(tokens, ner_tags):
    # 遍历标签，遇到 B-X 开始，连续 I-X 结束，合并为一条实体
```

### 3.2 关键可视化

| 图表 | 作用 | 教学意义 |
|------|------|----------|
| **实体类型分布** | 各类实体频次柱状图 | 类别不平衡是NER难点（如 LOC 远多于 PER） |
| **文本长度分布** | 字符数 histogram + max_length 参考线 | P95 长度决定 `max_length` 设定（建议 128） |
| **实体长度分布** | 实体字符数 histogram | 短实体边界识别更难，CRF 优势明显 |

### 3.3 数据集统计（peoples_daily）

- **训练集**：20,864 条，平均长度 46.9 字
- **实体分布**：LOC(16,571) > ORG(9,277) > PER(8,144)
- **P95 文本长度**：97 → 建议 `max_length=128`
- **实体平均长度**：3.2 字

---

## 四、数据格式转换：span ↔ BIO

### 4.1 span → BIO（cluener 需要）

```python
def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    bio = ["O"] * len(text)
    for etype, spans in label_dict.items():
        for surface, positions in spans.items():
            for start, end in positions:
                bio[start] = f"B-{etype}"
                for idx in range(start + 1, end + 1):
                    bio[idx] = f"I-{etype}"
    return [label2id[t] for t in bio]
```

### 4.2 BIO → span（评估/后处理时可能需要）

```python
# 合并连续的 B-X / I-X
def bio_to_span(ner_tags):
    entities = []
    i = 0
    while i < len(ner_tags):
        if ner_tags[i].startswith("B-"):
            start = i
            etype = ner_tags[i][2:]
            i += 1
            while i < len(ner_tags) and ner_tags[i] == f"I-{etype}":
                i += 1
            entities.append((start, i - 1, etype))
        else:
            i += 1
    return entities
```

---

## 五、模型训练（train_practice）

### 5.1 标签体系构建

```python
ENTITY_TYPES = ["PER", "ORG", "LOC"]
labels = ["O"]
for etype in ENTITY_TYPES:
    labels.append(f"B-{etype}")
    labels.append(f"I-{etype}")
# 结果: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
```

### 5.2 Dataset 构建：子词对齐（word_ids）

中文通常一字一 token，但 `[UNK]` 和特殊字符可能产生子词（`##xx`）。对齐策略：

```python
word_ids = encoding.word_ids(batch_index=0)  # [None, 0, 1, 1, 2, ..., None]
aligned_labels = []
prev_word_id = None
for wid in word_ids:
    if wid is None:           # [CLS]/[SEP]/[PAD]
        aligned_labels.append(-100)
    elif wid != prev_word_id: # 首子词 → 使用真实标签
        aligned_labels.append(char_labels[wid])
        prev_word_id = wid
    else:                     # 非首子词 → 忽略
        aligned_labels.append(-100)
```

> `-100` 是 PyTorch `cross_entropy` 的 `ignore_index`，这些位置不参与 loss 计算。

### 5.3 核心训练技巧

| 技巧 | 作用 | 代码体现 |
|------|------|----------|
| **分层学习率** | BERT 层用较小 lr（2e-5），分类头用较大 lr（1e-4）防止破坏预训练参数 | `AdamW([{"params": bert_params, "lr": 2e-5}, {"params": head_params, "lr": 1e-4}])` |
| **Linear Warmup** | 前 10% 步数 lr 从 0 线性增长到目标值，防止初期大梯度破坏 BERT 参数 | `get_linear_schedule_with_warmup(..., warmup_steps=..., num_training_steps=...)` |
| **梯度裁剪** | 限制梯度范数 ≤ 1.0，防止梯度爆炸 | `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| **梯度累积** | 小 GPU 显存下等效大 batch | `loss / grad_accum → backward() → step() every grad_accum` |

### 5.4 loss 不单调下降的原因

| 现象 | 原因 | 对策 |
|------|------|------|
| 前 10% 步 loss 上升 | **Warmup 正常**，lr 从 0 爬坡 | 不用管 |
| 全程锯齿状波动 | batch 内样本方差大（NER 类别极不平衡） | `--grad_accum 2` 等效大 batch |
| loss 不降 + val_f1 不升 | lr 偏大 | `--lr 1e-5 --head_lr_mult 2.0` |

> **关键指标是 `val_entity_f1`，不是 train_loss 的单调性。**

---

## 六、Resume 继续训练（train_practice_new）

### 6.1 使用场景
- 3 个 epoch 后 val_f1 仍在上升，想继续训到 epoch 5/6
- 需从 `best_peoples_daily_linear.pt` / `best_peoples_daily_crf.pt` 恢复

### 6.2 核心实现

```python
# 1. 加载 checkpoint（注意 weights_only=False，PyTorch 2.6+ 默认 True 会报错）
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# 2. 加载模型权重
model.load_state_dict(checkpoint["state_dict"])

# 3. 恢复历史最佳 F1 和已有日志
best_f1 = checkpoint.get("val_entity_f1", 0.0)
start_epoch = checkpoint.get("epoch", 0) + 1

# 4. 读取已有日志追加
with open(log_path, "r") as f:
    log_records = json.load(f)

# 5. 重新创建 optimizer + scheduler（checkpoint 未保存它们的状态）
#    建议降低 lr（默认 resume_lr_mult=0.1），防止破坏已收敛参数
```

### 6.3 使用命令

```bash
# Linear 模型：从 epoch 3 继续训到 epoch 5
python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_linear.pt --epochs 5

# CRF 模型：从 epoch 3 继续训到 epoch 5
python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_crf.pt --epochs 5 --use_crf

# 手动指定更低学习率
python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_linear.pt --epochs 5 --lr 5e-6
```

---

## 七、模型评估（evaluate_practice）

### 7.1 评估指标：entity-level vs token-level

- **token-level accuracy**：逐个 token 算对错，O 标签占比 80%+，指标虚高，无意义
- **entity-level F1**（seqeval）：**整个实体 span 必须完全匹配**才算对，更严格、更有意义

### 7.2 逐类型 F1

```python
from seqeval.metrics import classification_report
print(classification_report(all_golds, all_preds, digits=4))
# 输出 PER / ORG / LOC 各自的 Precision / Recall / F1
```

### 7.3 非法 BIO 序列统计（CRF vs Linear 的核心差异）

```python
def count_illegal_sequences(pred_seqs):
    # illegal_start: 序列以 I-X 开头
    # illegal_transition: B-X/I-X 后面跟 I-Y（X≠Y）
```

| 模型 | 非法序列 | 原因 |
|------|----------|------|
| **BERT + Linear** | 几十~几百条 | softmax 完全无约束 |
| **BERT + CRF (当前)** | **78 条** | 转移矩阵大部分收敛，少数边缘 case |
| **BERT + CRF + 硬约束** | **0 条** | 推理时将非法转移分数设为 -1e9 |

### 7.4 CRF 硬约束（evaluate_practice_new）

无需重新训练，在已有 checkpoint 上推理时修改转移矩阵：

```python
with torch.no_grad():
    # 禁止以 I-X 开头
    for i in range(num_labels):
        if id2label[i].startswith("I-"):
            model.crf.start_transitions[i] = -1e9
    
    # 禁止 O -> I-X
    o_idx = label2id["O"]
    for i in range(num_labels):
        if id2label[i].startswith("I-"):
            model.crf.transitions[o_idx, i] = -1e9
    
    # 禁止 B-X -> I-Y (X≠Y)
    for i in range(num_labels):
        for j in range(num_labels):
            tag_i, tag_j = id2label[i], id2label[j]
            if tag_i.startswith(("B-", "I-")) and tag_j.startswith("I-"):
                if tag_i[2:] != tag_j[2:]:
                    model.crf.transitions[i, j] = -1e9
```

---

## 八、CRF 实现原理（model.py）

### 8.1 库依赖

```bash
pip install pytorch-crf
```

```python
from torchcrf import CRF
self.crf = CRF(num_labels, batch_first=True)
```

### 8.2 CRF 内部可学习参数

| 参数 | 形状 | 含义 |
|------|------|------|
| `start_transitions` | `(num_labels,)` | 序列起始是标签 $i$ 的分数 |
| `end_transitions` | `(num_labels,)` | 序列结束是标签 $i$ 的分数 |
| `transitions` | `(num_labels, num_labels)` | 标签 $j$ → 标签 $i$ 的转移分数 |

**仅新增 63 个参数**（7×7+7+7），计算量增加 20~30%。

### 8.3 训练：前向-后向算法

```python
emissions = classifier(bert_output)   # (B, L, num_labels) 发射分数
mask = attention_mask.bool()          # 屏蔽 PAD

# CRF 不支持 ignore_index，将 -100 替换为 0，再用 mask 屏蔽
labels_crf = labels.clone()
labels_crf[labels_crf == -100] = 0

# 返回对数似然（正值），取负得到 NLLLoss
loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
```

核心公式：

$$P(\mathbf{y}|\mathbf{x}) = \frac{\exp(\text{score}(\mathbf{x}, \mathbf{y}))}{\sum_{\mathbf{y}'} \exp(\text{score}(\mathbf{x}, \mathbf{y}'))}$$

$$\text{score}(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^{T} e_{t, y_t} + \sum_{t=0}^{T} T_{y_t, y_{t+1}}$$

- 分子：真实路径的发射分 + 转移分
- 分母：所有合法路径分数之和（**前向-后向算法**动态规划高效计算）

### 8.4 推理：Viterbi 解码

```python
def decode(self, input_ids, attention_mask, token_type_ids):
    emissions = self._get_emissions(...)
    mask = attention_mask.bool()
    return self.crf.decode(emissions, mask=mask)  # Viterbi
```

与 `argmax` 的本质区别：

| | Linear (argmax) | CRF (Viterbi) |
|---|---|---|
| **决策方式** | 每个 token 独立取最大分数 | 整条序列全局最优 |
| **约束能力** | 无，可能输出 `B-PER → I-ORG` | 有，**永远不会输出非法序列** |
| **计算量** | O(L×K) | O(L×K²)，K为标签数 |

---

## 九、LLM NER 实践（llm_ner_new.py）

### 9.1 整体架构

```
输入文本 → Prompt 构造（zero-shot / few-shot）→ API（Kimi/DeepSeek）→
JSON 解析 → Span 提取 → 与 Gold 对比计算 F1 → 输出报告
```

### 9.2 API 平台配置

支持双平台切换，通过 `--platform` 参数控制：

```python
PLATFORM_CONFIG = {
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "env_key": "MOONSHOT_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
}
```

使用方式：
```bash
python src_llm/llm_ner_new.py --platform kimi --api_key sk-xxxxx --n_samples 10
python src_llm/llm_ner_new.py --platform deepseek --api_key sk-xxxxx --n_samples 10
```

### 9.3 Prompt 设计：zero-shot vs few-shot

#### Zero-shot
- 只给任务描述 + 输出格式约束
- 无示例，完全依赖 LLM 的预训练知识
- `temperature=0.0` 保证输出确定性

#### Few-shot
- 用 **3 个标注样例** 引导模型对齐输出格式
- 理论上比 zero-shot 格式更稳定、准确率更高
- 示例构造：System Prompt → User(示例1文本) → Assistant(示例1输出) → User(示例2文本) → ... → User(真实文本)

### 9.4 关键踩坑：Unicode 转义（\uXXXX）

**现象**：Few-shot 模式下 LLM 输出 `{"text": "\u6c64\u8f9b\u4f2f", "type": "LOC"}` 而不是 `{"text": "汤辛伯", "type": "LOC"}`。

**原因**：`json.dumps()` 默认 `ensure_ascii=True`，导致 few-shot 示例中的中文被编码为 `\uXXXX`。LLM 看到示例后学会了输出 ASCII-safe JSON。

**解决**：
1. 示例生成时用 `json.dumps(..., ensure_ascii=False)`
2. System Prompt 中明确要求：`text 字段请直接输出中文字符，不要使用 Unicode 转义（如 \uXXXX）`

### 9.5 结构化输出解析（pred_spans_from_response）

```python
def pred_spans_from_response(text, response_text):
    # 1. 正则提取 JSON 块（兼容 markdown 代码块）
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    
    # 2. JSON 解析（json.loads 会自动解码 \uXXXX）
    obj = json.loads(json_match.group())
    entities = obj.get("entities", [])
    
    # 3. 校验 + 在原文中定位
    for ent in entities:
        surface = ent.get("text", "")
        etype = ent.get("type", "")
        idx = text.find(surface)  # 取第一次出现的位置
        spans.add((surface, etype, idx, idx + len(surface) - 1))
```

**关键设计**：
- **正则容错**：兼容 ` ```json ` 代码块包裹
- **类型校验**：`etype not in ENTITY_TYPES_EN` 的实体被过滤
- **位置回查**：用 `text.find(surface)` 定位，不依赖 LLM 可能编造的位置

### 9.6 Span-level F1 计算

与 BERT 的 **token-level** 评估不同，LLM 的评估是 **span-level**：

```python
def compute_span_f1(all_golds, all_preds):
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total
    r = tp / gold_total
    f1 = 2 * p * r / (p + r)
```

判定标准：`(surface, type, start, end)` 四元组**完全匹配**才算 TP。

### 9.7 成本控制与分层采样

```python
def sample_records(n, seed=42):
    # 按实体类型分组，每类先取 n // 3 条，再随机补足
    # 避免随机采样导致某些稀有实体类型被遗漏
```

- 默认只跑 **100 条**（成本约 ¥1~3），不跑完整验证集
- 分层采样保证 **尽量覆盖全部 3 类实体**

### 9.8 环境变量设置（Windows）

```cmd
# CMD（临时，当前窗口有效）
set MOONSHOT_API_KEY=sk-xxxxx

# PowerShell（临时）
$env:MOONSHOT_API_KEY="sk-xxxxx"

# 永久设置：系统属性 → 高级 → 环境变量 → 用户变量 → 新建
```

---

## 十、工程踩坑记录

### 10.1 `HFValidationError: Repo id must use alphanumeric chars...`

**原因**：`transformers` 把 Windows 本地路径（含中文、反斜杠）误判为 HuggingFace Hub 的 `repo_id`。路径不存在时，transformers 先尝试按 repo_id 解析，触发格式校验失败。

**解决**：
```python
# 确保路径存在，或者回退到 hub 下载
bert_path = str(args.bert_path.resolve()) if args.bert_path.exists() else "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_path)
```

### 10.2 `torch.load` 报错 `weights_only load failed`

**原因**：PyTorch 2.6+ 默认 `weights_only=True`，对 pickle 反序列化做了严格安全检查。checkpoint 中包含 `numpy` 相关对象类型被拦截。

**解决**：
```python
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
```

### 10.3 终端输出不显示 / 乱码

**原因**：
1. `sys.stdout = io.TextIOWrapper(...)` 在 PowerShell 中会破坏输出缓冲
2. Windows 终端默认 GBK 编码，与 Python UTF-8 不匹配

**解决**：
1. **不要**在代码中重定向 `sys.stdout`
2. 所有 `print` 加 `flush=True`
3. 必要时在终端执行 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`

### 10.4 依赖清单

```bash
pip install torch transformers seqeval pytorch-crf tqdm matplotlib openai
```

---

## 十一、命令速查表

```bash
# ========== 数据 ==========
python src/download_data.py

# ========== 探索 ==========
python src/explore_data.py
python src/explore_data_practice.py

# ========== 训练（peoples_daily）==========
# 从头训练
python src/train_practice.py
python src/train_practice.py --use_crf

# 继续训练（resume）
python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_linear.pt --epochs 5
python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_crf.pt --epochs 5 --use_crf

# ========== 评估（peoples_daily）==========
python src/evaluate_practice.py
python src/evaluate_practice.py --use_crf
python src/evaluate_practice.py --split validation

# 带 CRF 硬约束（非法序列 guaranteed = 0）
python src/evaluate_practice_new.py --use_crf

# ========== LLM NER（peoples_daily）==========
# Kimi
python src_llm/llm_ner_new.py --platform kimi --api_key sk-xxxxx --n_samples 10

# DeepSeek
python src_llm/llm_ner_new.py --platform deepseek --api_key sk-xxxxx --n_samples 10
```

---

## 十一、方法对比与评估结果

### 11.1 peoples_daily 数据集多方对比

| 方法 | Precision | Recall | F1 | 非法序列/解析失败 | 样本数 | 数据来源 |
|------|-----------|--------|----|------------------|--------|----------|
| **BERT + Linear** | **0.9545** | **0.9582** | **0.9563** | 非法序列: 38/2318 | 2318（全量验证集） | `eval_peoples_daily_linear_validation.json` |
| **BERT + CRF** | **0.9564** | **0.9601** | **0.9582** | 非法序列: 27/2318 | 2318（全量验证集） | `eval_peoples_daily_crf_validation.json` |
| **LLM API (Kimi) Zero-shot** | 0.7512 | 0.5016 | 0.6015 | - | 100（采样） | `eval_llm_peoples_daily_kimi.json` |
| **LLM API (Kimi) Few-shot** | 0.8170 | 0.5737 | 0.6740 | - | 100（采样） | `eval_llm_peoples_daily_kimi.json` |
| **LLM API (Deepseek) Zero-shot** | 0.7917 | 0.6552 | 0.7170 | - | 100（采样） | `eval_llm_peoples_daily_deepseek.json` |
| **LLM API (Deepseek) Few-shot** | **0.8519** | **0.6489** | **0.7367** | - | 100（采样） | `eval_llm_peoples_daily_deepseek.json` |
| **Qwen2-0.5B SFT (LoRA)** | 0.6822 | 0.6197 | 0.6494 | 解析失败: 0/100 | 100（采样） | `eval_sft_peoples_daily.json` |

**原始数据来源：**
- **BERT + Linear**: precision=0.954508, recall=0.958162, f1=0.956332
- **BERT + CRF**: precision=0.956415, recall=0.960077, f1=0.958242
- **LLM API (Kimi) Zero-shot**: precision=0.751174, recall=0.501567, f1=0.601504
- **LLM API (Kimi) Few-shot**: precision=0.816964, recall=0.573668, f1=0.674033
- **LLM API (Deepseek) Zero-shot**: precision=0.791667, recall=0.655172, f1=0.716981
- **LLM API (Deepseek) Few-shot**: precision=0.851852, recall=0.648903, f1=0.736655
- **Qwen2-0.5B SFT**: precision=0.682171, recall=0.619718, f1=0.649446

> **说明**：
> - BERT 系列模型在全量验证集（2318条）上评估，LLM 系列在随机采样的100条上评估
> - 非法序列数：BERT+Linear/CRF 在全量验证集上统计
> - 解析失败数：SFT 模型输出 JSON 无法解析的条数

### 11.2 方法对比分析

#### 1. 本地模型 vs LLM API

| 维度 | 本地模型（BERT） | LLM API（Kimi） |
|------|-----------------|-----------------|
| **准确率** | **很高（F1 0.956~0.958）** | 中等（F1 0.60~0.67） |
| **成本** | 零调用成本，仅需显存 | 持续调用成本（约 ¥0.1/条） |
| **稳定性** | 完全可控，输出格式稳定 | 依赖网络和 API 可用性 |
| **数据隐私** | 数据不出本地，安全 | 数据上传第三方，有风险 |
| **推理延迟** | 毫秒级（~200-300条/秒） | 秒级（~10条/分钟） |

#### 2. BERT + Linear vs BERT + CRF

| 维度 | BERT + Linear | BERT + CRF |
|------|--------------|------------|
| **参数增量** | 0 | 63 个转移参数 |
| **计算开销** | 低（~300条/秒） | 稍高（~200条/秒） |
| **非法序列** | 存在（38/2318条，1.6%） | 较少（27/2318条，1.2%） |
| **F1（验证集）** | 0.9563 | **0.9582** |
| **适用场景** | 快速原型、资源受限 | 生产环境、高要求场景 |

#### 3. Zero-shot vs Few-shot（Kimi API）

| 维度 | Zero-shot | Few-shot (3例) |
|------|-----------|---------------|
| **提示词长度** | 短（仅任务描述） | 长（含3个示例） |
| **格式稳定性** | 一般 | 高（示例引导） |
| **Precision** | 0.7512 | **0.8170** |
| **Recall** | 0.5016 | **0.5737** |
| **F1** | 0.6015 | **0.6740** |
| **API 费用** | 较低 | 较高（prompt更长） |

#### 4. Qwen2-0.5B SFT vs BERT + CRF

| 维度 | Qwen2-0.5B SFT | BERT + CRF |
|------|----------------|------------|
| **模型大小** | ~500M 参数 | ~110M 参数 |
| **显存需求** | ≥8GB（FP16） | ≥4GB |
| **F1** | 0.6494 | **0.9582** |
| **输出格式** | JSON 结构化输出 | BIO 标签序列 |
| **解析失败** | 0/100（格式稳定） | - |
| **适用场景** | 需要灵活输出格式 | 标准 NER 任务 |

### 11.3 选择建议

```
                    ┌─────────────────────────────────────┐
                    │         任务需求分析                 │
                    └───────────────┬─────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
    数据隐私敏感            追求最高准确率            需要快速部署
          │                         │                         │
          ▼                         ▼                         ▼
    ┌───────────┐           ┌───────────┐           ┌───────────┐
    │ BERT+CRF  │           │ BERT+CRF  │           │LLM API    │
    │ 或 Qwen2  │           │ (硬约束)  │           │few-shot   │
    │ SFT(本地) │           └─────┬─────┘           └─────┬─────┘
    └─────┬─────┘                 │                       │
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────┐
                         │ 资源约束检查     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              GPU显存≥8GB                  GPU显存<8GB
                    │                           │
                    ▼                           ▼
           Qwen2-0.5B SFT              BERT + CRF
           (更高F1潜力)                (轻量高效)
```

### 11.4 典型场景决策树

1. **小样本场景**（训练数据 < 1000 条）
   - → **Qwen2-0.5B SFT**（LoRA 高效微调，数据效率高）

2. **生产环境**（要求零非法输出）
   - → **BERT + CRF（硬约束）**（保证合法 BIO 序列）

3. **快速验证**（原型开发，无 GPU）
   - → **LLM API few-shot**（5 分钟完成验证）

4. **低延迟推理**（要求 <100ms/条）
   - → **BERT + CRF**（最快的本地方案）

5. **多任务适配**（需同时支持 NER、分类、摘要）
   - → **Qwen2-0.5B SFT**（统一大模型框架）

---

## 十二、关键结论

1. **数据格式**：span 适合人类标注，BIO 适合模型训练。子词对齐用 `word_ids()` + `-100` 忽略非首子词。
2. **训练技巧**：分层学习率 + Warmup + 梯度裁剪是 BERT 微调的标准三板斧。
3. **评估标准**：看 **entity-level F1**，不要看 token-level accuracy。非法序列数是 Linear vs CRF 的量化差异。
4. **CRF 价值**：仅增加 63 个参数，计算量 +20~30%，换来 0 非法序列 + 通常更高的 F1。
5. **Resume 策略**：加载权重后降低 lr（×0.1），追加日志，继续训练到目标 epoch。
6. **LLM NER**：few-shot 通常优于 zero-shot，但需注意 **Unicode 转义问题**（`json.dumps(..., ensure_ascii=False)`）。
7. **成本控制**：LLM 评估采样 100 条即可做有效对比，分层采样保证实体类型覆盖。