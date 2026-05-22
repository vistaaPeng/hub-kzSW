# Transformer 单向语言模型总结

## 一、模型介绍

这是一个**字符级 Transformer Causal Language Model**，基于 Decoder-only Transformer 架构，使用 Causal Mask 实现自回归预测。

### 架构组件

| 组件 | 作用 |
|------|------|
| `Token Embedding` | 将每个字符映射为向量，与输出层权重共享 |
| `Position Embedding` | 为每个位置编码顺序信息 |
| `CausalSelfAttention` | 多头自注意力 + 上三角掩码（-inf），确保只能看到左边的历史字符 |
| `TransformerBlock` | Pre-LN 结构：LayerNorm → 注意力 → 残差 → LayerNorm → FFN → 残差 |
| `Output Head` | 线性层将隐藏状态映射回词表大小，预测下一个字符的概率分布 |

### 核心机制 — Causal Mask

```
位置:    0   1   2   3
      ┌───────────────┐
  0   │ 0  -inf -inf -inf │  ← 位置0只能看自己
  1   │ 0   0   -inf -inf │  ← 位置1能看到0和1
  2   │ 0   0    0   -inf │  ← 位置2能看到0,1,2
  3   │ 0   0    0    0  │  ← 位置3能看到所有历史
      └───────────────┘
```

上三角填 `-inf`，softmax 后权重为 0，实现"只能看过去，不能看未来"。

### 采样策略 — Temperature + Top-P

- **Temperature** 控制分布陡峭程度（低温更确定，高温更随机）
- **Top-P（Nucleus Sampling）** 自适应截断低概率尾部，从累积概率达到 p 的最小词集中采样

---

## 二、训练结果

### 训练配置

```bash
python transformer_lm.py --epochs 20 --hidden_size 128 --num_heads 4 --num_layers 2 \
  --seq_len 64 --intermediate_size 256 --batch_size 64
```

| 项目 | 值 |
|------|-----|
| 语料 | 中文金融新闻，237,190 字符 |
| 词表大小 | 2,575 个字符 |
| 模型参数量 | 603,008 |
| 训练/验证集划分 | 95% / 5% |

### 训练过程

```
 Epoch  Train Loss   Train PPL    Val Loss     Val PPL
--------------------------------------------------------
     1      3.9671       52.83      3.8053       44.94  *  ← 最优
     2      3.1303       22.88      3.8475       46.88
     ...
    20      2.5181       12.41      4.2059       67.08
```

- **最佳验证 PPL：44.94**（Epoch 1）
- 训练 PPL 持续下降（52→12），验证 PPL 持续上升（44→67），存在**过拟合**
- 模型参数量大于数据量，建议减小模型或增大 dropout

### 生成示例

```
Prompt: 黄金
生成:   黄金市场价格持续下滑，在股市场的一般可能跟踪...

Prompt: 原油
生成:   原油上涨。沪指数基金的7只基金...
```

---

## 三、模型使用

### 1. 训练

```bash
conda activate study_env

# 小模型（防过拟合）
python transformer_lm.py --epochs 10 --hidden_size 64 --num_heads 2 --num_layers 2 \
  --seq_len 32 --intermediate_size 128 --batch_size 16 --dropout 0.5

# 大模型（需要更多数据或GPU）
python transformer_lm.py --epochs 30 --hidden_size 256 --num_heads 4 --num_layers 4 \
  --seq_len 128 --intermediate_size 512 --batch_size 64
```

### 2. 加载模型生成文本

```python
import torch

device = torch.device("cpu")
ckpt = torch.load("best_transformer_lm.pt", map_location=device, weights_only=False)

# 重建模型
from transformer_lm import TransformerLM
args = ckpt["args"]
model = TransformerLM(
    vocab_size=len(ckpt["char2idx"]),
    hidden_size=args["hidden_size"],
    num_heads=args["num_heads"],
    num_layers=args["num_layers"],
    intermediate_size=args["intermediate_size"],
    max_seq_len=args["seq_len"],
    dropout=0.0,  # 推理时关闭dropout
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 生成
from transformer_lm import generate
text = generate(model, ckpt["char2idx"], ckpt["idx2char"], "黄金价格",
                max_new_tokens=100, temperature=0.8, top_p=0.9)
print(text)
```

### 3. 可调参数

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `temperature` | 低=保守，高=多样 | 0.7~1.0 |
| `top_p` | 采样范围，越低越确定 | 0.8~0.95 |
| `max_new_tokens` | 生成长度 | 50~200 |
