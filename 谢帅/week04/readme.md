# transformer_pytorch.py 文件总结

用纯 PyTorch `nn.Module` 实现完整的 **BERT Transformer Encoder**，默认配置与 bert-base-chinese 一致（768 hidden、12 heads、12 layers、102.3M 参数）。

## 类结构

| 类名 | 功能 | 对应 diy_bert.py |
|------|------|------------------|
| `MultiHeadSelfAttention` | 多头自注意力层 | `self_attention` |
| `FeedForward` | 前馈网络 (768→3072→768) | `feed_forward` |
| `TransformerEncoderLayer` | 单层 Encoder (注意力+FFN+残差+LayerNorm) | `single_transformer_layer_forward` |
| `TransformerEmbedding` | 词/位置/段落嵌入 + LayerNorm | `embedding_forward` |
| `TransformerEncoder` | 完整模型，组合以上所有组件 | `DiyBert` |

## 数据流

```
input_ids [batch, seq_len]
    ↓
TransformerEmbedding (词嵌入 + 位置嵌入 + 段落嵌入 + LayerNorm)
    ↓
×12层 TransformerEncoderLayer
    ├── MultiHeadSelfAttention (Q·K^T/√d → softmax → ·V → 拼接多头)
    ├── 残差连接 + LayerNorm
    ├── FeedForward (Linear → GELU → Linear)
    └── 残差连接 + LayerNorm
    ↓
├── sequence_output [batch, seq_len, 768]  (所有token的输出)
└── pooler_output   [batch, 768]           ([CLS]位置 → Linear → Tanh)
```

## 核心方法

- **`load_bert_weights(state_dict)`** — 通过参数名映射表，将 HuggingFace BertModel 的权重加载到自定义模型中
- **`from_pretrained(pretrained_path)`** — 类方法，自动读取 config 并加载预训练权重

## 验证结果

加载 bert-base-chinese 预训练权重后，与 HuggingFace 输出对比：

```
sequence_output 最大误差: 3.81e-06
pooler_output   最大误差: 2.62e-06
输出一致: True
```
