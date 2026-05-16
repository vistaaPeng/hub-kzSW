# coding: utf8
"""
PyTorch 实现的 Transformer 编码器层（与 BERT / diy_bert.py 结构一致）：
  1. Multi-Head Self-Attention + 残差 + LayerNorm
  2. Feed-Forward (Linear -> GELU -> Linear) + 残差 + LayerNorm

输入/输出形状: (batch, seq_len, d_model)

用法:
    python transformer_layer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (batch, num_heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)，1=可见 0=屏蔽
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)

        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """前馈网络: Linear -> GELU -> Linear"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer Encoder（Post-LN，与 BERT 一致）:
        x -> SelfAttn -> +残差 -> LN -> FFN -> +残差 -> LN
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 子层1: 自注意力
        attn_out = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层2: 前馈
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    """堆叠多层 TransformerEncoderLayer"""

    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


def build_padding_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    根据每条序列的真实长度构造 padding mask。
    返回 (batch, 1, 1, max_len)，有效位置为 1，padding 为 0。
    """
    batch = seq_lens.size(0)
    positions = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
    mask = (positions < seq_lens.unsqueeze(1)).float()
    return mask.view(batch, 1, 1, max_len)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch, seq_len, d_model = 2, 8, 64
    num_heads, d_ff = 4, 256

    # 随机输入（可理解为 embedding 后的序列）
    x = torch.randn(batch, seq_len, d_model)

    # 示例：第 2 条样本只有前 5 个 token 有效
    seq_lens = torch.tensor([8, 5])
    attn_mask = build_padding_mask(seq_lens, seq_len)

    layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    layer.eval()
    with torch.no_grad():
        out = layer(x, attn_mask)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("单层参数量:", sum(p.numel() for p in layer.parameters()))

    encoder = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    encoder.eval()
    with torch.no_grad():
        out_stack = encoder(x, attn_mask)
    print("两层堆叠输出形状:", out_stack.shape)
