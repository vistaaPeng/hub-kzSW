"""
单层 Transformer Encoder（PyTorch）
形状: (batch, L, d_model) = (B, 128, 768)

结构:
  输入 -> Multi-Head Self-Attention (Q/K/V 三个 Linear)
       -> Add & LayerNorm
       -> FFN (Linear -> GELU -> Linear)
       -> Add & LayerNorm
       -> 输出
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，仅使用 Q、K、V 三个线性层（无独立输出投影 W_o）。"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, d_model)
        attn_mask: 可选，(L, L) 或 (B, L, L)，0 表示屏蔽
        """
        b, l, _ = x.shape

        q = self.W_q(x).view(b, l, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(b, l, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(b, l, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attn_mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return out


class FeedForward(nn.Module):
    """两层 Linear，中间 GELU（常见 d_ff = 4 * d_model）。"""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TransformerEncoderLayer1(nn.Module):
    """
    单层 Transformer（Post-LN，与原始论文一致）:
    x = LayerNorm(x + MHA(x))
    x = LayerNorm(x + FFN(x))
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.mha(x, attn_mask=attn_mask))
        x = self.norm2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    L, d_model = 128, 768
    batch = 2
    num_heads = 12

    layer = TransformerEncoderLayer1(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch, L, d_model)
    y = layer(x)
    assert y.shape == (batch, L, d_model), y.shape
    print("输入:", tuple(x.shape), "输出:", tuple(y.shape))
