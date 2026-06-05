import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.header_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = q.reshape(batch_size, seq_length, self.num_heads, self.header_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.header_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.header_dim).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.header_dim)

        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device) * float('-inf'), diagonal=1)
        score = score + causal_mask
        attn_weights = F.softmax(score, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.embed_dim)
        return self.out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return nn.Sequential(
            self.linear1,
            nn.GELU(),
            self.dropout,
            self.linear2,
        )(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attn_out = self.attention(x)
        # x = self.norm1(x + self.dropout(attn_out))
        # ff_out = self.feed_forward(x)
        # return self.norm2(x + self.dropout(ff_out))
        return nn.Sequential(
            self.attention,
            self.norm1,
            self.feed_forward,
            self.norm2
        )(x)
