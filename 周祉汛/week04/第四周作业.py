import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 用pytorch实现一个transformer层

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # 投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换 + 分割多头: (batch, seq_len, d_model) -> (batch, nhead, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 缩放点积注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力加权和
        attn_output = torch.matmul(attn_weights, V)  # (batch, nhead, seq_len, d_k)

        # 合并多头: (batch, nhead, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性层
        output = self.W_o(attn_output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        d_model: 输入/输出维度
        nhead: 注意力头数
        dim_feedforward: 前馈网络隐藏层维度
        dropout: dropout 概率
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout 用于残差连接
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()  # 也可以用 nn.GELU()

    def forward(self, src, src_mask=None):
        """
        src: (batch, seq_len, d_model)
        src_mask: 可选的注意力掩码 (batch, seq_len, seq_len) 或广播形状
        使用 Post-LN 结构: x + Sublayer(x) 然后 LayerNorm
        """
        # 多头自注意力 + 残差连接 + 层归一化
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络 + 残差连接 + 层归一化
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


# ================== 测试代码 ==================
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    dropout = 0.1

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 实例化 Transformer 编码器层
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

    # 前向传播
    output = encoder_layer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    # 应输出: 输入形状: torch.Size([2, 10, 512])
    #        输出形状: torch.Size([2, 10, 512])
