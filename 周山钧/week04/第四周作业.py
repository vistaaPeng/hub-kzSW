import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim          # 词向量维度
        self.num_heads = num_heads          # 头数
        self.head_dim = embed_dim // num_heads # 每个头的维度

        # Q K V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # B=批次, T=序列长度, C=词向量维度

        # 1. 线性投影得到 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆成多头：[B, T, 头数*头维度] → [B, 头数, T, 头维度]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 4. 【关键】掩码：把未来位置遮住（解码器必备）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 5. softmax 得到权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 6. 加权求和
        out = attn_weights @ v

        # 7. 拼接多头
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. 最后投影
        out = self.out_proj(out)
        return out


# 2. 前馈网络 FFN
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# 3. 完整 Transformer 层（GPT 解码器单层）
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # 【自注意力 + 残差】
        x = x + self.attn(self.norm1(x), mask)

        # 【前馈 + 残差】
        x = x + self.ffn(self.norm2(x))
        return x


# 测试 
if __name__ == "__main__":
    # 超参数
    BATCH_SIZE = 2
    SEQ_LEN = 10    # 句子长度
    EMBED_DIM = 64  # 词向量维度
    NUM_HEADS = 4   # 多头注意力头数

    # 造输入
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # 构造解码器掩码（挡住未来字）
    mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))  # 下三角为1，上三角为0

    # 创建一层 Transformer
    layer = TransformerLayer(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, hidden_dim=128)

    # 前向传播
    out = layer(x, mask)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("单层 Transformer 运行成功！")