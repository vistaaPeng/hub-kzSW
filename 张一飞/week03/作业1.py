import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # 初始化 Transformer 层
        super(TransformerEncoderLayer, self).__init__()

        # 1. 多头自注意力机制
        # batch_first=True 表示输入张量的形状为 (batch_size, seq_len, d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)

        # 2. 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # 3. 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 4. Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 前向传播函数
        # --- 第一个子层：多头自注意力 + Add & Norm ---
        attn_output, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # --- 第二个子层：前馈神经网络 + Add & Norm ---
        ffn_output = self.ffn(x)

        # 再次进行残差连接和层归一化
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


if __name__ == "__main__":
    # 设定超参数
    batch_size = 2
    seq_len = 10  # 序列长度
    d_model = 512  # 词向量维度
    num_heads = 8  # 注意力头数
    d_ff = 2048  # 前馈网络隐藏层维度

    # 实例化 Transformer 层
    transformer_layer = TransformerEncoderLayer(d_model=d_model,
                                                num_heads=num_heads,
                                                d_ff=d_ff)

    dummy_input = torch.rand(batch_size, seq_len, d_model)

    # 前向传播计算
    output = transformer_layer(dummy_input)

    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("模型运行成功！输入输出形状保持一致，符合 Transformer 层的特性。")