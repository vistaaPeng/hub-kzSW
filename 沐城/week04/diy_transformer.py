from math import sqrt

import torch
import torch.nn as nn


class MyLayer(nn.Module):
    def __init__(self, hidden_size = 768, num_attention_heads = 12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = 4 * self.hidden_size
        # 在 __init__ 中定义所有需要学习的层，然后 forward 中调用它们。
        self.layer_norm_attn = nn.LayerNorm(self.hidden_size)
        self.layer_norm_ffn = nn.LayerNorm(self.hidden_size)

        # QKV 投影与输出投影
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_attn = nn.Dropout(0.1)
        self.dropout_output = nn.Dropout(0.1)

        # FFN 层
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.gelu = nn.GELU()
        self.dropout_ffn = nn.Dropout(0.1)

    # 执行单层transformer层计算
    def forward(self, x, attention_mask=None):
        # 多头自注意力子层（内部已包含 Pre-LN + 残差）
        x = self.self_attention(x, attention_mask)
        # 前馈网络子层（内部已包含 Pre-LN + 残差）
        x = self.feed_forward(x)
        return x

    def self_attention(self, x, attention_mask):
        """
        多头自注意力子层

        x，形状 [B, S, H]，其中 B=batch, S=seq_len, H=hidden_size。
        num_heads = 12（BERT-base），head_dim = H / num_heads = 768 / 12 = 64。
        """
        # 第一步：Pre-LayerNorm
        x_norm = self.layer_norm_attn(x)  # 形状 [B, S, H]

        # 第二步：线性变换生成 Q, K, V
        qkv = self.qkv_proj(x_norm)  # 线性层输入 H，输出 3*H，形状 [B, S, 3*H]
        q, k, v = qkv.chunk(3, dim=-1)  # 每个形状 [B, S, H]

        # 第三步：拆分多头
        # 将 q, k, v 从 [B, S, H] 重塑为 [B, S, num_heads, head_dim]，然后交换 S 和 num_heads 维度，以便并行计算注意力：
        head_dim = self.hidden_size // self.num_attention_heads
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1, 2)  # 形状 [B, num_heads, S, head_dim]
        k = k.view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1, 2)

        # 第四步：注意力分数计算
        """
        - q 形状 [B, num_heads, S, head_dim]
        - k.transpose(-2,-1) 形状 [B, num_heads, head_dim, S]
        - 乘积 scores 形状 [B, num_heads, S, S]，表示每个头下每个查询位置对所有键位置的相似度。
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)

        # 第五步：注意力掩码（Padding Mask）
        """
        在 BERT 中需要屏蔽填充位置（[PAD]），通常传入 attention_mask（形状 [B, S]，值为 0 表示填充，1 表示真实 token）。
        需要扩展 mask 到 [B, 1, 1, S]（广播到 [B, num_heads, S, S]），将填充位置对应的分数设为 -inf（例如 -10000.0）：
        """
        # attention_mask shape: [B, S] (1 for real token, 0 for pad)
        # 扩展为 [B, 1, 1, S] 以广播到 [B, num_heads, S, S]
        if attention_mask is None:
            # 如果用户不传 attention_mask，代码会跳过掩码，此时模型会 attend 到所有位置（包括填充位置，如果存在填充 token 且未提供 mask，会导致错误）。
            # 建议：若无 mask，可以默认为全 1 张量：
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)

        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            scores = scores.masked_fill(extended_mask == 0, float("-inf"))

        # 第六步：Softmax 与 Dropout
        attn_probs = self.softmax(scores)  # 对最后一个维度（键位置）归一化，形状不变
        attn_probs = self.dropout_attn(attn_probs)  # 训练时 dropout

        # 第七步：加权求和得到输出
        """
        - attn_probs：[B, num_heads, S, S]
        - v：[B, num_heads, S, head_dim]
        - 结果 attn_output 形状 [B, num_heads, S, head_dim]
        """
        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, S, head_dim]

        # 第八步：合并多头
        """
        - 先 transpose 得到 [B, S, num_heads, head_dim]
        - 再 view(B, S, H) 恢复为 [B, S, H]
        """
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 第九步：输出线性变换与残差
        """
        现在得到第一个子层的输出 output_after_attn，形状 [B, S, H]。
        """
        attn_output = self.out_proj(attn_output)  # 输出形状 [B, S, H]
        attn_output = self.dropout_output(attn_output)  # 训练时 dropout
        output_after_attn = attn_output + x  # 残差连接，x 是原始的输入（未归一化的）
        return output_after_attn

    def feed_forward(self, output_after_attn):
        # 1. 输入
        x_ffn = output_after_attn  # 形状[B, S, H]

        # 2. Pre-LayerNorm（注意：这里也是先归一化）
        x_ffn_norm = self.layer_norm_ffn(x_ffn)  # 形状 [B, S, H]

        # 3. 第一个线性层（扩展维度）
        """
        形状变为 [B, S, intermediate_size]
        """
        intermediate = self.fc1(x_ffn_norm)  # H -> intermediate_size，通常 4*H = 3072

        # 4. 激活函数（BERT 使用 GELU）
        intermediate = self.gelu(intermediate)  # 形状不变

        # 5. 第二个线性层（压缩回 H）
        ffn_output = self.fc2(intermediate)  # intermediate_size -> H，形状 [B, S, H]

        # 6. Dropout
        ffn_output = self.dropout_ffn(ffn_output)

        # 7. 残差连接
        """
        layer_output 就是这个完整 Transformer 层的最终输出，形状 [B, S, H]。
        """
        layer_output = ffn_output + x_ffn  # x_ffn 是进入 FFN 子层之前的输入（output_after_attn）
        return layer_output


class DiyTransformer(nn.Module):
    def __init__(self, num_hidden_layers=12, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.layers = nn.ModuleList([
            MyLayer(hidden_size=hidden_size, num_attention_heads=num_attention_heads) for _ in range(num_hidden_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

