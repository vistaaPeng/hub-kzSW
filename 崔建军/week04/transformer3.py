import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleTransformer(nn.Module):
    """简易Transformer Encoder实现，包含位置编码和一层Encoder。"""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        
        # 1. 词嵌入层：把词ID变成向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码：告诉模型词的位置
        self.pos_encoding = self.create_positional_encoding(1000, d_model)
        
        # 3. 一层Transformer Encoder
        self.encoder = TransformerEncoderLayer(d_model, n_heads, hidden_dim)
        
        # 4. 输出层：预测每个位置的词
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def create_positional_encoding(self, max_len, d_model):
        """创建位置编码矩阵"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, x):
        # x: [batch_size, seq_len] - 输入是词ID序列
        
        # 步骤1：词嵌入
        x = self.embedding(x)  # [batch, seq_len, d_model]
        
        # 步骤2：加上位置编码
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len].to(x.device)
        
        # 步骤3：通过Encoder层
        x = self.encoder(x)  # [batch, seq_len, d_model]
        
        # 步骤4：输出预测
        output = self.output_layer(x)  # [batch, seq_len, vocab_size]
        return output

class TransformerEncoderLayer(nn.Module):
    """Transformer的一层Encoder"""
    
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # 层归一化 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        
        # 1. 多头自注意力 + 残差连接
        attn_output = self.self_attention(x, x, x)  # 自注意力
        x = self.norm1(x + self.dropout(attn_output))  # 残差 + 归一化
        
        # 2. 前馈网络 + 残差连接  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差 + 归一化
        
        return x

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V 的线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出的线性变换
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        # 1. 线性变换
        q = self.w_q(q)  # [batch, seq_len, d_model]
        k = self.w_k(k)
        v = self.w_v(v)
        
        # 2. 分成多个头
        q = q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        k = k.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch, n_heads, seq_len, seq_len]
        
        # 4. softmax得到权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch, n_heads, seq_len, seq_len]
        
        # 5. 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, d_k]
        
        # 6. 拼接所有头
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.d_model)  # [batch, seq_len, d_model]
        
        # 7. 最终线性变换
        output = self.w_o(attn_output)  # [batch, seq_len, d_model]
        
        return output

# 训练示例
def train_example():
    vocab_size = 1000
    model = SimpleTransformer(vocab_size)
    
    # 模拟数据
    batch_size, seq_len = 4, 10
    input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        optimizer.zero_grad()
        
        output = model(input_data)  # [batch, seq_len, vocab_size]
        loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # 测试前向传播
    vocab_size = 1000
    model = SimpleTransformer(vocab_size)
    
    input_data = torch.randint(0, vocab_size, (2, 5))
    output = model(input_data)
    
    print("输入:", input_data)
    print("输出shape:", output.shape)  # [2, 5, 1000]
    
    # 运行训练示例
    print("\n开始训练示例...")
    train_example()
    