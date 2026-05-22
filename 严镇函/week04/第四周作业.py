import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
#Transformer模型

#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        #每个头的维度
        self.head_dim = embed_dim // num_heads
        #每个头的权重
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        #多头拼接
        self.out_linear = nn.Linear(embed_dim,embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        #计算每个头的权重
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        #划分多个头
        q = q.view(B,N,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,N,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,N,self.num_heads,self.head_dim).transpose(1,2)

        #算qk转置
        qk = torch.matmul(q,k.transpose(-2,-1))
        #算softmax和根号
        qk = qk/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        qk = F.softmax(qk,dim=-1)
        #在乘v
        out = torch.matmul(qk,v)
        #多头拼接
        out = out.transpose(1,2).contiguous().view(B,N,C)
        out = self.out_linear(out)
        return out

#前馈网络
class FeedForward(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        #线性层扩大4倍
        self.fc1 = nn.Linear(embed_dim, embed_dim*4)
        #还原
        self.fc2 = nn.Linear(embed_dim*4, embed_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        #开始拼接
        self.attention = MultiHeadAttention(embed_dim,num_heads)
        self.ffn = FeedForward(embed_dim)
        #层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    def forward(self,x):
        x = self.layer_norm(x+self.attention(x))
        #前馈网络
        x = self.layer_norm2(x+self.ffn(x))
        return x

if __name__ == "__main__":
    model = Transformer(embed_dim=512, num_heads=8)
    x = torch.randn(2, 16, 512)        # [B, T, H]
    print(model(x).shape)              # [2, 16, 512]