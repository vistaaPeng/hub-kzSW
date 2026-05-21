"""
单向 Transformer 语言模型（字符级）
用法：
    训练：python transformer_lm.py train
    生成：python transformer_lm.py generate
    训练 + 生成（默认）：python transformer_lm.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
import math
import os
import sys

# ==================== 配置 ====================
class Config:
    data_path = "input.txt"          # 训练文本文件
    block_size = 128                 # 上下文长度
    batch_size = 64
    n_embd = 256
    n_head = 8
    n_layer = 6
    dropout = 0.1
    learning_rate = 3e-4
    max_epochs = 10
    eval_interval = 200
    save_ckpt = "lm_checkpoint.pt"
    generate_max_tokens = 200
    temperature = 0.8
    top_k = 40
    vocab_size = None                # 会从数据自动推断

# ==================== 数据 ====================
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

def get_batch(data, block_size, batch_size, device):
    n = len(data)
    ix = torch.randint(0, n - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==================== 模型 ====================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj  = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==================== 训练 ====================
def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"请将训练文本放入 {config.data_path}")

    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = GPT(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    steps_per_epoch = len(train_data) // (config.batch_size * config.block_size)
    global_step = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, config.block_size, config.batch_size, device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % config.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_x, val_y = get_batch(val_data, config.block_size, config.batch_size, device)
                    _, val_loss = model(val_x, val_y)
                print(f"epoch {epoch}, step {global_step}, train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")
                model.train()
        scheduler.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': config,
    }, config.save_ckpt)
    print(f"模型已保存至 {config.save_ckpt}")

# ==================== 生成 ====================
def generate(prompt=None):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config.save_ckpt, map_location=device)
    tokenizer = checkpoint['tokenizer']
    config = checkpoint['config']

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if prompt is None:
        prompt = input("请输入起始文本（直接回车使用默认）: ").strip()
    if not prompt:
        prompt = "A"

    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(
        context,
        max_new_tokens=config.generate_max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
    )
    print("\n--- 生成结果 ---")
    print(tokenizer.decode(output_ids[0].tolist()))

# ==================== 主入口 ====================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "train":
            train()
        elif mode == "generate":
            generate()
        else:
            print("参数错误，可用：train, generate")
    else:
        # 默认模式：先训练再生成
        print("开始训练...")
        train()
        print("\n训练完成，开始生成...")
        generate(prompt="A")  # 可自定义默认提示
