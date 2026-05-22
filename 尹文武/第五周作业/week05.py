"""
字符级 Transformer LM 训练脚本，含 PPL 计算和 Beam Search 推理。
用法:
    python transformer_lm.py --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────── 数据 ───────────────────────────
def load_corpus(pattern="*.txt"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_pattern = os.path.join(base_dir, pattern)
    print(f"Searching for files: {full_pattern}")

    texts = []
    for path in glob.glob(full_pattern):
        if os.path.getsize(path) == 0:
            print(f"Skipping empty file: {path}")
            continue
        print(f"Reading file: {path}  (size={os.path.getsize(path)} bytes)")
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if content.strip():  # 非空内容才加入
                texts.append(content)
            else:
                print(f"Skipping file with only whitespace: {path}")

    if not texts:
        print("Warning: No non-empty text files found!")
    return "".join(texts)

def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

# ─────────────────────────── 模型 ───────────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(1024, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        x = self.dropout(x)

        # 下三角 mask 保证自回归
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        out = self.transformer(x, mask=mask)
        logits = self.fc(self.dropout(out))
        return logits

# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# ─────────────────────────── Beam Search ───────────────────────────

@torch.no_grad()
def beam_search(model, start_seq, char2idx, idx2char, beam_size=3, max_len=100, device="cpu"):
    model.eval()
    sequences = [(start_seq, 0.0)]  # (sequence string, log_prob)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            x = torch.tensor([[char2idx[c] for c in seq]], device=device)
            logits = model(x)
            probs = torch.softmax(logits[0, -1], dim=-1)  # 取最后一个时间步
            topk_probs, topk_idx = probs.topk(beam_size)

            for i in range(beam_size):
                candidate = (seq + idx2char[topk_idx[i].item()],
                             score + torch.log(topk_probs[i] + 1e-9).item())
                all_candidates.append(candidate)

        # 保留前 beam_size 条
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    return sequences[0][0]

# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="transformer_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text, char2idx, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 测试 Beam Search
    start_seq = "避险"
    result = beam_search(model, start_seq, char2idx, idx2char, beam_size=3, max_len=50, device=device)
    print(f"\nBeam Search 示例: {result}")


if __name__ == "__main__":
    main()
