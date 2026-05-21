import math
import argparse
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据模块 ───────────────────────────

def load_corpus(pattern="*.txt"):
    """加载匹配模式下的所有文本文件，拼成一个大字符串。"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """构建字符级词表，返回字符->ID 与 ID->字符映射。"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """字符级语言模型数据集。

    每个样本由连续的 seq_len 个字符组成，
    目标是预测下一个字符。
    """

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


# ─────────────────────────── Transformer 模型 ───────────────────────────


class PositionalEncoding(nn.Module):
    """经典正弦位置编码。"""

    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GPTDecoderBlock(nn.Module):
    """GPT 风格的单层解码器块。

    只包含自回归自注意力和前馈网络，适合无条件语言建模。
    """

    def __init__(self, embed_dim, nhead, hidden_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, tgt_mask=None):
        # 自回归自注意力，query/key/value 都来自相同输入 x
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络
        ff = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout2(ff))
        return x


class TransformerLM(nn.Module):
    """GPT 风格的单向 Transformer 语言模型。"""

    def __init__(self, vocab_size, embed_dim, nhead, hidden_dim, num_layers, dropout, max_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        self.blocks = nn.ModuleList(
            [GPTDecoderBlock(embed_dim, nhead, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, tgt_mask=None):
        x = self.token_embed(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x, tgt_mask=tgt_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def generate_square_subsequent_mask(sz, device):
    """生成上三角掩码，保证模型只能看到当前位置之前的 token。"""
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


# ─────────────────────────── 文本生成 ───────────────────────────


def top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
    """对 logits 进行 top-p 过滤，保留累积概率大于 top_p 的最小集合。"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[..., indices_to_remove] = filter_value
    return logits


def sample_sequence(model, prompt, char2idx, idx2char, max_len, temperature, top_p, device):
    """基于 prompt 逐步采样生成文本。"""
    model.eval()
    tokens = [char2idx.get(c, None) for c in prompt]
    tokens = [t for t in tokens if t is not None]
    if len(tokens) == 0:
        raise ValueError("Prompt 中没有有效字符，请检查词表和输入。")

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_len):
            mask = generate_square_subsequent_mask(input_ids.size(1), device)
            logits = model(input_ids, tgt_mask=mask)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            filtered_logits = top_p_filtering(next_logits, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return "".join(idx2char[idx.item()] for idx in input_ids[0])


# ─────────────────────────── 训练 / 评估 ───────────────────────────


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        seq_len = x.size(1)
        mask = generate_square_subsequent_mask(seq_len, device)
        logits = model(x, tgt_mask=mask)

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


# ─────────────────────────── 主函数 ───────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="transformer_lm_best.pt")
    parser.add_argument("--load", default="", help="若提供已保存的模型文件，则加载后可直接生成文本。")
    parser.add_argument("--prompt", default="你好，", help="生成文本时的 prompt。")
    parser.add_argument("--generate", action="store_true", help="训练完成后进行文本生成。")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_gen_len", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # 加载语料并构建词表
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何文本文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"训练集或验证集样本为空，请检查语料长度和 seq_len。"
            f" 当前 seq_len={args.seq_len}, 训练样本={len(train_ds)}, 验证样本={len(val_ds)}。"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.seq_len + args.max_gen_len,
    ).to(device)

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        print(f"已加载模型: {args.load}")

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

    if args.generate:
        if not args.prompt:
            raise ValueError("生成时必须提供 --prompt 参数。")
        text_out = sample_sequence(
            model,
            args.prompt,
            char2idx,
            idx2char,
            max_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print("\n=== 生成结果 ===")
        print(f"Prompt: {args.prompt}")
        print(text_out)


if __name__ == "__main__":
    main()