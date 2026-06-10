

import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare import *
from model import GPT, GPTConfig


# ─────────────────────────── 训练 / 评估 ───────────────────────────
def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x,y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def train():

    epochs = 2
    batch_size = 64
    seq_len = 128
    embed_dim = 768
    num_layers = 6
    num_heads = 12
    dropout = 0.3
    bias = False
    lr = 1e-3
    val_ratio = 0.05
    corpus = "*.txt"
    save = "best_model.pt"
    args = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "bias": bias,
        "lr": lr,
        "val_ratio": val_ratio,
        "corpus": corpus,
        "save": save
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 数据准备
    text = load_corpus(corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, seq_len)
    val_ds   = CharDataset(val_text,   char2idx, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True, drop_last=True)

    # 模型
    # model init
    model_args = dict(n_layer=num_layers, n_head=num_heads, n_embd=embed_dim, block_size=seq_len,
                    bias=bias, vocab_size=vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)

    total_params = model.get_num_params()
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            best_val_loss = va_loss
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": args,
            }, save)
            
            out_dir = 'ckpt.pt'
            torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': epoch,
                    'best_val_loss': best_val_loss
                }, out_dir)
            print(f"saving checkpoint to {out_dir}") 

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

        

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {save}")
