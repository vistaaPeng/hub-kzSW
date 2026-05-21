"""
基于 Transformer 的字符级语言模型。

用法:
    python transformer_lm.py --epochs 20
    python transformer_lm.py --num_layers 4 --num_heads 4 --d_model 256
"""

import math
import argparse
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════
#  一、数据层（与 language_model.py 保持一致）
# ══════════════════════════════════════════════════════════════════

def load_corpus(pattern: str = "*.txt") -> str:
    """读取当前目录所有 .txt 文件并拼接成一段文本。"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text: str):
    """
    根据文本构建字符级词表。

    :return: (char2idx, idx2char) 两个互逆映射字典
    """
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """滑动窗口字符数据集，每个样本为 (输入序列, 目标序列)。"""

    def __init__(self, text: str, char2idx: dict, seq_len: int):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ══════════════════════════════════════════════════════════════════
#  二、Transformer 核心模块
# ══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    固定正弦/余弦位置编码（Vaswani et al. 2017）。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵，shape: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)                           # (1, max_len, d_model)

        # 不作为参数，注册为 buffer（随模型保存/移动设备）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 词嵌入张量，shape (B, T, d_model)
        :return:  加上位置编码后的张量，shape (B, T, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力（仅解码器因果版本，使用下三角掩码屏蔽未来信息）。

    注意力公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q、K、V 和输出的线性投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (B, T, d_model)
        :return:  shape (B, T, d_model)
        """
        B, T, _ = x.shape

        # 1. 线性投影并拆分多头：(B, T, d_model) → (B, num_heads, T, d_k)
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))  # (B, H, T, d_k)
        K = split_heads(self.W_k(x))  # (B, H, T, d_k)
        V = split_heads(self.W_v(x))  # (B, H, T, d_k)

        # 2. 缩放点积注意力分数
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)

        # 3. 因果掩码：屏蔽位置 j > i（未来 token）
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 4. Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 加权聚合 V，合并多头
        context = torch.matmul(attn_weights, V)               # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous()        # (B, T, H, d_k)
        context = context.view(B, T, self.d_model)            # (B, T, d_model)

        # 6. 输出投影
        return self.W_o(context)


class FeedForward(nn.Module):
    """
    逐位置前馈网络（Position-wise FFN）。

    结构：Linear → ReLU → Dropout → Linear
    中间维度通常为 d_model 的 4 倍。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (B, T, d_model)
        :return:  shape (B, T, d_model)
        """
        return self.net(x)


class TransformerLayer(nn.Module):
    """
    单个 Transformer 解码器层（Pre-LayerNorm 风格，训练更稳定）。

    结构：
        x = x + Attention(LayerNorm(x))    # 残差 + 自注意力
        x = x + FFN(LayerNorm(x))          # 残差 + 前馈网络
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (B, T, d_model)
        :return:  shape (B, T, d_model)
        """
        # 子层 1：多头自注意力（Pre-LN + 残差）
        x = x + self.drop(self.attn(self.norm1(x)))
        # 子层 2：前馈网络（Pre-LN + 残差）
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════
#  三、完整的 Transformer 语言模型
# ══════════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """
    基于多层 TransformerLayer 的字符级语言模型。

    结构：
        Embedding → PositionalEncoding → N × TransformerLayer
        → LayerNorm → Linear(d_model, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        # 堆叠 N 个 Transformer 层
        self.layers  = nn.ModuleList(
            [TransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm    = nn.LayerNorm(d_model)          # 最终归一化
        self.fc      = nn.Linear(d_model, vocab_size) # 输出投影到词表

        # 权重绑定：让输出投影矩阵与词嵌入矩阵共享权重（减少参数、提升泛化）
        self.fc.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化线性层权重，零初始化偏置。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: token 索引，shape (B, T)
        :return:  每位置词表 logits，shape (B, T, vocab_size)
        """
        # 词嵌入 + 位置编码
        h = self.pos_enc(self.embed(x))   # (B, T, d_model)

        # 逐层前向传播
        for layer in self.layers:
            h = layer(h)

        # 最终归一化 + 线性输出
        logits = self.fc(self.norm(h))    # (B, T, vocab_size)
        return logits


# ══════════════════════════════════════════════════════════════════
#  四、训练 / 评估
# ══════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    运行一个 epoch，返回 (平均交叉熵损失, 困惑度 PPL)。

    :param train: True 则执行反向传播，False 则纯推理
    """
    model.train(train)
    total_loss   = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)                                          # (B, T, V)
        loss   = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl      = math.exp(avg_loss)
    return avg_loss, ppl


# ══════════════════════════════════════════════════════════════════
#  五、主函数
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transformer 字符级语言模型")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--d_model",    type=int,   default=128,  help="模型维度")
    parser.add_argument("--num_heads",  type=int,   default=4,    help="注意力头数")
    parser.add_argument("--num_layers", type=int,   default=2,    help="Transformer 层数")
    parser.add_argument("--d_ff",       type=int,   default=512,  help="FFN 中间维度（默认 4×d_model）")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer.pt")
    args = parser.parse_args()

    # 若未手动指定 d_ff，默认设为 4 × d_model
    if args.d_ff == 512 and args.d_model != 128:
        args.d_ff = args.d_model * 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 数据准备 ──
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split      = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── 模型 ──
    model = TransformerLM(
        vocab_size  = vocab_size,
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        d_ff        = args.d_ff,
        seq_len     = args.seq_len,
        dropout     = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(
        f"配置: d_model={args.d_model}, num_heads={args.num_heads}, "
        f"num_layers={args.num_layers}, d_ff={args.d_ff}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # 余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'LR':>8}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        marker = "  ✓" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "char2idx":    char2idx,
                    "idx2char":    idx2char,
                    "args":        vars(args),
                },
                args.save,
            )

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
            f"{va_loss:>10.4f}  {va_ppl:>10.2f}  {cur_lr:>8.2e}{marker}"
        )

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
