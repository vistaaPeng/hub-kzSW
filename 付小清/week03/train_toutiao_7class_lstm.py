#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_toutiao_7class_lstm.py
今日头条 7 类中文文本多分类 —— Embedding + LSTM（字符级）

与 train_toutiao_7class_rnn.py 的数据管线、词表、类别映射完全一致，仅将序列骨干由 RNN 换成 LSTM，
便于对照二者在长短期依赖与梯度上的表现差异。

LSTM 相对普通 RNN：
  - 通过输入门、遗忘门、输出门与细胞状态，缓解长序列上的梯度消失；
  - 参数量更大，训练通常略慢，但在较长文本上往往更稳。

结构：Embedding → LSTM → Max Pooling → BN → Dropout → Linear → CrossEntropy

依赖：pip install torch
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# 一、路径与超参数（与 RNN 脚本对齐，便于公平对比）
# ═══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "subset_7class_2000")

SEED = 42
MAXLEN = 64
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 7
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 15
DROPOUT = 0.35

SLUG_TO_ID: Dict[str, int] = {
    "news_culture": 0,
    "news_entertainment": 1,
    "news_sports": 2,
    "news_finance": 3,
    "news_tech": 4,
    "news_travel": 5,
    "news_game": 6,
}

ID_TO_SLUG = {v: k for k, v in SLUG_TO_ID.items()}

random.seed(SEED)
torch.manual_seed(SEED)


def _parse_line(line: str) -> Tuple[str, int]:
    line = line.strip()
    if not line:
        raise ValueError("空行")
    parts = line.split("_!_")
    if len(parts) < 4:
        raise ValueError(f"字段不足: {line[:80]}...")
    slug = parts[2]
    title = parts[3]
    if slug not in SLUG_TO_ID:
        raise ValueError(f"未知类别: {slug}")
    return title, SLUG_TO_ID[slug]


def load_split_txt(path: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(_parse_line(ln))
            except ValueError:
                continue
    return out


def build_vocab(samples: List[Tuple[str, int]]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text, _ in samples:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], maxlen: int = MAXLEN) -> List[int]:
    unk = vocab.get("<UNK>", 1)
    ids = [vocab.get(ch, unk) for ch in text]
    ids = ids[:maxlen]
    ids = ids + [0] * (maxlen - len(ids))
    return ids


class ToutiaoCharDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], vocab: Dict[str, int]):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 二、模型：Embedding → LSTM → MaxPool → BN → Dropout → Linear（7 类）
# ═══════════════════════════════════════════════════════════════════════════════


class ToutiaoLSTMClassifier(nn.Module):
    """
    LSTM 输出序列 out[:, t, :] 与 RNN 相同均为 (B, L, H)，后续池化与分类头与 RNN 版一致。

    nn.LSTM 返回值：
      output：每一时刻最顶层隐藏状态 h_t；
      (h_n, c_n)：最后时刻的细胞状态与隐藏状态（本实验采用全局 Max Pool，不直接使用 h_n）。
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        # out: 所有时间步隐藏状态；_:(hn, cn) 此处不需要，池化已聚合全局信息
        out, _ = self.lstm(emb)
        pooled = out.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> List[float]:
    model.eval()
    correct = [0] * num_classes
    counts = [0] * num_classes
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(dim=1)
        for p, t in zip(pred.tolist(), y.tolist()):
            counts[t] += 1
            if p == t:
                correct[t] += 1
    return [correct[i] / counts[i] if counts[i] else 0.0 for i in range(num_classes)]


def train_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = os.path.join(_DATA_DIR, "train.txt")
    val_path = os.path.join(_DATA_DIR, "val.txt")
    test_path = os.path.join(_DATA_DIR, "test.txt")

    for p in (train_path, val_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"未找到数据文件: {p}\n请先运行 prepare_7class_split.py 生成 subset_7class_2000。"
            )

    train_data = load_split_txt(train_path)
    val_data = load_split_txt(val_path)
    test_data = load_split_txt(test_path)

    vocab = build_vocab(train_data)
    vocab_size = len(vocab)

    train_loader = DataLoader(
        ToutiaoCharDataset(train_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ToutiaoCharDataset(val_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        ToutiaoCharDataset(test_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ToutiaoLSTMClassifier(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {device}")
    print(f"训练/验证/测试样本数: {len(train_data)} / {len(val_data)} / {len(test_data)}")
    print(f"词表大小: {vocab_size}，模型参数量: {n_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"\n测试集准确率: {test_acc:.4f}")
    pa = per_class_accuracy(model, test_loader, device, NUM_CLASSES)
    print("各类在测试集上的准确率:")
    for i, acc in enumerate(pa):
        print(f"  [{i}] {ID_TO_SLUG[i]:22s}  {acc:.4f}")

    print("\n--- 推理示例 ---")
    demo_texts = [
        "故宫春季展览本周开幕",
        "中超联赛第三轮综述",
        "新款智能手机芯片参数曝光",
        "自驾游川藏线攻略分享",
        "Steam 平台独立游戏推荐",
    ]
    model.eval()
    with torch.no_grad():
        for sent in demo_texts:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long, device=device)
            cid = model(ids).argmax(dim=1).item()
            print(f"  预测={ID_TO_SLUG[cid]}  |  {sent}")


if __name__ == "__main__":
    train_main()
