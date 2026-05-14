#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_toutiao_7class_rnn.py
今日头条 7 类中文文本多分类 —— Embedding + RNN（字符级）

数据：项目根目录 subset_7class_2000/ 下的 train.txt、val.txt、test.txt
每行格式：新闻ID_!_分类code_!_英文类别名_!_标题_!_关键词
标签：取第 3 个字段（如 news_culture），映射到 0～6 的整数类别。

模型思路（与 train_chinese_cls_rnn.py 一脉相承）：
  字符序列 → Embedding → 单向 RNN（逐时间步产出隐藏状态）
  → 在时间维上做 Max Pooling（聚合整条标题的信息，类似“任意位置出现强特征都可被捕获”）
  → BatchNorm + Dropout → 全连接 → 7 维 logits → CrossEntropyLoss

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
# 一、路径与超参数
# ═══════════════════════════════════════════════════════════════════════════════

# 当前脚本所在目录的上级 = 数据集项目根目录（内含 subset_7class_2000）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "subset_7class_2000")

SEED = 42
MAXLEN = 64          # 标题截断长度（字符数）；过长截断、过短右侧 PAD
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 7      # 7 个新闻类别
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 15
DROPOUT = 0.35

# 与 prepare_7class_split.py 中 CAT_SLUGS 顺序一致，便于复现实验与对照论文表格
SLUG_TO_ID: Dict[str, int] = {
    "news_culture": 0,       # 文化
    "news_entertainment": 1, # 娱乐
    "news_sports": 2,        # 体育
    "news_finance": 3,       # 财经
    "news_tech": 4,          # 科技
    "news_travel": 5,        # 旅游
    "news_game": 6,          # 游戏
}

ID_TO_SLUG = {v: k for k, v in SLUG_TO_ID.items()}

random.seed(SEED)
torch.manual_seed(SEED)


def _parse_line(line: str) -> Tuple[str, int]:
    """
    解析一行原始样本。
    返回：(标题文本, 类别 id)。
    若字段异常则抛出，便于发现损坏行。
    """
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
    """读取单个 txt，得到 [(标题, 标签id), ...]。"""
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
    """
    根据样本构建字符级词表。
    0：<PAD> 填充符；1：<UNK> 未知字符（预留，当前编码直接用 get 映射）。
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text, _ in samples:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], maxlen: int = MAXLEN) -> List[int]:
    """
    将字符串编码为定长 id 序列。
    超长截断前 maxlen；不足右侧用 0（PAD）补齐。
    """
    unk = vocab.get("<UNK>", 1)
    ids = [vocab.get(ch, unk) for ch in text]
    ids = ids[:maxlen]
    ids = ids + [0] * (maxlen - len(ids))
    return ids


# ═══════════════════════════════════════════════════════════════════════════════
# 二、PyTorch Dataset：把 (文本, 标签) 转为张量
# ═══════════════════════════════════════════════════════════════════════════════


class ToutiaoCharDataset(Dataset):
    """字符级编码数据集；标签为多分类整数 0..NUM_CLASSES-1。"""

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
# 三、模型：Embedding → RNN → MaxPool → BN → Dropout → Linear（7 类）
# ═══════════════════════════════════════════════════════════════════════════════


class ToutiaoRNNClassifier(nn.Module):
    """
    多分类头：最后一层输出维度 = NUM_CLASSES（不加 Sigmoid；损失函数用 CrossEntropy）。

    为何用 Max Pooling 而非只用最后一步 h_T：
      标题较短时，判别关键词可能出现在任意位置；Max 在时间维聚合，
      让每个时间步的隐藏向量都能“竞争”贡献最终表示（与参考脚本一致）。
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
        # padding_idx=0：与 encode 中的 PAD id 一致，填充位置不参与有效梯度（embedding 置零）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # batch_first=True：输入形状 (batch, seq_len, *)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        emb = self.embedding(x)           # (B, L, embed_dim)
        out, _ = self.rnn(emb)             # out: (B, L, hidden_dim)
        pooled = out.max(dim=1)[0]       # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)         # (B, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# 四、训练 / 评估
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """返回整体准确率。"""
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
    """各类召回式的「在该类样本上预测正确的比例」。"""
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

    # 词表仅用训练集构建，避免验证/测试信息泄漏到输入表征（严谨做法）
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

    model = ToutiaoRNNClassifier(vocab_size=vocab_size).to(device)
    # CrossEntropyLoss = LogSoftmax + NLLLoss；输入为 logits，标签为类别索引
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

    # ── 简单推理示例：手动构造几条标题观察预测类别 ──
    print("\n--- 推理示例（若干虚构/示意标题）---")
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
