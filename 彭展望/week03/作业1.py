# coding:utf8
"""
作业1：对一个任意包含"你"字的五个字的文本，"你"在第几位，就属于第几类。

改造自 train_chinese_cls_rnn.py，使用字符级 RNN 对五字文本做 5 分类。
模型：Embedding → RNN → MaxPool → BN → Dropout → Linear(5) → CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000   # 总样本数，每类各 1000 条
MAXLEN      = 5      # 文本固定长度
EMBED_DIM   = 32
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5      # "你"在第 1~5 位 → 5 个类别

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 用于填充"你"以外位置的常用汉字池
FILL_CHARS = list('好世界啊哦我他她们的了是在有大小多少来去上下前后东南西北热闹')


def make_sample(ni_pos: int):
    """
    生成一条样本：5 字文本，"你"固定在 ni_pos（0 索引）位，其余随机汉字。

    :param ni_pos: "你"的位置，取值 0~4
    :return: (文本, 标签)，标签 = ni_pos（0~4）
    """
    chars = [random.choice(FILL_CHARS) for _ in range(5)]
    chars[ni_pos] = '你'
    return ''.join(chars), ni_pos


def build_dataset(n: int = N_SAMPLES):
    """每类均衡生成 n // NUM_CLASSES 条样本，共 5 类。"""
    data = []
    per_class = n // NUM_CLASSES
    for pos in range(NUM_CLASSES):
        for _ in range(per_class):
            data.append(make_sample(pos))
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """逐字符构建词表，保留 <PAD> 和 <UNK> 两个特殊 token。"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent: str, vocab: dict, maxlen: int = MAXLEN):
    """将文本转换为定长 id 序列，超长截断，不足补 PAD。"""
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class NiDataset(Dataset):
    """
    "你"字位置分类数据集。

    X: 编码后的字符 id 序列 (长度固定为 MAXLEN=5)
    y: 标签 0~4，对应"你"在第 1~5 位
    """

    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class NiPositionRNN(nn.Module):
    """
    "你"字位置分类器（5 分类）。

    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear(hidden, 5)
    训练时配合 CrossEntropyLoss，推理时取 argmax 得到类别 0~4。
    """

    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, seq_len=5)
        e, _   = self.rnn(self.embedding(x))   # (B, 5, hidden_dim)
        pooled = e.max(dim=1)[0]               # (B, hidden_dim)  序列维度 max pooling
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)                 # (B, 5)  原始 logits


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """在给定 DataLoader 上计算分类准确率。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred     = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def predict(model: nn.Module, vocab: dict, text: str) -> int:
    """
    推理：输入一个恰好含"你"字的 5 字文本，返回预测类别（1~5）。

    :param model: 已训练的 NiPositionRNN
    :param vocab: 训练时构建的词表
    :param text:  恰好 5 个字符且含"你"的文本
    :return:      预测类别编号，取值 1~5
    :raises ValueError: 文本格式不符合要求时抛出
    """
    if len(text) != 5:
        raise ValueError(f'文本长度必须为 5，当前长度为 {len(text)}')
    if '你' not in text:
        raise ValueError(f'文本中未找到"你"字：{text}')
    model.eval()
    with torch.no_grad():
        ids    = torch.tensor([encode(text, vocab)], dtype=torch.long)
        logits = model(ids)
        pred   = logits.argmax(dim=1).item()   # 0~4
    return pred + 1   # 转为 1~5


def train():
    print('生成数据集...')
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f'  样本数：{len(data)}，词表大小：{len(vocab)}')

    split        = int(len(data) * TRAIN_RATIO)
    train_loader = DataLoader(
        NiDataset(data[:split], vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(
        NiDataset(data[split:], vocab), batch_size=BATCH_SIZE)

    model     = NiPositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  模型参数量：{total_params:,}\n')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f'Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}')

    print(f'\n最终验证准确率：{evaluate(model, val_loader):.4f}')

    print('\n--- 推理示例 ---')
    test_cases = [
        '你好世界啊',   # "你"在第 1 位 → 第 1 类
        '好你世界啊',   # "你"在第 2 位 → 第 2 类
        '好世你界啊',   # "你"在第 3 位 → 第 3 类
        '好世界你啊',   # "你"在第 4 位 → 第 4 类
        '好世界啊你',   # "你"在第 5 位 → 第 5 类
    ]
    for text in test_cases:
        real_cls = text.index('你') + 1
        pred_cls = predict(model, vocab, text)
        mark     = 'O' if pred_cls == real_cls else 'X'
        print(f'  [{mark}] 「{text}」  真实第 {real_cls} 类  预测第 {pred_cls} 类')


if __name__ == '__main__':
    train()
