"""
week03.py
中文句子位置分类 —— 简单 RNN 版本

任务：对任意一个包含“你”字的 5 个字文本，“你”在第几位，就属于第几类（1-5）；
不包含“你”的文本属于第0类。
模型：Embedding → RNN → 取最后隐藏状态 → Linear
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 6000
MAXLEN      = 5
EMBED_DIM   = 32
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 6  # 0~5

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
ALL_CHARS = list('你我他她它们好学习工作生活天气喜欢漂亮简单快速稳定是的很真不也和有在上')
OTHER_CHARS = [ch for ch in ALL_CHARS if ch != '你']


def gen_sentence(position):
    """生成一个 5 字句子。
    position=0 时不包含“你”，否则“你”出现在第 position 位（1~5）。
    """
    chars = random.choices(OTHER_CHARS, k=MAXLEN)
    if position != 0:
        chars[position - 1] = '你'
    return ''.join(chars)


def build_dataset(n):
    data = []
    # 保证每个类别样本数量均衡
    count_per_pos = n // NUM_CLASSES
    for pos in range(NUM_CLASSES):
        for _ in range(count_per_pos):
            data.append((gen_sentence(pos), pos))
    # 处理余数
    for _ in range(n - len(data)):
        pos = random.randint(0, NUM_CLASSES - 1)
        data.append((gen_sentence(pos), pos))
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, vocab['<UNK>']) for ch in sent]
    ids = ids[:maxlen]
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(sent, vocab) for sent, _ in data]
        self.y = [label for _, label in data]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.rnn(emb)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train() -> None:
    print('生成数据集...')
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print("前5个样本：",data[:5])  # 打印前5个样本看看
    print("vocab：",vocab)
    print(f'  样本数：{len(data)}，词表大小：{len(vocab)}')

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f'  模型参数量：{sum(p.numel() for p in model.parameters()):,}')
    print('\n开始训练...')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}')

    print(f'\n最终验证准确率：{evaluate(model, val_loader):.4f}')

    print('\n--- 推理示例 ---')
    model.eval()
    examples = [
        '你很好看呀',
        '我喜欢你啊',
        '他不在你身',
        '你好漂亮呀',
        '今天很冷哦',
        '它是好的呀',
    ]
    with torch.no_grad():
        for sent in examples:
            ids = torch.tensor([encode(sent[:MAXLEN], vocab)], dtype=torch.long)
            logits = model(ids)
            pred = logits.argmax(dim=1).item()
            print(f'  {sent:<8} 预测类别: {pred}')


if __name__ == '__main__':
    train()
