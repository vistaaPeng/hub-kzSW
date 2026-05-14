"""
train_ni_position_cls.py
"你"字位置多分类任务 —— RNN / LSTM / GRU 对比

GRU（门控循环单元）：RNN 的改进变体，通过"重置门"和"更新门"两个门控机制控制信息的保留与遗忘。
- 重置门 r：决定遗忘多少过去的记忆，r 接近 0 时忽略旧状态，相当于"重新开始"
- 更新门 z：决定保留多少旧状态、接入多少新状态，z 接近 1 时直接传递旧状态
- 相比 LSTM 少一个门且无独立细胞状态，参数更少、训练更快，多数任务效果接近 LSTM

任务：5 字中文文本中包含"你"，"你"在第几位 → 第几类（1~5，即 5 分类）
模型：Embedding → RNN/LSTM/GRU → MaxPool → Linear → CrossEntropyLoss
优化：Adam (lr=1e-3)   CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42           # 随机种子，保证每次运行结果可复现
N_SAMPLES   = 5000         # 训练集+验证集的总样本数
SEQ_LEN     = 5            # 输入文本的固定长度（5个字）
EMBED_DIM   = 32           # Embedding 向量维度（每个字的嵌入维度）
HIDDEN_DIM  = 64           # RNN/LSTM/GRU 隐藏层维度
LR          = 1e-3         # 学习率（Adam 优化器）
BATCH_SIZE  = 64           # 每个批次的样本数
EPOCHS      = 20           # 训练轮数（整个数据集反复训练的次数）
TRAIN_RATIO = 0.8          # 训练集占比（80%训练，20%验证）
NUM_CLASSES = 5            # 分类数："你"在第 1~5 位，共 5 个类别

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字（排除"你"），用于随机填充
FILLER_CHARS = list(
    '我他她它们这那个一上下不的大中小多少好'
    '来去说看想听吃做学走跑飞读写爱恨春夏秋'
    '冬天地水火山石木花草风雪月日星期早晚红蓝'
    '绿黄白黑猫狗鱼鸟牛马龙虎快乐难过高兴'
)


def make_sample():
    """生成一个 5 字文本，"你"在随机位置，返回 (文本, 标签)
    标签: 0~4 对应"你"在第 1~5 位（即索引位置）
    """
    pos = random.randint(0, 4)
    chars = [random.choice(FILLER_CHARS) for _ in range(5)]
    chars[pos] = '你'
    return ''.join(chars), pos


def build_dataset(n=N_SAMPLES):
    data = [make_sample() for _ in range(n)]
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


def encode(sent, vocab, maxlen=SEQ_LEN):
    # 查表编码，未知字用 <UNK>，填充到固定长度
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),   # 多分类用 long
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class BaseClassifier(nn.Module):
    """基类：Embedding → RNN变体 → MaxPool → FC"""
    def __init__(self, vocab_size, rnn_layer, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = rnn_layer
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e = self.embedding(x)                 # (B, L, embed_dim)
        e, _ = self.rnn(e)                    # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]              # (B, hidden_dim)
        return self.fc(self.dropout(pooled))   # (B, num_classes)  → CrossEntropy


def make_rnn_model(vocab_size):
    rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
    return BaseClassifier(vocab_size, rnn)


def make_lstm_model(vocab_size):
    rnn = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
    return BaseClassifier(vocab_size, rnn)


def make_gru_model(vocab_size):
    rnn = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
    return BaseClassifier(vocab_size, rnn)


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred   = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def train_one(model, train_loader, val_loader, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"模型: {name}  参数量: {total_params:,}")
    print(f"{'='*50}")

    t0 = time.time()
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
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    elapsed = time.time() - t0
    final_acc = evaluate(model, val_loader)
    print(f"  最终验证准确率: {final_acc:.4f}  耗时: {elapsed:.1f}s")
    return model, final_acc


def infer_demo(model, vocab, name):
    """推理演示"""
    print(f"\n--- {name} 推理示例 ---")
    model.eval()
    test_sents = [
        '你好我好',   # 你在位1
        '我你他她们',  # 你在位2
        '大中小你我',  # 你在位3
        '春夏秋冬你',  # 你在位4
        '好好学习你',  # 你在位5
        '你追我赶他',  # 你在位1
        '猫狗鱼鸟你',  # 你在位5
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids    = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred   = logits.argmax(dim=1).item() + 1  # 转为第几位
            prob   = torch.softmax(logits, dim=1)[0]
            conf   = prob[pred - 1].item()
            print(f'  "{sent}" -> 你在第{pred}位 (置信度 {conf:.2f})')


# ─── 6. 主流程 ──────────────────────────────────────────────
def main():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数: {len(data)}，词表大小: {len(vocab)}")
    print(f"  示例: {data[0][0]} → 类别 {data[0][1]+1}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data, vocab),
                              batch_size=BATCH_SIZE)

    # ── 三种模型对比 ──
    results = {}
    models  = [
        ('RNN',  make_rnn_model),
        ('LSTM', make_lstm_model),
        ('GRU',  make_gru_model),
    ]

    for name, build_fn in models:
        model = build_fn(len(vocab))
        model, acc = train_one(model, train_loader, val_loader, name)
        results[name] = acc
        infer_demo(model, vocab, name)

    # ── 汇总 ──
    print(f"\n{'='*50}")
    print("模型对比汇总")
    print(f"{'='*50}")
    for name, acc in results.items():
        print(f"  {name:6s}  val_acc = {acc:.4f}")


if __name__ == '__main__':
    main()
