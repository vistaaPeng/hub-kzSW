"""
任务：五字句中“你”的位置分类（5分类）
输入：长度为5的中文句子（包含“你”）
输出：类别 0~4（表示“你”的位置）

模型流程：
Embedding → RNN → MaxPooling → Linear → logits
损失函数：CrossEntropyLoss（内部自带 softmax）
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42        # 随机种子（保证实验可复现）
N_SAMPLES   = 5000      # 样本数量
MAXLEN      = 5         # 句子长度固定为5
EMBED_DIM   = 32        # 词向量维度
HIDDEN_DIM  = 64        # RNN隐藏层维度
LR          = 1e-3      # 学习率
BATCH_SIZE  = 64        # 批大小
EPOCHS      = 15        # 训练轮数
TRAIN_RATIO = 0.8       # 训练集比例
NUM_CLASS   = 5         # 分类数（对应位置1~5）

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────

# 构造基础字符集合（不包含“你”）
VOCAB_CHARS = list("今天天气很好我想去玩学习生活快乐吃饭睡觉写代码真不错")

def make_sample():
    """
    生成一个样本：
    1. 先随机生成4个字符
    2. 再随机插入“你”
    3. 返回句子 + 标签（位置）
    """
    others = [random.choice(VOCAB_CHARS) for _ in range(4)]

    pos = random.randint(0, 4)  # 插入位置（0~4）
    sent = others[:pos] + ['你'] + others[pos:]

    return ''.join(sent), pos  # label: 0~4


def build_dataset(n=N_SAMPLES):
    """
    构造数据集
    """
    data = []
    for _ in range(n):
        data.append(make_sample())
    return data


# ─── 2. 构建词表 ────────────────────────────────────────────
def build_vocab(data):
    """
    构建字符级词表
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}

    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)

    return vocab


def encode(sent, vocab):
    """
    将句子转为ID序列
    """
    return [vocab.get(ch, 1) for ch in sent]


# ─── 3. Dataset ────────────────────────────────────────────
class TextDataset(Dataset):
    """
    自定义数据集类
    """
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]  # 输入序列
        self.y = [lb for _, lb in data]               # 标签（位置）

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),   # (seq_len)
            torch.tensor(self.y[i], dtype=torch.long),   # 分类标签
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionClassifier(nn.Module):
    """
    多分类模型（RNN）

    架构说明：
    Embedding → RNN → MaxPooling → Linear → logits

    """

    def __init__(self, vocab_size):
        super().__init__()

        # 词嵌入层：将字符ID映射为向量
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        # RNN层：处理序列信息
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)

        # 全连接层：输出5分类
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASS)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """

        emb = self.embedding(x)      # (B, L, D)

        out, _ = self.rnn(emb)       # (B, L, H)

        # 对时间维做最大池化（提取最重要特征）
        pooled = out.max(dim=1)[0]   # (B, H)

        logits = self.fc(pooled)     # (B, 5)

        return logits   # 注意：不加softmax


# ─── 5. 评估 ────────────────────────────────────────────────
def evaluate(model, loader):
    """
    计算分类准确率
    """
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)

            # 取概率最大的类别
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total


# ─── 6. 训练 ────────────────────────────────────────────────
def train():
    print("生成数据...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    print(f"样本数: {len(data)}  词表大小: {len(vocab)}")

    # 划分训练集 / 验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 初始化模型
    model = PositionClassifier(len(vocab))

    # 多分类损失函数（自动包含 softmax）
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ─── 开始训练 ───
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            logits = model(X)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1:2d}  loss={total_loss:.4f}  val_acc={acc:.4f}")

    print("\n最终准确率:", evaluate(model, val_loader))

    # ─── 推理测试 ─────────────────────────────
    print("\n--- 推理示例 ---")

    test_sents = [
        "你今天天气好",
        "今你天天气好",
        "今天天你气好",
        "今天天气你好",
        "今天天气好你",
    ]

    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)])

            pred = torch.argmax(model(ids), dim=1).item()

            print(f"{sent} → 位置: {pred+1}")


# ─── 主函数 ────────────────────────────────────────────────
if __name__ == "__main__":
    train()
