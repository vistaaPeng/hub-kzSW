"""
train_chinese_news6cls_rnn.py
中文新闻6分类 —— 简单 RNN 版本（基于你原代码修改）

任务：新闻6分类：体育、娱乐、房产、股票、教育、游戏
模型：Embedding → RNN → MaxPool → BN → Dropout → Linear → 6分类输出
损失：CrossEntropyLoss
优化：Adam (lr=1e-3)
无需 GPU，CPU 即可运行
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000
MAXLEN      = 48
EMBED_DIM   = 128
HIDDEN_DIM  = 128
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 30
TRAIN_RATIO = 0.8
NUM_CLASSES = 6  # 6分类！

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 新闻6分类 数据生成 ─────────────────────────────────
# 6个类别：体育(0)、娱乐(1)、房产(2)、股票(3)、教育(4)、游戏(5)

def build_news_dataset(n=N_SAMPLES):
    data = []
    samples_per_class = n // NUM_CLASSES

    # 体育
    for _ in range(samples_per_class):
        sent = random.choice([
            "篮球比赛今天大胜对手",
            "足球联赛球员表现出色",
            "乒乓球世锦赛中国队夺冠",
            "跑步运动员打破纪录",
            "排球决赛打得非常激烈",
            "教练安排战术非常合理",
        ])
        data.append((sent, 0))

    # 娱乐
    for _ in range(samples_per_class):
        sent = random.choice([
            "电影票房突破十亿",
            "歌手发布新专辑好评如潮",
            "演员参加综艺节目录制",
            "明星红毯造型惊艳全场",
            "演唱会门票瞬间售罄",
            "新剧开播热度持续上升",
        ])
        data.append((sent, 1))

    # 房产
    for _ in range(samples_per_class):
        sent = random.choice([
            "房价近期保持稳定趋势",
            "新楼盘开盘销售火爆",
            "小区环境改善居民满意",
            "户型设计合理使用率高",
            "开发商推出优惠活动",
            "二手房市场成交量上升",
        ])
        data.append((sent, 2))

    # 股票
    for _ in range(samples_per_class):
        sent = random.choice([
            "股市大盘指数上涨",
            "科技股表现强势领涨",
            "基金收益率创新高",
            "投资策略获得良好回报",
            "板块轮动速度加快",
            "散户投资者情绪乐观",
        ])
        data.append((sent, 3))

    # 教育
    for _ in range(samples_per_class):
        sent = random.choice([
            "高考成绩陆续公布",
            "学生开学进入新学期",
            "老师教学方法受到好评",
            "考研报名人数创新高",
            "课程改革提升教学质量",
            "校园文化活动丰富多彩",
        ])
        data.append((sent, 4))

    # 游戏
    for _ in range(samples_per_class):
        sent = random.choice([
            "手游新版本上线人气火爆",
            "电竞比赛战队获得冠军",
            "英雄皮肤限时折扣发售",
            "游戏副本难度适中",
            "主播直播人气突破百万",
            "玩家公会活动热闹非凡",
        ])
        data.append((sent, 5))

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
            torch.tensor(self.y[i], dtype=torch.long),  # 6分类必须是long
        )

# ─── 4. 模型定义（6分类版） ───────────────────────────────────
class NewsRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)  # 输出6个分数

    def forward(self, x):
        e = self.embedding(x)
        e, _ = self.rnn(e)
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)
        return out

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train():
    print("生成新闻6分类数据集...")
    data = build_news_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = NewsRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,}\n")

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
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率:{evaluate(model, val_loader)*100:.4f}")

    # 测试示例
    print("\n--- 新闻6分类 推理示例 ---")
    class_names = ["体育", "娱乐", "房产", "股票", "教育", "游戏"]
    test_sents = [
        "篮球比赛大胜对手",
        "电影票房突破五十亿",
        "新楼盘开盘销售火爆",
        "股市大盘全红",
        "高考成绩陆续公布",
        "电竞比赛获得第一名",
        "今天游戏真难打",
    ]
    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_idx = torch.argmax(logits, dim=1).item()
            print(f"预测类别：[{class_names[pred_idx]}]  句子：{sent}")

if __name__ == '__main__':
    train()