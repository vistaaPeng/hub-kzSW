"""
train_you_position.py
"你"字位置多分类任务

任务：给定5字中文文本（必含"你"字），"你"字在第几位（1-5类）
模型：Embedding → RNN/LSTM → 取最后隐藏状态 → Linear → Softmax
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss

依赖：torch >= 2.0
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000
MAXLEN      = 5  # 固定5字
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 15
TRAIN_RATIO = 0.8
N_CLASSES   = 5  # 1-5位

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字池
CHARS = '的一是在不了有和人这中大上个国我以要他会做着地出过得到她就' \
        '看说都好去你我他她它我们你们他们它们' \
        '来自己这那此彼何谁孰' \
        '一二三四五六七八九十' \
        '天地山水火风土石' \
        '日月星辰云雨雷电' \
        '金木水火土东南西北'

def generate_sentence():
    """生成5字文本，确保包含一个"你"字"""
    positions = list(range(5))  # 0-4对应第1-5位
    you_pos = random.choice(positions)  # "你"字的位置
    
    chars_list = list(CHARS)
    sent = []
    for i in range(5):
        if i == you_pos:
            sent.append('你')
        else:
            # 确保其他位置不是"你"
            char = random.choice([c for c in chars_list if c != '你'])
            sent.append(char)
    
    return ''.join(sent), you_pos + 1  # 返回文本和类别(1-5)

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        sent, label = generate_sentence()
        data.append((sent, label))
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
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb - 1 for _, lb in data]  # 转换为0-4索引

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class RNNClassifier(nn.Module):
    """基础RNN分类器"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        _, h_n = self.rnn(self.embedding(x))  # h_n: (1, B, hidden_dim)
        out = self.fc(self.dropout(self.bn(h_n.squeeze(0))))  # (B, n_classes)
        return out

class LSTMClassifier(nn.Module):
    """LSTM分类器"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        _, (h_n, _) = self.lstm(self.embedding(x))  # h_n: (1, B, hidden_dim)
        out = self.fc(self.dropout(self.bn(h_n.squeeze(0))))  # (B, n_classes)
        return out

class BiLSTMClassifier(nn.Module):
    """双向LSTM分类器"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)  # 双向所以乘2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        _, (h_n, _) = self.lstm(self.embedding(x))  # h_n: (2, B, hidden_dim)
        # 拼接两个方向的隐藏状态
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, 2*hidden_dim)
        out = self.fc(self.dropout(self.bn(h_n)))  # (B, n_classes)
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

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    print(f"\n=== 训练 {model_name} ===")
    for epoch in range(1, epochs + 1):
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
        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    
    final_acc = evaluate(model, val_loader)
    print(f"最终验证准确率：{final_acc:.4f}")
    return final_acc

def inference(model, vocab, test_sents):
    """推理测试"""
    model.eval()
    print("\n--- 推理示例 ---")
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item() + 1  # 转换回1-5
            confidence = prob[0, pred-1].item()
            print(f"  '你'在第{pred}位({confidence:.2f})  {sent}")

# ─── 主函数 ──────────────────────────────────────────────────
def main():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")
    
    # 查看一些样本示例
    print("\n样本示例：")
    for i in range(5):
        sent, label = data[i]
        print(f"  '你'在第{label}位: {sent}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(PositionDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PositionDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 测试句子
    test_sents = [
        '你好世界啊',  # 你在第1位
        '我爱你中国',  # 你在第3位
        '今天你开心',  # 你在第3位
        '明天去你家',  # 你在第4位
        '大家喜欢你',  # 你在第5位
        '你是我的人',  # 你在第1位
        '我想你了呀',  # 你在第3位
    ]

    # 训练不同模型
    results = {}
    
    # RNN
    model_rnn = RNNClassifier(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_rnn.parameters(), lr=LR)
    results['RNN'] = train_model('RNN', model_rnn, train_loader, val_loader, criterion, optimizer)
    inference(model_rnn, vocab, test_sents)

    # LSTM
    model_lstm = LSTMClassifier(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=LR)
    results['LSTM'] = train_model('LSTM', model_lstm, train_loader, val_loader, criterion, optimizer)
    inference(model_lstm, vocab, test_sents)

    # BiLSTM
    model_bilstm = BiLSTMClassifier(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model_bilstm.parameters(), lr=LR)
    results['BiLSTM'] = train_model('BiLSTM', model_bilstm, train_loader, val_loader, criterion, optimizer)
    inference(model_bilstm, vocab, test_sents)

    # 对比结果
    print("\n=== 模型对比 ===")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.4f}")

if __name__ == '__main__':
    main()
