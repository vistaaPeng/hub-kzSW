"""
五字文本分类 —— 判断“你”字的位置（第1~5位）
模型：Embedding → RNN / LSTM → 取最后时刻隐藏状态 → Linear(5) → CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000          # 总样本数（每个类别 1000 条）
MAXLEN = 5                # 输入固定为 5 个字
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 常用汉字池（500个常用字） ───────────────────────────────
CHAR_POOL = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞")

# ─── 1. 数据生成 ────────────────────────────────────────────
def generate_sample(label):
    """生成一个五字字符串，其中‘你’位于指定位置（label 0~4，对应第1~5位）"""
    other_chars = random.choices(CHAR_POOL, k=4)   # 随机选 4 个汉字
    other_chars.insert(label, '你')
    return ''.join(other_chars)

def build_dataset(n_samples=N_SAMPLES):
    """生成平衡数据集，每个类别样本数相同"""
    data = []
    samples_per_class = n_samples // 5
    for label in range(5):
        for _ in range(samples_per_class):
            sent = generate_sample(label)
            data.append((sent, label))   # 标签用 0~4 表示
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
    ids = ids[:maxlen] + [0] * (maxlen - len(ids))   # 固定长度 5，不需要补零，但保留
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]          # 0~4 的整数标签

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long)   # CrossEntropyLoss 要求 LongTensor
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionRNN(nn.Module):
    """RNN 版本"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)   # 5 个类别

    def forward(self, x):
        x = self.embedding(x)                # (B, L, E)
        out, _ = self.rnn(x)                 # (B, L, H)
        last = out[:, -1, :]                 # 取最后时刻隐藏状态 (B, H)
        logits = self.fc(last)               # (B, 5)
        return logits

class PositionLSTM(nn.Module):
    """LSTM 版本"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)                # (B, L, H)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

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

def train_model(model, train_loader, val_loader, model_name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        print(f"{model_name} | Epoch {epoch:2d}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"{model_name} 最终验证准确率：{final_acc:.4f}\n")
    return final_acc

# ─── 6. 主程序 ──────────────────────────────────────────────
def main():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本总数：{len(data)}，词表大小：{len(vocab)}")

    # 划分训练集/验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data, val_data = data[:split], data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 训练 RNN
    print("\n========== 训练 RNN ==========")
    rnn_model = PositionRNN(len(vocab))
    rnn_acc = train_model(rnn_model, train_loader, val_loader, "RNN")

    # 训练 LSTM
    print("========== 训练 LSTM ==========")
    lstm_model = PositionLSTM(len(vocab))
    lstm_acc = train_model(lstm_model, train_loader, val_loader, "LSTM")

    # 对比
    print(f"RNN 验证准确率：{rnn_acc:.4f}，LSTM 验证准确率：{lstm_acc:.4f}")

    # ─── 推理示例 ──────────────────────────────────────────────
    print("\n--- 推理示例 (用 LSTM 模型) ---")
    test_samples = [
        "你我他她它",
        "爱你没商量",
        "今天你累了",
        "大家你好吗",
        "想你再一遍",
    ]
    # 将测试句子处理成 5 字（不足补<PAD>，过长截断）
    def pad_or_trunc(s, length=5):
        if len(s) < length:
            s += '<PAD>' * (length - len(s))
        return s[:length]

    lstm_model.eval()
    with torch.no_grad():
        for raw in test_samples:
            s = pad_or_trunc(raw, 5)
            ids = torch.tensor([encode(s, vocab)], dtype=torch.long)
            logits = lstm_model(ids)
            pred = logits.argmax(dim=1).item() + 1   # 转为 1~5
            print(f"输入：'{raw}' -> 处理后：'{s}' => 预测位置：第 {pred} 位")

if __name__ == '__main__':
    main()
