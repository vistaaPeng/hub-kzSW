"""
中文五字句“你”字位置分类 —— 简单 RNN 版本

任务：给定任意一个包含“你”字的 5 字文本，“你”出现在第几位（1~5）就属于第几类（标签 0~4）
模型：Embedding → RNN → MaxPooling → BN → Dropout → Linear(→5分类无Sigmoid)
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 5          # 固定 5 个字
EMBED_DIM   = 64
HIDDEN_DIM  = 64
NUM_CLASSES = 5          # 位置 0~4
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 15
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成：包含“你”的 5 字随机文本 ─────────────────
# 使用一个较大的常用汉字集合，确保多样性与合理性
CHARS_POOL = list(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之"
    "进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与"
    "关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常"
    "文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光"
    "门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目"
    "至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际"
    "商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王"
    "按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙你我他她它妈妈爸爸哥哥姐姐弟弟妹妹朋友同学老师医生警察工人"
    "农民学生司机老板顾客服务员小孩老人男人女人日月天地山水风云雨雪花草树木书笔纸张电脑手机电视桌椅床窗门路桥车船飞机春夏秋冬东南西北左右"
    "上下前后大小多少高矮胖瘦长短远近好坏新旧冷热酸甜苦辣红黄蓝绿黑白灰喜怒哀乐笑哭生死爱恨情仇说读写听看闻尝走跑跳飞舞游玩学习工作休息吃"
    "饭喝水睡觉醒梦思想记忘记回念愿祝福谢赞美夸奖批评责备争论问答解题考试成败胜负输赢"
)
# 保证“你”一定在池中
if '你' not in CHARS_POOL:
    CHARS_POOL.append('你')


def make_five_word_sequence():
    """生成一个包含“你”的 5 字文本，并返回文本和位置标签（0~4）"""
    pos = random.randint(0, 4)          # “你”的位置索引（0~4）
    seq_chars = []
    for i in range(5):
        if i == pos:
            seq_chars.append('你')
        else:
            # 从池中选一个非“你”的字，避免出现第二个“你”
            ch = random.choice(CHARS_POOL)
            while ch == '你':
                ch = random.choice(CHARS_POOL)
            seq_chars.append(ch)
    return ''.join(seq_chars), pos


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        sent, label = make_five_word_sequence()
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
    if len(ids) < maxlen:      # 理论上恒为 5，保留填充逻辑
        ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]          # 整数标签 0~4

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),   # CrossEntropyLoss 需要 long
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionRNN(nn.Module):
    """五分类 RNN：预测“你”字的位置"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)   # 输出 5 个 logits

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))   # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]             # (B, hidden_dim)  max pooling
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)             # (B, num_classes) 无 Sigmoid
        return logits


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)          # 预测类别
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

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
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # ─── 推理示例 ─────────────────────────────────────────────
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        "你我他她它",   # 第1位 → 类0
        "爱你在心头",   # 第2位 → 类1
        "今天你最美",   # 第3位 → 类2
        "花开花你落",   # 第4位 → 类3
        "山高水长你",   # 第5位 → 类4
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_class = logits.argmax(dim=1).item()
            # 显示习惯的位置（1~5）
            print(f"  预测位置：{pred_class+1} (类别{pred_class})  文本：{sent}")


if __name__ == '__main__':
    train()
