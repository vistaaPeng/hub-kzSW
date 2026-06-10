'''
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 超参数
SEED        = 42
N_SAMPLES   = 1000
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)



# 1. 生成数据：生成包含“你”字的五个字的文本，“你”在第几位，就属于第几类
CHARS = '一二三四五六七八九十百千万亿'

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        # 生成包含“你”字的五个字的文本，“你”字的位置是随机的
        label = random.randint(0, 4)
        text = ''.join(random.choices(CHARS, k=4))
        text = text[:label] + '你' + text[label:]
        data.append((text, label))
    random.shuffle(data)
    return data

# 2. 构建词汇表
def build_vocab(data):
    vocab = {'<pad>': 0, '<unk>': 1}
    for text, _ in data:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

# 3. 编码文本
def encode(text, vocab, maxlen=MAXLEN):
    vec = [vocab.get(char, vocab['<unk>']) for char in text]
    vec = vec[:maxlen] + [vocab['<pad>']] * (maxlen - len(vec))
    return vec

# 4. 构建数据集
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode(text, vocab) for text, _ in data]
        self.y  = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.LongTensor(self.x[i]), 
            # torch.LongTensor(self.y[i]), # 参数为一个数字时，只会返回一个全0的tensor
            torch.tensor(self.y[i], dtype=torch.long),
        )
    
# 5. 定义模型
class MultiClassfication(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.maxpool = nn.MaxPool1d(MAXLEN)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x, y = None):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x)
        x = x.squeeze(2)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        if y is not None:
            loss = F.cross_entropy(x, y)
            return x, loss
        else:
            return x
        
# 6. 训练与评估
def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for x, y in loader:
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            correct += torch.sum(y_pred == y)
            total += len(y)
        acc = correct / total
    return acc

def train():

    print('生成训练数据集...')
    train_data = build_dataset()
    print('生成验证数据集...')
    valid_data = build_dataset()
    print('生成测试数据集...')
    test_data = build_dataset(5)

    print('构建词汇表...')
    vocab = build_vocab(train_data)

    print('构建数据集...')
    train_dataset = TextDataset(train_data, vocab)
    valid_dataset = TextDataset(valid_data, vocab)

    print('构建数据加载器...')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('构建模型...')
    model = MultiClassfication(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('开始训练...')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, valid_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    print(f"\n最终验证准确率：{evaluate(model, valid_loader):.4f}")

    print('开始推理预测...')
    model.eval()
    with torch.no_grad():
        for text, y in test_data:
            x = torch.tensor([encode(text, vocab)], dtype=torch.long)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            print(f"  预测结果：[{y_pred.item()}],  实际数据：[{text}, {y}]")
if __name__ == '__main__':
    train()