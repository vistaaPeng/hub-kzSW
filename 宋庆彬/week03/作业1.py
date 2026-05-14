import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# -------------------- 1. 生成数据集 --------------------
char_pool = list(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
)

def generate_sample(num_classes=5):
    position = random.randint(0, num_classes - 1)  # 0-indexed
    chars = random.choices(char_pool, k=num_classes)
    chars[position] = '你'
    sentence = ''.join(chars)
    return sentence, position

def generate_dataset(num_samples=10000):
    sentences, labels = [], []
    for _ in range(num_samples):
        s, l = generate_sample()
        sentences.append(s)
        labels.append(l)
    return sentences, labels

# -------------------- 2. 构建词汇表 --------------------
all_chars = set(char_pool) | {'你'}
char_to_idx = {ch: i + 1 for i, ch in enumerate(sorted(all_chars))}
vocab_size = len(char_to_idx) + 1  # 0 留给 padding（但实际不用）

def sentence_to_tensor(sentence):
    return torch.tensor([char_to_idx[ch] for ch in sentence], dtype=torch.long)

# -------------------- 3. Dataset --------------------
class MyDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = sentence_to_tensor(self.sentences[idx])
        y = self.labels[idx]
        return x, y

# -------------------- 4. GRU 模型 --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)                # (batch, seq_len, emb_dim)
        _, h_n = self.gru(embedded)                 # h_n: (1, batch, hidden_dim)
        out = self.fc(h_n.squeeze(0))               # (batch, output_dim)
        return out

# -------------------- 5. 训练配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 5
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

sentences, labels = generate_dataset(5000)
dataset = MyDataset(sentences, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- 6. 训练与评估 --------------------
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

# -------------------- 7. 测试样例 --------------------
print("\n测试自定义输入：")
test_samples = [
    "你好天地人",
    "有你我他她",
    "他她它有你",
    "你好好好好",
    "好好的你我",
]
for s in test_samples:
    if len(s) != 5 or '你' not in s:
        print(f"'{s}' 不符合要求（必须5字且包含'你'）")
        continue
    model.eval()
    with torch.no_grad():
        x = sentence_to_tensor(s).unsqueeze(0).to(device)
        output = model(x)
        pred = output.argmax().item() + 1  # 转换为1-based位置
        print(f"输入: '{s}' -> 预测位置: {pred}")