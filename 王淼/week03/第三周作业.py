'''
任务:训练一个模型，完成如下文本分类任务
对任意一个包含“你”的五个字的文本，你在第几位，就属于第几类
'''

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

SEED = 42
N_SAMPLES   = 8000
MAXLEN = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
random.seed(SEED)
torch.manual_seed(SEED)

#1.数据生成
TEMPLATES = [
    "清晨观山水",
    "清风拂林间",
    "闲坐看云舒",
    "晚灯照街巷",
    "举杯赏明月",
    "林间闻鸟语",
    "山野听风声",
    "窗前观花落",
    "庭前赏花开",
    "星河落远山",
    "晚风送凉意",
    "登高望长空",
    "静水映天光",
    "书香伴流年",
    "远山藏云雾"
]

CHAR_POOL = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞")

def make_words():
    #tmpl = random.choice(TEMPLATES)
    #随机生成一个模版和一个数字
    # ran_num = random.randint(0,4)
    # #用“你”替换此数字
    # rep_tmpl = tmpl[:ran_num] + '你' + tmpl[ran_num+1:]
    # return rep_tmpl,ran_num 
    pos = random.randint(0, MAXLEN - 1)
    chars = []
    for i in range(MAXLEN):
        if i == pos:
            chars.append("你")
        else:
            chars.append(random.choice(CHAR_POOL))
    return "".join(chars), pos

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_words())
    random.shuffle(data)
    return data

#2.词表构建与编码
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

#3.Dataset / DataLoader
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )

#4.模型定义
class KeywordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 把数字ID转化为64维的向量 每个字输入是一个编码，输出是64维向量
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True,batch_first=True)
        #用lstm模型进行双向学习，学习后每个向量都有记忆功能
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        #归一层是做数据归一化，让训练更稳定
        self.dropout = nn.Dropout(dropout)
        #用dropout层随机将某些向量置0，使之能学习到大多数特征
        self.linear = nn.Linear(hidden_dim * 2, 5)
        #用线性层做矩阵乘法映射维度

    def forward(self,x):
        emb = self.embedding(x)
        #输入是一个batch的文本，输出是一个batch的文本向量
        lstm_out, _ = self.lstm(emb)
        #输入是(batch,5,64),输出是(batch,5,128)
        pooled = lstm_out.max(dim=1)[0]
        #池化：把5个时刻的输出，压缩成1个句子向量,输出维度是(batch,128)
        pooled = self.dropout(self.bn(pooled))
        #归一化，随机失活某些向量
        out = self.linear(pooled)
        #线性映射到5维，输出是(batch,5)
        return out
    
#5.训练与评估
def evaluate(model,loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X,y in loader:
            out = model(X)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0

def train():
    print("正在构建数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    #划分训练集和验证集
    train_size = int(len(data) * TRAIN_RATIO)
    train_data = data[:train_size]
    val_data = data[train_size:]

    #将训练集和验证集封装成DataLoader
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    #实例化模型，定义损失函数和优化器
    model = KeywordLSTM(len(vocab)).to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #训练循环
    for epoch in range(1,EPOCHS + 1):
        model.train()
        total_loss = 0
        for X,y in train_loader:
            #前向传播得到输出
            out = model(X)
            #计算损失
            loss = criterion(out, y.long())
            #优化器优化前先把梯度置0
            optimizer.zero_grad()
            #反向传播计算梯度
            loss.backward()
            #更新模型参数
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    
    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
       '清你观山水',
       '清晨你山水 ',
       '手机号你啊 '
    ]
    with torch.no_grad():
        for sent in test_sents:
            encoded = torch.tensor(encode(sent, vocab), dtype=torch.long).unsqueeze(0)
            out = model(encoded)
            pred = out.argmax(dim=1).item()
            print(f"输入文本：'{sent}' 预测类别：{pred}")


if __name__ == '__main__':
    train()
        

 
