import torch
import torch.nn as nn
import torch.optim as optim
import random

#构造任务：5个字，判断“你”在第几位（1~5类）
vocab = ["我", "你", "他", "她", "它", "好", "坏", "爱", "吃", "睡"]
vocab_size = len(vocab)
word2idx = {w:i for i,w in enumerate(vocab)}

# 生成样本
def generate_sample():
    pos = random.randint(0,4)  # 0~4 对应标签 1~5
    sent = ["你" if i==pos else random.choice([w for w in vocab if w!="你"]) for i in range(5)]
    label = pos + 1  # 1~5 类
    return sent, label

# 生成一批数据
def build_data(n=2000):
    data = []
    for _ in range(n):
        sent, label = generate_sample()
        idx = [word2idx[w] for w in sent]
        data.append( (idx, label-1) )  # label转0~4
    return data

# 转dataloader
def get_loader(data, batch_size=32):
    sents = torch.tensor([x[0] for x in data], dtype=torch.long)
    labels = torch.tensor([x[1] for x in data], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(sents, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#模型：RNN 和 LSTM
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden=32, num_class=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_class)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = out[:,-1,:]  
        return self.fc(out)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden=32, num_class=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_class)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)

# 训练函数
def train(model, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (pred.argmax(1)==y).sum().item()
            total += len(x)
        
        print(f"Epoch {epoch+1} | loss={total_loss:.2f} | acc={correct/total:.2%}")

if __name__ == "__main__":
    data = build_data(2000)
    loader = get_loader(data)
    
    print("=============== RNN ===============")
    rnn = RNNModel(vocab_size)
    train(rnn, loader)
    
    print("\n=============== LSTM ===============")
    lstm = LSTMModel(vocab_size)
    train(lstm, loader)

    # ======================
    # 5. 测试一句
    # ======================
    def predict(model, text):
        idx = [word2idx[w] for w in text]
        x = torch.tensor([idx])
        y = model(x).argmax().item() + 1
        print(f"文本：{''.join(text)} → 预测‘你’在第 {y} 位")

    print("\n【预测 demo】")
    predict(lstm, ["我", "你", "他", "她", "它"])