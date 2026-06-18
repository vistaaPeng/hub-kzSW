# 【第二周作业】 尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# 1. 准备数据
def build_sample():
    x = np.random.choice(np.linspace(0, 1, 10000), size=5, replace=False)
    y = np.argmax(x)
    return x, y

def build_dataset(sample_num):
    x, y = [], []
    for _ in range(sample_num):
        x_, y_ = build_sample()
        x.append(x_)
        y.append(y_)
    return torch.FloatTensor(x), torch.LongTensor(y)

# 2. 定义模型
class MultiClassfication(nn.Module):
    def __init__(self, input_size):
        super(MultiClassfication, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        # self.fc2 = nn.Linear(3, 3)
        # self.fc3 = nn.Linear(3, 5)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        out = self.fc1(x)
        # out = self.fc2(out)
        # out = self.fc3(out)
        if y is None:
            return self.softmax(x)
        else:
            return self.loss(out, y)

# 评估函数
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        acc = torch.sum(y_pred == y) / len(y)
        return acc

# 3. 训练模型
def main():
    # 1. 准备数据
    train_sample_size = 1000
    valid_sample_size = 200
    x_train, y_train = build_dataset(train_sample_size)
    x_valid, y_valid = build_dataset(valid_sample_size)

    # 2. 定义模型
    epochs = 1000
    batch_size = 50
    lr = 0.01
    model = MultiClassfication(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 3. 训练模型
    log = []
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            optimizer.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, x_valid, y_valid)
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Acc: {acc}')
        log.append([acc, loss.item()])
    # 4. 可视化
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 5. 保存模型
    torch.save(model.state_dict(), "./week2_深度学习基本原理/my_practice/model.bin")

# 测试模型
def predict(model_path, input_vec):
    model = MultiClassfication(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)
    for vec, res in zip(input_vec, y_pred):
        print("输入：%s, 预测类别：%d" % (vec, res))  # 打印结果

if __name__ == '__main__':
    
    # main()

    # 6. 预测
    input_vec = build_dataset(5)[0]
    predict("./week2_深度学习基本原理/my_practice/model.bin", input_vec)

