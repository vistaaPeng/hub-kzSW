import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务：
输入一个5维向量，哪一维数字最大，就属于哪一类（共5类：0,1,2,3,4）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 线性层：输入5维 → 输出5类（每类一个分数）
        self.linear = nn.Linear(input_size, class_num)
        # 多分类loss用交叉熵
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 计算loss
        else:
            return torch.argmax(x, dim=1)


# 生成一个样本：哪一维最大，标签就是几
def build_sample(input_size):
    x = np.random.random(input_size)
    y = np.argmax(x)  # 取最大值所在的下标（0,1,2,3,4）
    return x, y


# 生成一批样本
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试准确率
def evaluate(model, input_size):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        for yp, yt in zip(y_pred, y):
            if yp == yt:
                correct += 1
    acc = correct / test_sample_num
    print("正确预测：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 超参数
    epoch_num = 20        # 训练轮数
    batch_size = 20       # 批次大小
    train_sample = 5000   # 总样本数
    input_size = 5        # 输入5维向量
    class_num = 5         # 分成5类（0~4）
    learning_rate = 0.01

    # 创建模型
    model = TorchModel(input_size, class_num)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample, input_size)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("=========")
        print("第%d轮 平均loss：%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, input_size)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")

    # 画图
    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


# 使用模型预测
def predict(model_path, input_vec, input_size, class_num):
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec} → 预测类别：{res.item()}")


if __name__ == "__main__":
    main()

    # 测试代码
    test_vec = [
        [0.9,0.1,0.1,0.1,0.1],  # 0类
        [0.1,0.8,0.2,0.2,0.1],  # 1类
        [0.2,0.1,0.7,0.1,0.2],  # 2类
        [0.1,0.1,0.1,0.9,0.1],  # 3类
        [0.1,0.2,0.1,0.1,0.8]   # 4类
    ]
    predict("multi_class_model.bin", test_vec, 5, 5)
