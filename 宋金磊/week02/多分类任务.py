"""

基于pytorch框架编写模型训练
完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类

"""

import os
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt




class MultiClsModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClsModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        return x



# 随机生成一个5维向量，获取最大值对应的索引作为类别
def build_sample():
    x = np.random.random(5)      # 生成5维随机向量
    y = np.argmax(x)             # 取最大值对应的索引 0/1/2/3/4
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample = 1000
    test_x, test_y = build_dataset(test_sample)
    with torch.no_grad():
        outputs = model(test_x)
        pred = torch.argmax(outputs, dim=1)
        acc = torch.mean((pred == test_y).float())
    return acc


def main():
    # 配置参数
    epoch_num = 1000  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = MultiClsModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cross_loss = torch.nn.CrossEntropyLoss()
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            outputs = model(x)  # 得到5个分数
            loss = cross_loss(outputs, y)  # 计算loss

            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show(block=False)
    plt.pause(2)
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = MultiClsModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 输出 5 个分数

    # 取5个分数里最大的那个下标 → 就是预测类别
    pred_classes = torch.argmax(result, dim=1)

    # 打印结果
    for vec, res, cls in zip(input_vec, result, pred_classes):
        print("输入：%s, 预测类别：%d, 原始输出：%s" % (vec, cls.item(), res.numpy()))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.88, 0.15, 1.31, 0.03, 0.88],
        [0.94, 0.55, 0.95, 0.95, 3.84],
        [0.90, 0.67, 0.13, 0.34, 0.19],
        [0.99, 0.59, 0.92, 1.41, 0.13]
    ]

    predict("model.bin", test_vec)
