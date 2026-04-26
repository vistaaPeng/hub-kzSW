import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
五分类任务，哪一维度数字最大 → 属于第几类（0,1,2,3,4）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出=类别数
        self.loss = nn.CrossEntropyLoss()  # 多分类用交叉熵损失

    # 有标签返回loss，无标签返回预测结果
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 生成样本：5维随机向量 → 最大值所在索引=标签
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 哪一维最大，标签就是几
    return x, y

# 生成一批样本
def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签必须是LongTensor

# 测试准确率（多分类）
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        pred_classes = torch.argmax(y_pred, dim=1)  # 取最大概率类别
        correct = (pred_classes == y).sum().item()
    acc = correct / test_sample_num
    print(f"正确预测：{correct}，正确率：{acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 25        # 训练轮数
    batch_size = 32       # 批次大小
    train_sample = 5000   # 总样本数
    input_size = 5        # 输入5维
    num_classes = 5       # 5分类
    learning_rate = 0.005

    # 初始化模型
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 构建训练集
    train_x, train_y = build_dataset(train_sample)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        total_loss = []

        for batch_idx in range(train_sample // batch_size):
            x = train_x[batch_idx*batch_size : (batch_idx+1)*batch_size]
            y = train_y[batch_idx*batch_size : (batch_idx+1)*batch_size]

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss.append(loss.item())

        avg_loss = np.mean(total_loss)
        print(f"========= 第{epoch+1}轮 =========")
        print(f"平均loss：{avg_loss:.4f}")
        acc = evaluate(model)
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "model_5class.bin")

    # 画图
    plt.figure(figsize=(10,4))
    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.title("5-class classification task")
    plt.legend()
    plt.show()

# 使用模型预测
def predict(model_path, input_vecs):
    model = TorchModel(5,5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        vec_tensor = torch.FloatTensor(input_vecs)
        pred = model(vec_tensor)
        pred_classes = torch.argmax(pred, dim=1)

    for idx, vec in enumerate(input_vecs):
        true_cls = np.argmax(vec)
        pred_cls = pred_classes[idx].item()
        print(f"输入向量：{vec}")
        print(f"真实类别：{true_cls}，预测类别：{pred_cls}\n")

if __name__ == "__main__":
    main()

    # 测试用的5维向量
    test_vecs = [
        [0.1, 0.2, 0.9, 0.3, 0.1],   # 最大是第2位 → 类别2
        [0.7, 0.1, 0.2, 0.1, 0.0],   # 最大是第0位 → 类别0
        [0.2, 0.8, 0.1, 0.1, 0.3],   # 最大是第1位 → 类别1
        [0.1, 0.1, 0.2, 0.9, 0.1],   # 最大是第3位 → 类别3
        [0.0, 0.1, 0.1, 0.2, 0.9],   # 最大是第4位 → 类别4
    ]
    predict("model_5class.bin", test_vecs)
