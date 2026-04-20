# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个10维向量，按照最大值位置分类

"""
INPUT_SIZE = 10  # 输入向量维度

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, INPUT_SIZE)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测logits
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, 10)
        if y is not None:
            return self.loss(logits, y)  # 预测值和真实值计算损失
        else:
            return logits  # 输出预测logits

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = torch.rand(total_sample_num, INPUT_SIZE)
    Y = torch.argmax(X, dim=1)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        logits = model(x)  # 模型预测logits
        predicted = torch.argmax(logits, dim=1)
        correct = int((predicted == y).sum().item())
    wrong = test_sample_num - correct
    acc = correct / test_sample_num
    print("本次预测集样本数：%d，正确：%d，错误：%d，准确率：%f" % (test_sample_num, correct, wrong, acc))
    return acc


def main(model_path = None):
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 500000  # 每轮训练总共训练的样本总数
    input_size = INPUT_SIZE  # 输入向量维度
    learning_rate = 0.2  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD 
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        acc = 0.0
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad(set_to_none=True)  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        if epoch % 5 == 0:  # 每5轮测试一次模型结果
            acc = evaluate(model)  # 测试本轮模型结果
            log.append([acc, float(np.mean(watch_loss))])
        if acc > 0.9999:  # 如果准确率超过99.99%，提前结束训练
            print("准确率超过99.99%，提前结束训练")
            break
        if epoch % 10 == 0:  # 每10轮调整一次学习率
            learning_rate *= 0.5
            for param_group in optim.param_groups:
                param_group['lr'] = learning_rate
            print("调整学习率为：%f" % learning_rate)
            
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.savefig("training_curve.png")  # 保存图片到文件
    print("训练曲线已保存到 training_curve.png")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec, input_label):
    input_size = INPUT_SIZE
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model(torch.FloatTensor(input_vec))  # 模型预测
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)
    for vec, label, prob, true_label in zip(input_vec, predicted, probs, input_label):
        print("输入：%s\n 预测类别：%d, 最大概率：%f, 真实类别：%d" % (vec, int(label.item()), float(prob.max().item()), int(true_label.item())))

    # acc = evaluate(model)
    # print("预测准确率：%f" % acc)


if __name__ == "__main__":
    # main()
    # main("model.bin") # 加载之前训练好的模型继续训练
    test_x,test_y = build_dataset(10)
    predict("model.bin", test_x,test_y)
    
