import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务的训练:
    一个随机向量，哪一维数字最大就属于第几类
"""

class ClassifyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.activation = nn.Softmax(dim=1)  # 引入非线性函数,softmax函数用于多分类输出层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数(用于分类任务)

    def forward(self, x, y=None):
        """
        前向传播
        :param x: 输入数据
        :param y: 目标标签
        :return: 输出结果
        """
        x = self.linear(x)
        y_pred = self.activation(x)
        # 如果有目标标签返回损失函数值
        if y is not None:
            return self.loss(y_pred, y)
        # 如果没有目标标签返回预测结果
        return y_pred

def build_sample():
    # 生成一个6维随机向量
    x = np.random.random(6)
    # 找到向量中最大的索引，作为分类标签
    y = np.argmax(x)
    return x, y

def build_dataset(sample_num):
    """
    生成训练集
    :param sample_num: 训练样本总数
    :return: 训练集输入数据x和分类标签y
    """
    # np.empty 预先分配内存，后续循环中填充数据，比每次 append 更高效
    # x[i] 是6维向量
    x = np.empty((sample_num, 6))
    # y[i] 是0-5之间的整数
    y = np.empty((sample_num,))
    for i in range(sample_num):
        x[i], y[i] = build_sample()
    # 列表转换为 PyTorch 张量,PyTorch 模型只能处理张量,要求的输入必须是 torch.Tensor 类型
    return torch.FloatTensor(x), torch.LongTensor(y)

def evaluate(model):
    """
    评估模型在测试集上的准确率
    :param model: 模型
    :return: 准确率
    """
    '''
    model.eval() 将模型切换到 评估模式，主要有以下作用：
    1. 关闭dropout层等随机操作(Dropout 在训练时会随机丢弃神经元以防止过拟合)
    2. 训练时 BatchNorm 使用当前 batch 的均值和方差进行归一化，评估时使用全局均值和方差进行归一化
    3. 关闭梯度计算：eval() 本身不关闭梯度，但通常配合 torch.no_grad() 使用
    '''
    model.eval()  # 评估模式
    with torch.no_grad():  # 不计算梯度
        test_x, test_y = build_dataset(1000)  # 生成测试集
        y_pred = model(test_x)  # 前向传播，得到概率分布Softmax 输出），如 [[0.1, 0.4, 0.3, ...], ...]
        # 转成类别，如 [1, 0, 2, ...]
        y_pred = torch.argmax(y_pred, dim=1)
        # 计算正确预测个数
        correct = (y_pred == test_y).sum().item()
        # 计算错误率
        wrong = (y_pred != test_y).sum().item()
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)

def train():
    """
    训练模型
    """
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 6  # 输入向量维度
    output_size = 6  # 输出类别数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = ClassifyModel(input_size, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入 train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
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
    plt.show()
    return

def predict(model_path, input_vec):
    """
    使用训练好的模型做预测
    :param model_path: 模型路径
    :param input_vec: 输入向量
    :return: 预测结果
    """
    input_size = 6
    output_size = 6
    model = ClassifyModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        y_pred = model(torch.FloatTensor(input_vec))
        for i in range(len(y_pred)):
            print(f"输入向量：{input_vec[i]}，预测类别: {torch.argmax(y_pred[i], dim=0).item()}")

if __name__ == "__main__":
    #train()
    # 测试预测
    input_vec = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.3, 0.6, 0.1, 0.5, 0.2, 0.4],
    ])
    predict("model.bin", input_vec)
