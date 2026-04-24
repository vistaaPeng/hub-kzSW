
# coding:utf8
# 解决 OpenMP 库冲突问题（Windows系统下pytorch+matplotlib可能出现的多线程库冲突）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入核心库
import torch  # pytorch核心库，提供张量计算和神经网络模块
import torch.nn as nn  # 神经网络层、损失函数等核心模块
import numpy as np  # 数值计算库，用于生成随机数据、计算均值等
import random  # 随机数生成（本代码中未直接使用，保留备用）
import json  # 数据序列化（本代码中未直接使用，保留备用）
import matplotlib.pyplot as plt  # 绘图库，用于可视化训练过程

"""
基于pytorch框架改造为多分类任务
任务规律：x是一个5维向量，哪一维的数字最大，该样本就属于对应的类别（0-4类，对应第1-5维）
例如：向量[0.2,0.8,0.1,0.3,0.5]中第2维（索引1）最大，属于类别1
"""

# 定义神经网络模型类，继承nn.Module（pytorch所有模型的基类）
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        模型初始化函数
        :param input_size: 输入向量维度（这里是5）
        :param num_classes: 分类类别数（这里是5）
        """
        super(TorchModel, self).__init__()  # 调用父类初始化函数，必须执行
        # 线性层：将输入维度(input_size)映射到类别数(num_classes)
        # 多分类任务中，输出维度等于类别数，代表每个类别的预测得分
        self.linear = nn.Linear(input_size, num_classes)  
        # 激活函数：Softmax将线性层输出的得分转换为概率（总和为1）
        # dim=1表示按行（每个样本）做softmax
        self.activation = nn.Softmax(dim=1)  
        # 损失函数：交叉熵损失（多分类任务的标准损失函数）
        # nn.CrossEntropyLoss会自动对输入做softmax，因此前向传播中可选择是否提前做softmax
        self.loss = nn.CrossEntropyLoss()  

    # 前向传播函数：定义模型的计算流程
    # x: 输入张量 (batch_size, input_size)
    # y: 真实标签（可选），有则返回loss，无则返回预测概率
    def forward(self, x, y=None):
        # 线性层计算：输入->类别数维度 (batch_size, input_size) -> (batch_size, num_classes)
        x = self.linear(x)  
        # 激活函数：将得分转换为概率分布 (batch_size, num_classes)
        y_pred = self.activation(x)  
        if y is not None:
            # 计算损失：注意CrossEntropyLoss要求y是类别索引（整数），y_pred是概率分布
            return self.loss(y_pred, y)  
        else:
            # 无真实标签时，返回每个类别的预测概率
            return y_pred  


# 生成单个样本：核心逻辑是"哪一维最大就属于哪一类"
def build_sample(input_size):
    """
    生成单个训练/测试样本
    :param input_size: 向量维度（固定为5）
    :return: x(5维随机向量), y(类别索引，0-4)
    """
    # 生成5维随机向量，值范围[0,1)
    x = np.random.random(input_size)  
    # 找到最大值所在的索引（即类别），例如x=[0.1,0.9,0.2]，argmax返回1
    y = np.argmax(x)  
    return x, y


# 生成批量样本数据集
def build_dataset(total_sample_num, input_size):
    """
    生成批量样本
    :param total_sample_num: 样本总数
    :param input_size: 向量维度
    :return: X(张量, [total_sample_num, input_size]), Y(张量, [total_sample_num])
    """
    X = []  # 存储输入向量
    Y = []  # 存储类别标签
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    # 转换为pytorch张量：X为FloatTensor（神经网络输入要求浮点型），Y为LongTensor（交叉熵损失要求整型）
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 模型评估函数：测试模型在新样本上的准确率
def evaluate(model, input_size):
    """
    评估模型性能
    :param model: 训练好的模型实例
    :param input_size: 输入向量维度
    :return: 准确率
    """
    model.eval()  # 切换到评估模式（禁用dropout、batchnorm等训练特有的层）
    test_sample_num = 1000  # 评估用的样本数
    x, y = build_dataset(test_sample_num, input_size)  # 生成测试集
    # 统计测试集中每个类别的样本数（验证样本分布均匀性）
    class_count = [0]*5
    for label in y:
        class_count[label] += 1
    print("本次预测集中各类别样本数：", class_count)
    
    correct, wrong = 0, 0
    with torch.no_grad():  # 评估阶段禁用梯度计算（节省内存、加快速度）
        y_pred = model(x)  # 模型预测，输出每个类别的概率 (1000,5)
        # 遍历每个样本的预测结果和真实标签
        for y_p, y_t in zip(y_pred, y):
            # torch.argmax(y_p)：找到概率最大的类别索引；y_t：真实类别索引
            if torch.argmax(y_p) == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1  # 预测错误
    acc = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 错误预测个数：{wrong}, 正确率：{acc:.4f}")
    return acc


def main():
    # ===================== 超参数配置 =====================
    epoch_num = 30  # 训练轮数：整个训练集重复训练的次数
    batch_size = 32  # 批次大小：每次送入模型的样本数（太小不稳定，太大内存不足）
    train_sample = 10000  # 每轮训练的总样本数
    input_size = 5  # 输入向量维度（固定为5）
    num_classes = 5  # 分类类别数（固定为5）
    learning_rate = 0.005  # 学习率：控制权重更新的步长（太大震荡不收敛，太小训练慢）
    
    # ===================== 模型初始化 =====================
    model = TorchModel(input_size, num_classes)  # 实例化模型
    # 优化器：Adam（自适应学习率优化器，比SGD更稳定）
    # 传入模型参数和学习率，负责更新模型权重
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 记录每轮的准确率和损失，用于绘图
    
    # ===================== 训练集准备 =====================
    train_x, train_y = build_dataset(train_sample, input_size)  # 生成训练集
    
    # ===================== 训练过程 =====================
    for epoch in range(epoch_num):
        model.train()  # 切换到训练模式（启用dropout、batchnorm等）
        watch_loss = []  # 记录本轮每个批次的损失值
        # 按批次遍历训练集：train_sample//batch_size 表示总批次数
        for batch_index in range(train_sample // batch_size): 
            # 截取当前批次的样本：从batch_index*batch_size到(batch_index+1)*batch_size
            # 例如batch_size=32，第0批取0-31，第1批取32-63，以此类推
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size
            x = train_x[start:end]
            y = train_y[start:end]
            
            # 核心训练步骤（四步走）：
            loss = model(x, y)  # 1. 前向传播：计算预测值和损失
            loss.backward()     # 2. 反向传播：计算每个参数的梯度
            optim.step()        # 3. 优化器更新：根据梯度更新权重
            optim.zero_grad()   # 4. 梯度清零：避免下一批次梯度累积
            
            watch_loss.append(loss.item())  # 记录当前批次的损失值（item()取出张量的数值）
        
        # 每轮训练结束后，打印平均损失并评估模型
        avg_loss = np.mean(watch_loss)
        print(f"\n========= 第{epoch+1}轮训练结束 =========")
        print(f"本轮平均loss：{avg_loss:.4f}")
        acc = evaluate(model, input_size)  # 评估本轮模型准确率
        log.append([acc, avg_loss])  # 记录准确率和损失
    
    # ===================== 模型保存与可视化 =====================
    torch.save(model.state_dict(), "multi_class_model.bin")  # 保存模型权重（仅保存参数，不保存模型结构）
    print("\n训练日志（每轮准确率、损失）：", log)
    
    # 绘制训练曲线：准确率和损失随轮数的变化
    plt.figure(figsize=(12, 4))
    # 子图1：准确率曲线
    plt.subplot(1,2,1)
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    
    # 子图2：损失曲线
    plt.subplot(1,2,2)
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()  # 调整子图间距
    plt.show()


# 预测函数：使用训练好的模型对新数据做预测
def predict(model_path, input_vec):
    """
    加载模型并预测
    :param model_path: 模型权重文件路径
    :param input_vec: 待预测的向量列表（二维列表）
    """
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)  # 重新实例化模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print("模型权重参数：", model.state_dict())  # 打印权重（可选，用于调试）
    
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        # 将输入向量转换为FloatTensor，送入模型预测
        result = model(torch.FloatTensor(input_vec))  
    # 遍历每个输入向量和预测结果，打印详细信息
    for vec, res in zip(input_vec, result):
        # torch.argmax(res)：概率最大的类别；res.max().item()：最大概率值
        pred_class = torch.argmax(res).item()
        max_prob = res.max().item()
        print(f"输入向量：{vec}, 预测类别：{pred_class}, 该类别概率：{max_prob:.4f}")  


# 主函数入口：程序从这里开始执行
if __name__ == "__main__":
    main()  # 执行训练流程
    # 测试预测功能（训练完成后取消注释）
    test_vec = [
        [0.1, 0.8, 0.05, 0.03, 0.02],   # 第2维最大，类别1
        [0.05, 0.1, 0.9, 0.03, 0.02],   # 第3维最大，类别2
        [0.9, 0.05, 0.03, 0.01, 0.01],   # 第1维最大，类别0
        [0.02, 0.03, 0.05, 0.9, 0.0],    # 第4维最大，类别3
        [0.01, 0.02, 0.03, 0.04, 0.9]    # 第5维最大，类别4
    ]
    predict("multi_class_model.bin", test_vec)
