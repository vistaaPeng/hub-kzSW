# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
'''
基于pytorch框架编写模型训练
规律:x是一个4维随机向量，作分类任务，向量里哪个数最大就属于第几类

'''

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 4) #线性层,输出为4维，代表4类
        self.loss = nn.CrossEntropyLoss()  # 更适合分类：输入logits + 整数标签

    #前向传播过程
    def forward(self,x,y=None):
        logits = self.linear(x)  # (N,4)
        if y is not None: #用到了标签y，在训练的过程中
            return self.loss(logits, y) # logits + 类别id 计算交叉熵
        else:
            return torch.softmax(logits, dim=1) # 输出概率分布
        
        
#生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(4)
    y = int(np.argmax(x)) # 返回最大值的索引(类别id: 0~3)
    return x, y

#随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long) #将数组数据转换为Torch能识别的形式
    

#测试代码
#测试每轮模型的准确率
def evaluate(model):
    model.eval() #开启测试阶段
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    correct,wrong = 0,0
    with torch.no_grad(): #测试阶段，不更新权重
        probs = model(x) # (N,4) 概率分布
        pred = torch.argmax(probs, dim=1) # (N,)
        for y_p,y_t in zip(pred,y): #与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    
    print(f"正确: {correct}, 错误: {wrong}, 准确率: {correct/(correct+wrong)}")

def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 4  # 输入向量维度
    learning_rate = 0.01  # 学习率

    #模型建立
    model = TorchModel(input_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #优化器，随机梯度下降
    for epoch in range(epoch_num):
        model.train() #开启训练阶段
        x,y = build_dataset(train_sample) #生成训练数据
        for i in range(0,train_sample,batch_size):  #小批量进行训练，每一个批次一次
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            loss = model(x_batch,y_batch) #前向传播，计算损失
            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播，计算梯度
            optimizer.step() #更新权重
        print(f"第{epoch}轮训练完成")
        evaluate(model) #评估模型准确率

    torch.save(model.state_dict(), "model.pth")
    predict("model.pth", [0.2, 0.1, 0.9, 0.3])

#使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 4
    model = TorchModel(input_size)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)  # 加载训练好的权重

    x = torch.tensor(input_vec, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (4,) -> (1,4)

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        probs = model(x)  # (N,4)
        pred_cls = torch.argmax(probs, dim=1)  # (N,)

    for vec, cls_idx, prob in zip(x.tolist(), pred_cls.tolist(), probs.tolist()):
        print(f"输入：{vec}，预测类别：{cls_idx}，概率分布：{prob}")

if __name__ == "__main__":
    main()
