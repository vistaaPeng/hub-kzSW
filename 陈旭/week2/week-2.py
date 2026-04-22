# 完成一个多酚类任务的训练：一个随机向量，哪一维数字大就属于第几类

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 定义数据集
def generate_data(n_samples=100):
    """生成随机5维向量，标签为最大值的索引"""
    X = np.random.randn(n_samples, 5)  # 随机5维向量
    print(f'5维向量数据形状x: {X.shape}')  # 输出数据的形状 (1000, 5)

    labels = np.argmax(X, axis=1)       # 最大值的索引作为标签 (0-4)
    print(f'标签数据形状labels: {labels.shape}')  # 输出标签的形状 (1000,)
    return X, labels

# 2. 定义神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(5, 5)  # 5维输入 -> 5类输出
        print(f'模型参数形状: {list(self.parameters())[0].shape}')  # 输出模型参数的形状 (5, 5)
    
    def forward(self, x): # 前向传播函数，输入x是一个(batch_size, 5)的张量
        return self.fc(x) # 输出未经过softmax的logits

# 3. 训练模型
def train():
    # 生成数据
    X_train, y_train = generate_data(1000) # 生成1000个训练样本
    X_test, y_test = generate_data(200) # 生成200个测试样本
    print(f'---------------trina----------------------------')
    print(f'x_train shape: {X_train.shape}, y_train shape: {y_train.shape}')  # 输出训练数据的形状
    print(f'x_test shape: {X_test.shape}, y_test shape: {y_test.shape}')  # 输出测试数据的形状
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train) # 将训练数据转换为PyTorch张量
    y_train = torch.LongTensor(y_train) # 将训练标签转换为PyTorch张量
    X_test = torch.FloatTensor(X_test) # 将测试数据转换为PyTorch张量
    y_test = torch.LongTensor(y_test) # 将测试标签转换为PyTorch张量

    print(f'x_train tensor: {X_train.shape}, y_train tensor: {y_train.shape}')  # 输出训练数据张量的形状
    print(f'x_test tensor: {X_test.shape}, y_test tensor: {y_test.shape}')  # 输出测试数据张量的形状

    
    # 初始化模型、损失函数和优化器
    model = SimpleClassifier()
    print(f'模型结构: {model}')  # 输出模型结构信息
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    print(f'损失函数: {criterion}')  # 输出损失函数信息
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器，学习率为0.1
    print(f'优化器: {optimizer}')  # 输出优化器信息
    
    # 训练循环
    print("开始训练...")
    for epoch in range(10): # 训练100轮
        # 前向传播
        outputs = model(X_train) # 模型输出 (1000, 5)，每行是一个样本的5类logits
        print(f'第{epoch+1}轮 - 模型输出形状: {outputs}')  # 输出模型输出的形状 (1000, 5)
        loss = criterion(outputs, y_train) # 计算损失，输入是模型输出和真实标签
        print(f'第{epoch+1}轮 - 损失值: {loss.item():.4f}')  # 输出当前轮次的损失值
        
        # 反向传播和优化
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 更新模型参数
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}") # 每20轮输出一次损失值
    
    # 4. 测试模型
    with torch.no_grad():
        outputs = model(X_test)
        print(f'测试集模型输出形状: {outputs.shape}')  # 输出测试集模型输出的形状 (200, 5)

        # predicted是模型预测的类别索引，torch.max函数返回每行最大值和对应的索引，这里我们只需要索引作为预测结果
        # _, 作用：丢弃第一个返回值（最大值），只保留第二个返回值（索引）
        _, predicted = torch.max(outputs, 1) # 获取每行最大值的索引作为预测类别
        print(f'测试集预测结果形状: {predicted}')  # 输出测试集预测结果的形状 (200,)

        # 计算准确率, y_test.size(0)是测试集样本数量,即200, predicted == y_test是一个布尔张量，表示每个预测是否正确，sum()计算正确预测的数量    
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'测试集真实标签形状: {y_test.shape}')  # 输出测试集真实标签的形状 (200,)
        print(f"测试集准确率: {accuracy:.2%}")
    
    # 5. 演示预测
    print("\n演示预测:")
    #  测试输入向量，包含5个随机值，模型将根据哪个值最大来预测类别 双层括号是为了保持输入的形状为(1, 5)，即一个样本的5维向量 1个batch，1个样本
    #  单样本	torch.FloatTensor([0.5, 2.1, 0.8])	    (3,)	1个样本，3个特征
    #  批量	    torch.FloatTensor([[0.5, 2.1, 0.8]])	(1, 3)	1个batch，1个样本
    #  批量	    torch.FloatTensor([[1,2,3], [4,5,6]])	(2, 3)	1个batch，2个样本
    test_vec = torch.FloatTensor([[0.5, 0.999, 8, 67, 4.77]])
    print(f"输入向量: {test_vec}")  # 输出输入向量


    # 模型输出是一个包含5个元素的张量，每个元素对应一个类别的logit值，值越大表示模型越倾向于预测该类别
    output = model(test_vec)
    print(f"模型输出: {output}")  # 输出模型的原始输出（logits）


    # predicted是模型预测的类别索引，torch.argmax函数返回每行最大值的索引，这里我们只需要索引作为预测结果
    # 获取模型预测的类别索引，item()将单元素张量转换为Python数值
    pred = torch.argmax(output, 1).item()
    print(f"预测类别: {pred} (最大值为第{pred}维)")

if __name__ == "__main__":
    train()

