import torch
import torch.nn as nn
import torch.optim as optim

# 1. 生成合成数据
num_samples = 10000       # 总样本数
input_dim = 5             
num_classes = input_dim   

# 生成 0 到 1 之间的随机浮点数作为特征 X
X = torch.rand(num_samples, input_dim)

# 标签 Y：找出每一行中最大值的索引 (0, 1, 2, 3, 或 4)
y = torch.argmax(X, dim=1)

# 划分训练集 (80%) 和测试集 (20%)
train_X, test_X = X[:8000], X[8000:]
train_y, test_y = y[:8000], y[8000:]

# 2. 定义神经网络模型
class ArgMaxNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ArgMaxNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),  # 输入层到隐藏层
            nn.ReLU(),                 # 激活函数，引入非线性
            nn.Linear(32, 16),         # 隐藏层到隐藏层
            nn.ReLU(),
            nn.Linear(16, num_classes) # 隐藏层到输出层 (输出每个类别的 logits)
        )

    def forward(self, x):
        return self.network(x)

# 实例化模型
model = ArgMaxNet(input_dim, num_classes)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
epochs = 200

print("开始训练...")
for epoch in range(epochs):
    # 前向传播：将数据输入模型得到预测值
    outputs = model(train_X)
    
    # 计算损失：预测值和真实标签之间的差距
    loss = criterion(outputs, train_y)
    
    # 反向传播和优化
    optimizer.zero_grad() # 清空过往梯度
    loss.backward()       # 反向传播，计算当前梯度
    optimizer.step()      # 根据梯度更新网络参数
    
    # 每 20 轮打印一次结果
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 测试模型评估
model.eval() 
with torch.no_grad(): 
    test_outputs = model(test_X)
    
    # 取出每个样本预测概率最大的类别索引
    _, predicted = torch.max(test_outputs.data, 1)
    
    # 计算准确率
    correct = (predicted == test_y).sum().item()
    total = test_y.size(0)
    accuracy = correct / total * 100
    
    print(f'\n测试完成！模型在测试集上的准确率: {accuracy:.2f}%')

# 6. 预测示例
sample_input = torch.tensor([[0.1, 0.8, 0.3, 0.2, 0.5]]) 
model.eval()
with torch.no_grad():
    output = model(sample_input)
    _, pred_class = torch.max(output, 1)
    print(f"\n输入向量: {sample_input.tolist()[0]}")
    print(f"真实最大维度: 1")
    print(f"模型预测维度: {pred_class.item()}")
