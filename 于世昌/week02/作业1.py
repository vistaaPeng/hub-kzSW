"""
作业：
尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 准备数据， 交叉熵 数据要张量， 标签要标量， 这块标签不能返回 向量 改成标量返回
def genVectorPair(dim):
    """生成入参dim维度的向量 以及 对应向量的标注(独热)
    """
    random_n_vector = np.random.rand(dim)
    # 找到向量中最大的值位置下标
    vector_max_index = np.argmax(random_n_vector)
    # 创建和目标向量同维度的全0向量
    # one_hot_vector = np.zeros_like(random_n_vector)
    # 将对应位置标成1
    # one_hot_vector[vector_max_index] = 1
    
    # 返回二维元组， 0号下标是 数据样本， 1号下标对应样本的标注
    # return random_n_vector, one_hot_vector
    # 交叉熵 数据要张量， 标签要标量， 这块标签不能返回 向量 改成标量返回
    return random_n_vector, vector_max_index


class ClassificationModel(torch.nn.Module):
    """分类模型"""
    def __init__(self, dim):
        """dim是分类维度
        """
        super(ClassificationModel, self).__init__()
        # 创建一个线性层， 入参是dim维的向量， 线性层入参 m维向量， 出参 m维向量
        self.linear = torch.nn.Linear(dim, dim)
        # 分类任务用softmax做激活函数, 获取一个能把 模型预测向量同比缩小 加和是1的softmax函数
        # 交叉熵本身自带softmax函数了，这里不用处理
        #self.active_func = torch.nn.Softmax(dim=1)
        # 损失函数 用交叉熵
        # 交叉熵 不应该放到模型内部
        # self.loss_func = torch.nn.CrossEntropyLoss()
    
    def forward(self, input):
        """正向传播
        """
        # 线性层计算
        result = self.linear(input)
        # 激活函数转非线性， 交叉熵自带了激活函数，这里不用处理
        # result = self.active_func(result)
        return result
    
# ======================
# 【模型验证 / 测试】
# ======================
def test_model(model, dim):
    
    data_tensor_list = []
    label_tensor_list = []
    for item in range(200):
        data, label = genVectorPair(dim)
        data_tensor_list.append(torch.tensor(data, dtype=torch.float32))
        label_tensor_list.append(torch.tensor(label, dtype=torch.long))

    data_tensor = torch.stack(data_tensor_list)
    label_tensor = torch.stack(label_tensor_list)
    tensor_dataset = TensorDataset(data_tensor, label_tensor)
    test_dataloader = DataLoader(tensor_dataset, batch_size=1)
    # 1. 开启【评估模式】：BN、Dropout 层固定，不训练
    model.eval()

    # 2. 关闭梯度计算（验证必须加！）
    with torch.no_grad():
        total_correct = 0  # 预测对的数量
        total_num = 0      # 总数量

        # 循环测试集
        for batch_x, batch_y in test_dataloader:
            # 前向预测
            outputs = model(batch_x)  

            """
                outputs = [
                    [-7.1, -4.5, -3.9, -0.8, 1.6],   # 第 1 个样本的预测
                    [-2.1, -5.5, -1.9, -0.3, 0.6],   # 第 2 个样本
                    [-3.2, -0.5, -2.9, -4.8, 3.1],   # 第 3 个样本
                    [-1.1, -2.5, -6.9, -3.8, 0.2],   # ...
                    [-5.1, -3.5, -0.9, -2.8, 4.6],
                    [-0.1, -6.5, -1.9, -2.8, 3.6],
                    [-2.2, -1.5, -3.9, -0.8, 2.6],
                    [-4.1, -2.5, -5.9, -1.8, 0.6],
                    [-3.1, -0.5, -2.9, -3.8, 1.2],
                    [-1.1, -4.5, -0.9, -2.8, 5.6],
                ]
            """
            # 3. 批量取预测类别（取最大概率的下标）[4,4,4,4,4,4,4,4,4,4]
            predict_labels = torch.argmax(outputs, dim=1)

            # 4. 统计正确个数
            # 
            total_correct += (predict_labels == batch_y).sum().item()
            total_num += len(batch_y)

        # 5. 计算准确率
        acc = total_correct / total_num
        print(f"测试集准确率：{acc:.4f}")
        return acc
        
def main():
    dim = 5
    # 准备数据集,200个数据和标注
    data_tensor_list = []
    label_tensor_list = []
    for item in range(200):
        data, label = genVectorPair(dim)
        data_tensor_list.append(torch.tensor(data, dtype=torch.float32))
        label_tensor_list.append(torch.tensor(label, dtype=torch.long))
    # 后面只能要张量，不要list<张量>，转一下
    data_tensor = torch.stack(data_tensor_list)
    label_tensor = torch.stack(label_tensor_list)
    
    model = ClassificationModel(dim)
    # 损失函数用交叉熵
    loss_func = torch.nn.CrossEntropyLoss()
    # 一共学习多少轮
    max_epoch = 1000
    # 优化器用Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 用批量数据集来处理
    data_set = TensorDataset(data_tensor, label_tensor)
    data_loader = DataLoader(data_set, batch_size=10, shuffle=True)
    for item in range(max_epoch):
        for data_batch, label_batch in data_loader:
            # 批次开始前清空优化器
            optimizer.zero_grad()
            # 批量拿到模型预测结果
            y_pred_batch = model(data_batch)
            # 用交叉熵计算loss
            loss = loss_func(y_pred_batch, label_batch)
            # 参数整体求偏导
            loss.backward()
            # 调参
            optimizer.step()
    
    test_model(model, dim)
    print(model(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)))
            
    
        

if __name__ == "__main__":
    main()