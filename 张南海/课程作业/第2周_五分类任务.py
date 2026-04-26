''' 
 创建一个多分类任务，一个随机向量，哪一维数字最大，就是第几类 
 ''' 
 
import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
 
# 第一步，定义模型 
class TorchModel(nn.Module): 
    def __init__(self, input_size, num_classes): 
        '''
        初始化模型的参数，五维任务，输出也是五维（5个类别，每个类别的概率）
        :param input_size: 输入维度，5维
        :param num_classes: 类别数，5个
        '''
        super(TorchModel, self).__init__() 
        # 线性层：输入维度到类别数 
        self.linear = nn.Linear(input_size, num_classes) 
        # self.activation = torch.sigmoid # torch.sigmoid函数适用于二分类任务，多分类使用softmax函数
        # 交叉熵损失已经包含了softmax，所以不需要单独的激活函数 
        self.loss = nn.functional.cross_entropy 
 
    # 当输入真实标签，返回loss值；无真实标签，返回预测值 
    def forward(self, x, y=None): 
        x = self.linear(x)  # 输出原始分数(logits) 
        if y is not None: 
            # 预测值和真实值计算损失 
            return self.loss(x, y) 
        else: 
            return x  # 输出预测结果 
        
# 随机生成一个5维向量，最大的数在第几维，就返回对应下标，即第几类（从0开始） 
def build_sample(): 
    x = np.random.random(5) 
    # 直接使用numpy的argmax，返回0-based索引，等价于以下注释代码
    max_num_idx = np.argmax(x)
    return x, max_num_idx 
    '''
    max_num = 0
    max_num_idx = 0
    for idx,num in enumerate(x):
        if num > max_num:
            max_num = num
            max_num_idx = idx
    return x, max_num_idx
    '''
 
# 随机生成一批样本 
def build_dataset(total_sample_num): 
    X = [] 
    Y = [] 
    for i in range(total_sample_num): 
        x, y = build_sample() 
        X.append(x) 
        Y.append(y)  # 直接添加标签，不需要包装在列表中 
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 使用LongTensor存储标签 
 
# 测试代码 
# 用来测试每轮模型的准确率 
def evaluate(model): 
    # 标识模型在预测中 
    model.eval() 
    test_sample_num = 100 
    x, y = build_dataset(test_sample_num) 
    correct, wrong = 0, 0 
    # 告诉模型不需要计算梯度 
    with torch.no_grad(): 
        y_pred = model(x)  # 模型预测 
        # 取最大值的索引作为预测类别 
        y_pred_class = torch.argmax(y_pred, dim=1) 
        for y_p, y_t in zip(y_pred_class, y): 
            # 与真实标签进行对比 
            if y_p == y_t: 
                correct += 1 
            else: 
                wrong += 1 
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong))) 
    return correct / (correct + wrong) 
 
def main(): 
    # 配置参数 
    epoch_num = 20  # 训练轮数 
    batch_size = 20  # 每次训练样本个数 
    train_sample = 5000  # 增加训练样本数量 
    input_size = 5  # 输入向量维度 
    num_classes = 5  # 类别数 
    learning_rate = 0.01  # 学习率 
    # 建立模型 
    model = TorchModel(input_size, num_classes) 
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
            #取出一个batch数据作为输入 
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] 
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size] 
            loss = model(x, y)  # 计算loss 
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
 
# 使用训练好的模型做预测 
def predict(model_path, input_vec): 
    input_size = 5 
    num_classes = 5 
    model = TorchModel(input_size, num_classes) 
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重 
    print(model.state_dict()) 
    # 标识模型现在是测试模式 
    model.eval()
    with torch.no_grad():  # 不计算梯度 
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测 
        # 取最大值的索引作为预测类别 
        pred_class = torch.argmax(result, dim=1) 
    print(f"模型预测结果：{pred_class}") 
    for vec, res, cls in zip(input_vec, result, pred_class): 
        # 计算每个类别的概率 
        probabilities = torch.softmax(res, dim=0) 
        print("输入：%s, 预测类别：%d, 概率值：%s" % (vec, cls.item() + 1, probabilities.tolist()))  # 打印结果类别从1开始显示 
 
if __name__ == "__main__": 
    main() 
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843], 
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681], 
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392], 
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]] 
    # predict("model.bin", test_vec)
