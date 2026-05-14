# week3 的作业

# 设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
# 如果不知道怎么设计，可以选择如下任务:
# 对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

# 选择任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

#步骤：
#自定义模型的类 
#  1，embedding层
#  2，循环层RUN/LSTM
#  3，定义线性层
#  4，附加其他：3,。。。。4.。。。。
#  5，定义loss层
#自定义模型的forward：跑数据和计算的位置
#定义辅助函数
#超参数定义
#模型在主函数中的定义
#定义优化器
#定义需要训练的数据
#开始训练
#保存模型
#测试模型


import numpy as np
import torch 
import torch.nn as nn
import random
#import matplotlib.pyplot as plt

class Torch_RUN(nn.Module):
    #类中初始化：包含模型及模型相关的输入量
    def __init__(self,VOCAB_SIZE, EMBED_DIM,input_size,hidden_size):
        super().__init__()#继承父类的工具，并执行初始化
        self.fc = nn.Linear(hidden_size, 5)  # 5分类：你在第0/1/2/3/4位
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)#embedding层
        self.rnn = nn.RNN(input_size, hidden_size, bias=True, batch_first=True)#RUN层
        self.loss = nn.CrossEntropyLoss() # loss函数,交叉熵损失
    
    #真正跑数据、计算的位置
    def forward(self,x,y=None):
        x = self.embedding(x)

        x, _ = self.rnn(x)

        x = x[:, -1, :]
        # x 的形状是：[批次，5个字，特征]
        # : → 所有样本都要
        # -1 → 取最后一个字的位置
        # : → 所有特征都要

        x = self.fc(x) 
        # 5分类：你在第0/1/2/3/4位

        if y is not None:
            return self.loss(x, y)
        return x

#从文本中读取训练集
def readtrain(name):
    with open(name,"r",encoding = "utf-8") as f:
        content = f.read()
    print(content)
    return content

#对从训练集的读取的内容，进行处理，使之符合模型的输入格式
#需要包含[pad]，[unk]
#可以把程序改成从文本中读取
def trainchange():
    vocab = {"[pad]":0, "你":1, "好":2, "中":3, "国":4, "欢":5, "迎":6, "[unk]":7}
    return vocab

#生成训练数据：
def build_data(vocab, num=1000):
    # vocab：词汇表（字 → 数字）
    # num=1000：默认生成 1000 条训练数据

    chars = [k for k in vocab if k not in ["[pad]", "[unk]"]]
    #等价于:
    #chars = []
    #for k in vocab:
    #    if k not in ["[pad]", "[unk]"]:
    #        chars.append(k)

    x_list = [] # 存输入文本（数字序列）
    y_list = [] # 存标签（你在第几位）

    for _ in range(num):

        sent = random.choices(chars, k=5)
        #Python 随机函数，随机选东西
        #sent的结构为一条 5 字文本句子，例如：sent = ["好", "中", "国", "欢", "迎"]

        pos = random.randint(0, 4)
        #随机生成一个 0 ~ 4 之间的整数（包括 0 和 4）

        sent[pos] = "你"

        x = [vocab[c] for c in sent]
        #vocab[c]: 查字典！把字变成数字 
        #据vocab的结构，可知： c:代表文字。vocab[c]:代表数字。

        x_list.append(x)
        y_list.append(pos)

    return torch.LongTensor(x_list), torch.LongTensor(y_list)
    #torch.LongTensor(x_list)把普通的数字列表 → 转换成 PyTorch 能识别的整数张量（长整型）
    #torch：用 PyTorch 的工具
    #Long：整数（int64），存单词编号用
    #Tensor：张量 = 模型能吃的数据格式

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model,vocab):
    model.eval()
    test_sample_num = 100
    x, y = build_data(vocab,test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    #函数入口，超参数初始化
    # 2，embedding超参数
    vocab = trainchange()
    print (vocab)
    VOCAB_SIZE = len(vocab)   # 字符数# 8
    EMBED_DIM  = 8    # 向量维度

    # 3，RUN超参数
    input_size =  EMBED_DIM  # 等于 词向量维度 EMBED_DIM 
    hidden_size = 16 #代表模型记忆能力
    learning_rate = 0.005 #学习率

    # 4，训练时的超参数
    epoch_num =  20 #训练轮数
    batch_size = 32 #每次喂给模型的数据条数
    train_sample = 100

    #模型在主函数中的定义
    model = Torch_RUN(VOCAB_SIZE, EMBED_DIM,input_size,hidden_size)

    #优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    #torch.optim.Adam()意思是torch中的optim（优化器）：使用Adam优化器
    #model.parameters()意思是模型里面所有的需要的参数
    #lr  学习率

    #数据预处理
    train_x,train_y = build_data(vocab)

    # 训练过程
    print("开始训练：。。。")
    log = []

    #开始每轮的训练
    for epoch in range(epoch_num):

        model.train()
        #模型切换为训练模式

        watch_loss = []
        #把每一步算出来的 loss 存起来，最后求平均值，看这一轮模型学得好不好

        for batch_index in range(train_sample // batch_size): #整除

        #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] #本轮从开始到结束的数据
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            #计算损失函数
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度

            #优化器部分
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零

            watch_loss.append(loss.item())
            #loss.item()：从 PyTorch 张量里，把里面的那个数字拿出来，变成普通的小数！

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,vocab)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    print("打印",log)
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return vocab, VOCAB_SIZE, EMBED_DIM, input_size, hidden_size


# 使用训练好的模型做预测
def predict(model_path, vocab, VOCAB_SIZE, EMBED_DIM,input_size,hidden_size):

    # 构造5条测试数据
    test_sents = [
        ["你", "好", "中", "国", "欢"],
        ["好", "你", "中", "国", "迎"],
        ["中", "国", "你", "好", "欢"],
        ["国", "欢", "迎", "你", "好"],
        ["欢", "迎", "中", "国", "你"]
    ]
    
    # 转成数字
    x = [[vocab[c] for c in sent] for sent in test_sents]
    
    model = Torch_RUN(VOCAB_SIZE, EMBED_DIM, input_size, hidden_size)

    model.load_state_dict(torch.load(model_path))
    # torch.load(model_path)
    # 从文件 "model.bin" 里
    # 把训练好的权重读取出来
    # 2. model.load_state_dict( ... )
    # 把读取到的权重
    # 装进当前的模型里
    # 让模型瞬间变成训练完成、可以预测的状态

    model.eval()
    #把模型切换成「评估 / 测试模式」

    with torch.no_grad():
        #关闭梯度计算

        result = model(torch.LongTensor(x))
        
    for sent, res in zip(test_sents, result):
        # sent：输入的 5 个字，比如 ["你", "好", "中", "国", "欢"]
        # res：模型对这句话的预测结果（5 个分数，对应位置 0~4）
        # zip(...)：一句输入 ↔ 一个结果，一一配对

        print(f"输入：{sent} → 预测“你”在第 {torch.argmax(res).item()} 位")
        # torch.argmax(res):就是模型认为 “你” 在哪里
        # res 是 5 个数字：代表0、1、2、3、4 五个位置的概率
        # argmax = 找出数字最大的那个位置
        # 2. .item()
        # 把 torch 张量 → 变成普通数字

vocab, VOCAB_SIZE, EMBED_DIM, input_size, hidden_size = main()
predict("model.bin",vocab, VOCAB_SIZE, EMBED_DIM, input_size, hidden_size)
