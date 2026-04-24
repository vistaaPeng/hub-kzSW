# coding:utf8
"""
多分类任务训练
一个随机向量，哪一维数字最大就属于第几类（argmax 即为标签）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 5   # 向量长度 = 类别数，调小一些更容易学

# -------------------------------------------------------
# 1. 模型定义：输入 -> 隐层 ReLU -> 输出 logits
# -------------------------------------------------------
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)   # 隐层
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(32, num_classes)  # 输出 logits

    def forward(self, x, y=None):
        x = self.relu(self.layer1(x))
        y_pred = self.layer2(x)                   # shape: [N, num_classes]
        if y is not None:
            # 训练阶段：返回交叉熵 loss
            return nn.functional.cross_entropy(y_pred, y)
        # 预测阶段：返回 logits
        return y_pred


# -------------------------------------------------------
# 2. 数据生成：随机向量 + argmax 标签
# -------------------------------------------------------
def build_dataset(num_samples):
    X, Y = [], []
    for _ in range(num_samples):
        x = np.random.random(NUM_CLASSES)   # 均匀随机向量
        y = int(np.argmax(x))               # 最大值所在下标即类别
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# -------------------------------------------------------
# 3. 评估：在固定测试集上算准确率
# -------------------------------------------------------
def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        logits = model(test_x)                        # [N, num_classes]
        pred   = torch.argmax(logits, dim=1)          # 取概率最大的类别
        acc    = (pred == test_y).float().mean().item()
    print("测试准确率：%.4f" % acc)
    return acc


# -------------------------------------------------------
# 4. 训练主流程
# -------------------------------------------------------
def main():
    epoch_num     = 20
    batch_size    = 32
    train_samples = 5000
    lr            = 0.01

    model = TorchModel(NUM_CLASSES, NUM_CLASSES)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 生成训练集 & 固定测试集（训练期间不换，acc 曲线更稳）
    train_x, train_y = build_dataset(train_samples)
    test_x,  test_y  = build_dataset(200)

    log = []
    for epoch in range(epoch_num):
        model.train()
        losses = []
        for i in range(0, train_samples, batch_size):
            x = train_x[i : i + batch_size]
            y = train_y[i : i + batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        print("第%d轮  loss: %.4f" % (epoch + 1, avg_loss))
        acc = evaluate(model, test_x, test_y)
        log.append((acc, avg_loss))

    # 保存权重
    torch.save(model.state_dict(), "argmax_classify_model.bin")
    print("模型已保存")

    # 画 loss / acc 曲线
    epochs = range(1, len(log) + 1)
    plt.plot(epochs, [x[0] for x in log], label="acc")
    plt.plot(epochs, [x[1] for x in log], label="loss")
    plt.legend()
    plt.title("argmax train")
    plt.savefig("argmax_classify_train_curve.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
