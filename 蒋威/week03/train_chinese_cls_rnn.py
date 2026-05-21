# 中文文本多分类任务  支持 RNN/LSTM/GRU
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 【超参数设置】
# ============================================================
SEED = 42  # 随机种子，保证实验可复现
N_SAMPLES = 6000  # 总样本数（每类2000，共3类）
# MAXLEN      = 40         # 最大序列长度，超过截断，不足填充  64
MAXLEN = 40
# EMBED_DIM   = 128        # 词嵌入维度，将每个字映射到128维向量  512
EMBED_DIM = 512
# HIDDEN_DIM  = 128        # RNN隐藏层维度  512
HIDDEN_DIM = 256
N_LAYERS = 2  # RNN层数，多层可提取更复杂特征 2~3比较合适小数据集
DROPOUT = 0.3  # Dropout比率，防止过拟合  0.2-0.5
LR = 1e-3  # 学习率  1e-4 ~ 1e-2（Adam优化器常使用1e-3）
BATCH_SIZE = (
    64  # 批次大小，每次训练64个样本 32-256（受GPU显存限制） 越大越快但是显存也更多
)
EPOCHS = 30  # 训练轮数  根据数据集大致设置
# TRAIN_RATIO = 0.8        # 训练集占比  训练集70%-80%，验证集10%-15%，测试集10%-15%
TRAIN_RATIO = 0.7
"""
模型类型：'RNN' | 'LSTM' | 'GRU'
短句分类（avg_len=10）  → GRU 足够
长文档情感（avg_len=500）→ LSTM 更稳
代码生成（avg_len=1000） → Transformer
"""
MODEL_TYPE = "LSTM"

# 设置随机种子，确保每次运行结果一致
random.seed(SEED)
torch.manual_seed(SEED)

# 数据生成
# ============================================================

# 情感关键词列表 - 这些词是情感分类的重要特征
POS_KEYS = ["好", "棒", "赞", "喜欢", "满意", "棒极了", "太棒了", "超赞"]  # 正面关键词
NEG_KEYS = ["差", "糟", "烂", "失望", "后悔", "极差", "太烂", "坑人"]  # 负面关键词

# 正面情感模板
TEMPLATES_POS = [
    "这家{}真的很{}，下次还来",  # 示例：这家餐厅真的很棒
    "这款{}设计让我非常{}",  # 示例：这款产品设计让我非常满意
    "{}的服务态度让我感到{}",  # 示例：服务的服务态度让我感到贴心
    "{}体验非常{}，强烈推荐",  # 示例：产品体验非常赞，强烈推荐
    "这次购物感觉{}极了",  # 示例：这次购物感觉好极了
    "太{}了，{}超出预期",  # 示例：太棒了，产品超出预期
    "用起来很{}，效果特别{}",  # 示例：用起来很方便，效果特别好
    "朋友推荐的，果然没有{}",  # 示例：朋友推荐的，果然没有失望
    "{}很高，值得入手",  # 示例：性价比很高，值得入手
    "包装精美，{}质量{}",  # 示例：包装精美，产品质量好
]

# 负面情感模板
TEMPLATES_NEG = [
    "这家{}太{}了，再也不会来",  # 示例：这家餐厅太差了
    "这款{}设计让我非常{}",  # 示例：这款产品设计让我非常失望
    "{}的服务态度让人感到{}",  # 示例：服务的服务态度让人感到恶心
    "{}体验{}透顶，强烈不推荐",  # 示例：产品体验糟透顶，强烈不推荐
    "这次购物感觉{}透了",  # 示例：这次购物感觉糟透了
    "太{}了，{}完全不符合描述",  # 示例：太烂了，产品完全不符合描述
    "用起来很{}，效果特别{}",  # 示例：用起来很卡顿，效果特别差
    "朋友推荐的，结果上当{}",  # 示例：朋友推荐的，结果上当受骗
    "{}很低，完全不值得",  # 示例：性价比很低，完全不值得
    "包装破损，{}质量{}",  # 示例：包装破损，产品质量差
]

# 中性情感模板
TEMPLATES_NEU = [
    "今天去{}看了看，{}感觉",  # 示例：今天去餐厅看了看，一般感觉
    "这款{}设计比较{}",  # 示例：这款产品设计比较普通
    "{}的服务态度{}般",  # 示例：服务的服务态度一般般
    "{}体验{}",  # 示例：产品体验中规中矩
    "这次购物感觉{}吧",  # 示例：这次购物感觉还行吧
    "对{}没有{}印象",  # 示例：对产品没有特别印象
    "用起来还算{}",  # 示例：用起来还算正常
    "朋友提到过{}",  # 示例：朋友提到过产品
    "{}适中，质量也{}",  # 示例：价格适中，质量也一般
    "包装{}，{}也就那样",  # 示例：包装普通，产品也就那样
]

# 对象词列表 - 描述评价的对象
OBJ_WORDS = [
    "店铺",
    "餐厅",
    "产品",
    "服务",
    "环境",
    "系统",
    "设计",
    "课程",
    "应用",
    "酒店",
]

# 形容词列表 - 用于补充描述
ADJ_POS = [
    "方便",
    "简洁",
    "独特",
    "舒适",
    "高效",
    "贴心",
    "专业",
    "周到",
    "出色",
    "完美",
]  # 正面形容词
ADJ_NEG = [
    "麻烦",
    "复杂",
    "简陋",
    "压抑",
    "低效",
    "冷漠",
    "业余",
    "敷衍",
    "糟糕",
    "差劲",
]  # 负面形容词
ADJ_NEU = ["一般", "普通", "还行", "正常", "尚可", "平淡", "寻常"]  # 中性形容词


# 正面情感句子生成
def make_positive():
    kw = random.choice(POS_KEYS)
    tmpl = random.choice(TEMPLATES_POS)
    obj = random.choice(OBJ_WORDS)
    adj = random.choice(ADJ_POS)

    try:
        sent = tmpl.format(obj, kw)  # 尝试填充2个占位符
    except Exception:
        try:
            sent = tmpl.format(obj)  # 尝试填充1个占位符
        except Exception:
            sent = obj + kw + adj  # 如果模板没有占位符，直接拼接

    # 30%概率随机插入额外词汇，增加多样性
    if random.random() < 0.3:
        extra = random.choice(POS_KEYS + ADJ_POS)
        pos = random.randint(0, len(sent))
        sent = sent[:pos] + extra + sent[pos:]

    return sent


# 负面情感句子生成
def make_negative():
    kw = random.choice(NEG_KEYS)
    tmpl = random.choice(TEMPLATES_NEG)
    obj = random.choice(OBJ_WORDS)
    adj = random.choice(ADJ_NEG)

    try:
        sent = tmpl.format(obj, kw)
    except Exception:
        try:
            sent = tmpl.format(obj)
        except Exception:
            sent = obj + kw + adj

    if random.random() < 0.3:
        extra = random.choice(NEG_KEYS + ADJ_NEG)
        pos = random.randint(0, len(sent))
        sent = sent[:pos] + extra + sent[pos:]

    return sent


# 中性情感句子生成
def make_neutral():
    tmpl = random.choice(TEMPLATES_NEU)
    obj = random.choice(OBJ_WORDS)
    adj = random.choice(ADJ_NEU)

    try:
        sent = tmpl.format(obj, adj)
    except Exception:
        try:
            sent = tmpl.format(obj)
        except Exception:
            sent = "关于" + obj + "的评价" + adj

    if random.random() < 0.2:  # 中性句子变化不需要太多
        extra = random.choice(ADJ_NEU)
        pos = random.randint(0, len(sent))
        sent = sent[:pos] + extra + sent[pos:]

    return sent


# 数据集生成
def build_dataset(n=N_SAMPLES):
    data = []
    n_per_class = n // 3
    for _ in range(n_per_class):
        data.append((make_positive(), 2))  # 正面标签为2
        data.append((make_negative(), 0))  # 负面标签为0
        data.append((make_neutral(), 1))  # 中性标签为1

    # 随机打乱数据集顺序，避免模型学到顺序特征
    random.shuffle(data)

    return data


# 词表构建与编码
# ============================================================


# 构建字符级词表构建让出现的文字都有一个ID
def build_vocab(data):
    # <PAD>: 填充符号，用于将短句子补齐到MAXLEN
    # <UNK>: 未知字符，用于处理词表中没有的字符
    vocab = {"<PAD>": 0, "<UNK>": 1}

    # 因为训练集和测试集共用，所以UNK是可能会出现的字符
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)

    return vocab


# 将句子编码为数字序列，用于模型输入
def encode(sent, vocab, maxlen=MAXLEN):
    """
    :param sent: 输入句子
    :param vocab: 词表
    :param maxlen: 最大长度
    :return: 数字序列
    """
    # 将每个字符转换为对应的ID，未知字符用<UNK>的ID(1)
    ids = [vocab.get(ch, 1) for ch in sent]

    # 截断过长的句子
    ids = ids[:maxlen]

    # 用<PAD>的ID(0)填充过短的句子
    ids += [0] * (maxlen - len(ids))

    return ids


# PyTorch数据加载机制，将数据封装为可迭代的批次
# ============================================================


class TextDataset(Dataset):
    """
    继承PyTorch的Dataset类
    必有__len__, __getitem__方法
    __init__可以看情况
    """

    def __init__(self, data, vocab):
        """
        初始化数据集
        :param data: 原始数据列表，每个元素是(句子, 标签)
        :param vocab: 词表
        """
        # 将句子编码为数字序列，用于模型输入
        self.X = [encode(s, vocab) for s, _ in data]
        # 提取所有对应标签
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),  # 输入序列，long类型
            torch.tensor(self.y[i], dtype=torch.long),  # 标签，long类型（多分类用long）
        )


# 生成RNN/LSTM/GRU 文本分类模型
# ============================================================


class TextClassifier(nn.Module):
    """
    中文文本多分类器
    架构：Embedding → RNN/LSTM/GRU → 取最后隐藏状态 → Dropout → Linear

    【模型选择说明】
    - RNN: 基础循环神经网络，梯度消失问题较严重
    - LSTM: 长短期记忆网络，通过门机制缓解梯度消失
    - GRU: 门控循环单元，LSTM的简化版本，参数量更少
    """

    def __init__(
        self,
        vocab_size,  # 词表大小
        num_classes=3,  # 类别数（负面/中性/正面）
        embed_dim=EMBED_DIM,  # 词嵌入维度
        hidden_dim=HIDDEN_DIM,  # 隐藏层维度
        n_layers=N_LAYERS,  # RNN层数
        dropout=DROPOUT,  # Dropout比率
        model_type=MODEL_TYPE,  # 模型类型
    ):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 词嵌入层
        # 将字符ID转换为稠密向量
        # padding_idx=0 表示<PAD>符号不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # RNN类型通过字典映射实现模型切换
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[model_type]
        self.rnn = rnn_cls(
            embed_dim,  # 输入维度（词嵌入维度）
            hidden_dim,  # 隐藏层维度
            num_layers=n_layers,  # 层数
            batch_first=True,  # 输入格式为(batch, seq_len, feature)
            dropout=dropout if n_layers > 1 else 0,  # 多层时使用Dropout
            bidirectional=False,  # 单向RNN
        )

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 全连接层，将隐藏状态映射到类别空间
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为(batch_size, seq_len)
        :return: 输出logits，形状为(batch_size, num_classes)
        """
        # 1. 词嵌入
        # x: (B, L) -> embedded: (B, L, embed_dim)
        embedded = self.embedding(x)

        # 2. RNN前向传播
        # LSTM返回(output, (hidden, cell))
        # RNN/GRU返回(output, hidden)
        if self.model_type == "LSTM":
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        # 3. 取最后一层的最后时间步隐藏状态
        # hidden形状: (n_layers, B, hidden_dim)
        # 取[-1]即最后一层的隐藏状态
        last_hidden = hidden[-1]  # 形状: (B, hidden_dim)

        # 4. Dropout和全连接
        out = self.dropout(last_hidden)
        out = self.fc(out)  # 形状: (B, num_classes)

        return out  # 返回logits（未经过softmax）


# ============================================================
# 【5. 训练与评估模块】
# 训练循环和评估逻辑
# ============================================================


def evaluate(model, loader, device):
    """
    评估模型性能
    :param model: 模型
    :param loader: 数据加载器
    :param device: 设备（CPU/GPU）
    :return: 准确率、预测结果、真实标签
    """
    model.eval()  # 设置模型为评估模式，关闭Dropout等训练特有的操作
    correct = total = 0
    all_preds = []  # 存储所有预测结果
    all_labels = []  # 存储所有真实标签

    # torch.no_grad() 上下文管理器，禁用梯度计算，节省内存和计算资源
    with torch.no_grad():
        for X, y in loader:
            # 将数据移动到指定设备
            X, y = X.to(device), y.to(device)

            # 前向传播，得到logits
            logits = model(X)

            # 取概率最大的类别作为预测
            preds = logits.argmax(dim=1)

            # 统计正确预测数和总数
            correct += (preds == y).sum().item()
            total += len(y)

            # 收集预测和标签用于后续分析
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = correct / total
    return acc, all_preds, all_labels


def print_confusion_matrix(preds, labels, class_names):
    """
    打印混淆矩阵，分析各类别的分类情况
    :param preds: 预测结果列表
    :param labels: 真实标签列表
    :param class_names: 类别名称列表
    """
    n = len(class_names)
    # 初始化混淆矩阵，n行n列，初始值为0
    matrix = [[0] * n for _ in range(n)]

    # 统计每个类别的预测情况
    for p, l in zip(preds, labels):
        matrix[l][p] += 1  # matrix[真实类别][预测类别] += 1

    # 打印混淆矩阵
    print("\n混淆矩阵:")
    # 打印表头（预测类别）
    print("      ", "  ".join(f"{name:>6}" for name in class_names))
    # 打印每一行（真实类别）
    for i, name in enumerate(class_names):
        print(f"{name:>6}", "  ".join(f"{matrix[i][j]:>6}" for j in range(n)))


def train():
    """主训练函数"""
    # 1. 选择设备：优先使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"模型类型: {MODEL_TYPE}")

    # 2. 生成数据集和词表
    print("\n生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 3. 划分训练集和验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]  # 前80%作为训练集
    val_data = data[split:]  # 后20%作为验证集
    print(f"  训练集：{len(train_data)}，验证集：{len(val_data)}")

    # 4. 创建DataLoader
    # DataLoader负责将数据集切分为批次，并在训练时打乱顺序
    train_loader = DataLoader(
        TextDataset(train_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练集需要打乱
    )
    val_loader = DataLoader(
        TextDataset(val_data, vocab),
        batch_size=BATCH_SIZE,
        # 验证集不需要打乱
    )

    # 5. 初始化模型、损失函数、优化器
    model = TextClassifier(vocab_size=len(vocab)).to(device)  # 移动到指定设备
    criterion = nn.CrossEntropyLoss()  # 多分类损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam优化器
    # 学习率调度器：每10轮学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 6. 训练循环
    best_val_acc = 0.0  # 记录最佳验证准确率
    class_names = ["负面", "中性", "正面"]  # 类别名称

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0

        # 遍历训练集的每个批次
        for X, y in train_loader:
            # 将数据移动到指定设备
            X, y = X.to(device), y.to(device)

            # 前向传播
            logits = model(X)
            loss = criterion(logits, y)  # 计算损失

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数

            # 累加损失
            total_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 计算平均损失和验证准确率
        avg_loss = total_loss / len(train_loader)
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device)

        # 打印训练信息
        print(
            f"Epoch {epoch:2d}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # 更新最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 7. 训练结束，输出最终结果
    print(f"\n最佳验证准确率：{best_val_acc:.4f}")

    # 打印混淆矩阵
    final_acc, final_preds, final_labels = evaluate(model, val_loader, device)
    print_confusion_matrix(final_preds, final_labels, class_names)

    # 打印各类别准确率
    print("\n各类别统计:")
    for i, name in enumerate(class_names):
        # 筛选出属于当前类别的样本
        mask = [l == i for l in final_labels]
        if any(mask):
            # 计算当前类别的正确预测数
            class_correct = sum(
                1 for p, l, m in zip(final_preds, final_labels, mask) if m and p == l
            )
            class_total = sum(mask)
            print(
                f"  {name}: {class_correct}/{class_total} = {class_correct/class_total:.4f}"
            )

    # 8. 推理示例：用训练好的模型预测新句子
    print("\n--- 推理示例 ---")
    model.eval()  # 设置为评估模式
    test_sents = [
        "这款产品真的很棒，非常满意",
        "这家餐厅太差了，再也不会来",
        "今天去看了看，没什么特别感觉",
        "服务太赞了，下次还来",
        "用起来很卡顿，效果特别差",
        "对这款产品没有特别的印象",
    ]
    with torch.no_grad():
        for sent in test_sents:
            # 将句子编码并转换为张量
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long).to(device)
            # 前向传播得到logits
            logits = model(ids)
            # 转换为概率分布
            prob = torch.softmax(logits, dim=1)
            # 取概率最大的类别
            pred = logits.argmax(dim=1).item()
            # 获取置信度
            confidence = prob[0][pred].item()
            # 获取类别名称
            label = class_names[pred]
            print(f"  [{label}({confidence:.2f})]  {sent}")


if __name__ == "__main__":
    # 执行训练
    train()
