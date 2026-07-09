import requests
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_pdf"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_pdf(url, filename):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filepath = DATA_DIR / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"下载成功: {filename}")
        return True
    except Exception as e:
        print(f"下载失败 {filename}: {e}")
        return False


pdf_urls = [
    ("https://arxiv.org/pdf/2310.06825.pdf", "llama3_paper.pdf"),
    ("https://arxiv.org/pdf/2103.00020.pdf", "clip_paper.pdf"),
    ("https://arxiv.org/pdf/2203.15556.pdf", "flamingo_paper.pdf"),
]


def create_simple_pdf(text, filename):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    filepath = DATA_DIR / filename
    c = canvas.Canvas(str(filepath), pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica", 12)
    lines = text.split('\n')
    y = height - 50
    for line in lines:
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
        c.drawString(50, y, line[:100])
        y -= 15
    c.save()
    print(f"生成: {filename}")


def generate_text_pdfs():
    ai_intro = """人工智能技术导论

第一章 概述
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。

人工智能的发展历程可以分为三个阶段：
1. 符号主义阶段（1956-1980）：基于规则和逻辑推理
2. 连接主义阶段（1980-2010）：基于神经网络和机器学习
3. 深度学习阶段（2010-至今）：基于深度神经网络和大数据

第二章 机器学习基础
机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习并做出预测或决策，而无需被明确编程。

主要的机器学习算法包括：
- 监督学习：线性回归、逻辑回归、支持向量机、决策树
- 无监督学习：聚类、降维、关联规则挖掘
- 强化学习：Q-learning、策略梯度、深度Q网络

第三章 深度学习
深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的高级特征表示。

常用的深度学习架构：
- CNN（卷积神经网络）：用于图像识别
- RNN/LSTM（循环神经网络）：用于序列数据
- Transformer：用于自然语言处理，如BERT、GPT

第四章 自然语言处理
自然语言处理（NLP）是人工智能的一个分支，使计算机能够理解、处理和生成人类语言。

NLP的主要任务包括：
- 分词、词性标注、命名实体识别
- 文本分类、情感分析、机器翻译
- 问答系统、文本摘要、对话系统

第五章 人工智能应用
人工智能技术已经广泛应用于各个领域：
- 医疗健康：疾病诊断、药物研发
- 金融科技：风险评估、智能投顾
- 自动驾驶：感知、决策、控制
- 智能制造：质量检测、工艺优化"""

    python_intro = """Python编程入门

第一章 Python简介
Python是一种高级、通用、解释型编程语言，由Guido van Rossum于1991年创建。

Python的特点：
- 语法简洁优雅，易于学习
- 跨平台，可在Windows、Linux、Mac上运行
- 丰富的标准库和第三方库
- 支持面向对象、函数式、过程式编程

第二章 基础语法
Python的基础语法包括变量、数据类型、控制流程等。

常用数据类型：
- 数值类型：int、float、complex
- 序列类型：list、tuple、str
- 映射类型：dict
- 集合类型：set、frozenset

第三章 函数与模块
函数是组织好的、可重复使用的代码块，用于执行单一、相关的功能。

Python中的内置函数：
- print()：输出信息
- len()：返回长度
- range()：生成序列
- type()：返回类型

常用第三方库：
NumPy：数值计算
Pandas：数据分析
Matplotlib：数据可视化
TensorFlow：深度学习
PyTorch：深度学习

第四章 面向对象编程
面向对象编程（OOP）是一种编程范式，使用对象来组织代码。

Python中的OOP概念：
- 类（class）：对象的模板
- 对象（object）：类的实例
- 继承（inheritance）：子类继承父类属性
- 封装（encapsulation）：隐藏内部实现
- 多态（polymorphism）：同一接口不同实现

第五章 文件操作
Python提供了丰富的文件操作功能。

常用文件操作：
- open()：打开文件
- read()：读取文件内容
- write()：写入文件内容
- close()：关闭文件"""

    dl_practice = """深度学习实战

第一章 深度学习概述
深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的高级特征表示。

深度学习的关键组件：
- 神经网络层：卷积层、池化层、全连接层
- 激活函数：ReLU、Sigmoid、Tanh
- 损失函数：MSE、交叉熵
- 优化器：SGD、Adam、RMSprop

第二章 卷积神经网络
卷积神经网络（CNN）是一种专门用于处理具有网格结构数据的神经网络，如图像。

CNN的核心操作：
- 卷积（Convolution）：提取局部特征
- 池化（Pooling）：降维，保留关键信息
- 批归一化（Batch Normalization）：加速收敛

经典模型：
LeNet-5（1998）：首个实用CNN
AlexNet（2012）：ImageNet冠军，深度学习复兴
VGG（2014）：更深网络，统一架构
ResNet（2015）：残差连接，解决梯度消失
Transformer（2017）：自注意力机制

第三章 循环神经网络
循环神经网络（RNN）是一种专门用于处理序列数据的神经网络。

RNN的特点：
- 具有记忆功能，能记住之前的输入
- 适合处理时间序列、文本等顺序数据
- 存在梯度消失和梯度爆炸问题

第四章 Transformer架构
Transformer是一种基于自注意力机制的神经网络架构，由Google在2017年提出。

Transformer的核心组件：
- 自注意力（Self-Attention）：计算序列中每个位置与其他位置的关系
- 多头注意力（Multi-Head Attention）：多个注意力头并行计算
- 位置编码（Positional Encoding）：为序列添加位置信息
- 前馈神经网络（Feed-Forward）：对每个位置独立处理

第五章 实战项目
深度学习实战项目包括：
- 图像分类：CIFAR-10、ImageNet
- 目标检测：YOLO、Faster R-CNN
- 语音识别：CTC、Attention-based
- 机器翻译：Transformer模型
- 文本生成：GPT系列模型"""

    create_simple_pdf(ai_intro, "人工智能技术导论.pdf")
    create_simple_pdf(python_intro, "Python编程入门.pdf")
    create_simple_pdf(dl_practice, "深度学习实战.pdf")


if __name__ == "__main__":
    print("尝试从网络下载示例PDF...")
    for url, filename in pdf_urls:
        download_pdf(url, filename)
    
    print("\n尝试生成本地示例PDF...")
    try:
        generate_text_pdfs()
    except ImportError:
        print("reportlab未安装，跳过本地PDF生成")
    
    print("\n可用的PDF文件：")
    for f in DATA_DIR.glob("*.pdf"):
        size = os.path.getsize(f) / 1024
        print(f"  - {f.name} ({size:.1f} KB)")