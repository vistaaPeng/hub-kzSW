from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_pdf"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf_1():
    pdf_path = DATA_DIR / "人工智能技术导论.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("人工智能技术导论", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第一章 概述", styles['Heading1']))
    elements.append(Paragraph("人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。", styles['BodyText']))
    elements.append(Paragraph("人工智能的发展历程可以分为三个阶段：", styles['BodyText']))
    elements.append(Paragraph("1. 符号主义阶段（1956-1980）：基于规则和逻辑推理", styles['BodyText']))
    elements.append(Paragraph("2. 连接主义阶段（1980-2010）：基于神经网络和机器学习", styles['BodyText']))
    elements.append(Paragraph("3. 深度学习阶段（2010-至今）：基于深度神经网络和大数据", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第二章 机器学习基础", styles['Heading1']))
    elements.append(Paragraph("机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习并做出预测或决策，而无需被明确编程。", styles['BodyText']))
    elements.append(Paragraph("主要的机器学习算法包括：", styles['BodyText']))
    elements.append(Paragraph("- 监督学习：线性回归、逻辑回归、支持向量机、决策树", styles['BodyText']))
    elements.append(Paragraph("- 无监督学习：聚类、降维、关联规则挖掘", styles['BodyText']))
    elements.append(Paragraph("- 强化学习：Q-learning、策略梯度、深度Q网络", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第三章 深度学习", styles['Heading1']))
    elements.append(Paragraph("深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的高级特征表示。", styles['BodyText']))
    elements.append(Paragraph("常用的深度学习架构：", styles['BodyText']))
    elements.append(Paragraph("- CNN（卷积神经网络）：用于图像识别", styles['BodyText']))
    elements.append(Paragraph("- RNN/LSTM（循环神经网络）：用于序列数据", styles['BodyText']))
    elements.append(Paragraph("- Transformer：用于自然语言处理，如BERT、GPT", styles['BodyText']))
    elements.append(Spacer(1, 12))

    data = [
        ['模型', '发布年份', '参数数量', '主要应用'],
        ['BERT', '2018', '110M', '文本分类、问答'],
        ['GPT-2', '2019', '1.5B', '文本生成'],
        ['GPT-3', '2020', '175B', '通用语言任务'],
        ['GPT-4', '2023', '未知', '多模态理解'],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第四章 自然语言处理", styles['Heading1']))
    elements.append(Paragraph("自然语言处理（NLP）是人工智能的一个分支，使计算机能够理解、处理和生成人类语言。", styles['BodyText']))
    elements.append(Paragraph("NLP的主要任务包括：", styles['BodyText']))
    elements.append(Paragraph("- 分词、词性标注、命名实体识别", styles['BodyText']))
    elements.append(Paragraph("- 文本分类、情感分析、机器翻译", styles['BodyText']))
    elements.append(Paragraph("- 问答系统、文本摘要、对话系统", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第五章 人工智能应用", styles['Heading1']))
    elements.append(Paragraph("人工智能技术已经广泛应用于各个领域：", styles['BodyText']))
    elements.append(Paragraph("- 医疗健康：疾病诊断、药物研发", styles['BodyText']))
    elements.append(Paragraph("- 金融科技：风险评估、智能投顾", styles['BodyText']))
    elements.append(Paragraph("- 自动驾驶：感知、决策、控制", styles['BodyText']))
    elements.append(Paragraph("- 智能制造：质量检测、工艺优化", styles['BodyText']))

    doc.build(elements)
    print(f"生成: {pdf_path.name}")


def create_pdf_2():
    pdf_path = DATA_DIR / "Python编程入门.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Python编程入门", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第一章 Python简介", styles['Heading1']))
    elements.append(Paragraph("Python是一种高级、通用、解释型编程语言，由Guido van Rossum于1991年创建。", styles['BodyText']))
    elements.append(Paragraph("Python的特点：", styles['BodyText']))
    elements.append(Paragraph("- 语法简洁优雅，易于学习", styles['BodyText']))
    elements.append(Paragraph("- 跨平台，可在Windows、Linux、Mac上运行", styles['BodyText']))
    elements.append(Paragraph("- 丰富的标准库和第三方库", styles['BodyText']))
    elements.append(Paragraph("- 支持面向对象、函数式、过程式编程", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第二章 基础语法", styles['Heading1']))
    elements.append(Paragraph("Python的基础语法包括变量、数据类型、控制流程等。", styles['BodyText']))
    elements.append(Paragraph("常用数据类型：", styles['BodyText']))
    elements.append(Paragraph("- 数值类型：int、float、complex", styles['BodyText']))
    elements.append(Paragraph("- 序列类型：list、tuple、str", styles['BodyText']))
    elements.append(Paragraph("- 映射类型：dict", styles['BodyText']))
    elements.append(Paragraph("- 集合类型：set、frozenset", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第三章 函数与模块", styles['Heading1']))
    elements.append(Paragraph("函数是组织好的、可重复使用的代码块，用于执行单一、相关的功能。", styles['BodyText']))
    elements.append(Paragraph("Python中的内置函数：", styles['BodyText']))
    elements.append(Paragraph("- print()：输出信息", styles['BodyText']))
    elements.append(Paragraph("- len()：返回长度", styles['BodyText']))
    elements.append(Paragraph("- range()：生成序列", styles['BodyText']))
    elements.append(Paragraph("- type()：返回类型", styles['BodyText']))
    elements.append(Spacer(1, 12))

    data = [
        ['库名', '用途', '安装命令'],
        ['NumPy', '数值计算', 'pip install numpy'],
        ['Pandas', '数据分析', 'pip install pandas'],
        ['Matplotlib', '数据可视化', 'pip install matplotlib'],
        ['TensorFlow', '深度学习', 'pip install tensorflow'],
        ['PyTorch', '深度学习', 'pip install torch'],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第四章 面向对象编程", styles['Heading1']))
    elements.append(Paragraph("面向对象编程（OOP）是一种编程范式，使用对象来组织代码。", styles['BodyText']))
    elements.append(Paragraph("Python中的OOP概念：", styles['BodyText']))
    elements.append(Paragraph("- 类（class）：对象的模板", styles['BodyText']))
    elements.append(Paragraph("- 对象（object）：类的实例", styles['BodyText']))
    elements.append(Paragraph("- 继承（inheritance）：子类继承父类属性", styles['BodyText']))
    elements.append(Paragraph("- 封装（encapsulation）：隐藏内部实现", styles['BodyText']))
    elements.append(Paragraph("- 多态（polymorphism）：同一接口不同实现", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第五章 文件操作", styles['Heading1']))
    elements.append(Paragraph("Python提供了丰富的文件操作功能。", styles['BodyText']))
    elements.append(Paragraph("常用文件操作：", styles['BodyText']))
    elements.append(Paragraph("- open()：打开文件", styles['BodyText']))
    elements.append(Paragraph("- read()：读取文件内容", styles['BodyText']))
    elements.append(Paragraph("- write()：写入文件内容", styles['BodyText']))
    elements.append(Paragraph("- close()：关闭文件", styles['BodyText']))

    doc.build(elements)
    print(f"生成: {pdf_path.name}")


def create_pdf_3():
    pdf_path = DATA_DIR / "深度学习实战.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("深度学习实战", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第一章 深度学习概述", styles['Heading1']))
    elements.append(Paragraph("深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的高级特征表示。", styles['BodyText']))
    elements.append(Paragraph("深度学习的关键组件：", styles['BodyText']))
    elements.append(Paragraph("- 神经网络层：卷积层、池化层、全连接层", styles['BodyText']))
    elements.append(Paragraph("- 激活函数：ReLU、Sigmoid、Tanh", styles['BodyText']))
    elements.append(Paragraph("- 损失函数：MSE、交叉熵", styles['BodyText']))
    elements.append(Paragraph("- 优化器：SGD、Adam、RMSprop", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第二章 卷积神经网络", styles['Heading1']))
    elements.append(Paragraph("卷积神经网络（CNN）是一种专门用于处理具有网格结构数据的神经网络，如图像。", styles['BodyText']))
    elements.append(Paragraph("CNN的核心操作：", styles['BodyText']))
    elements.append(Paragraph("- 卷积（Convolution）：提取局部特征", styles['BodyText']))
    elements.append(Paragraph("- 池化（Pooling）：降维，保留关键信息", styles['BodyText']))
    elements.append(Paragraph("- 批归一化（Batch Normalization）：加速收敛", styles['BodyText']))
    elements.append(Spacer(1, 12))

    data = [
        ['经典模型', '提出年份', '主要贡献'],
        ['LeNet-5', '1998', '首个实用CNN'],
        ['AlexNet', '2012', 'ImageNet冠军，深度学习复兴'],
        ['VGG', '2014', '更深网络，统一架构'],
        ['ResNet', '2015', '残差连接，解决梯度消失'],
        ['Transformer', '2017', '自注意力机制'],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第三章 循环神经网络", styles['Heading1']))
    elements.append(Paragraph("循环神经网络（RNN）是一种专门用于处理序列数据的神经网络。", styles['BodyText']))
    elements.append(Paragraph("RNN的特点：", styles['BodyText']))
    elements.append(Paragraph("- 具有记忆功能，能记住之前的输入", styles['BodyText']))
    elements.append(Paragraph("- 适合处理时间序列、文本等顺序数据", styles['BodyText']))
    elements.append(Paragraph("- 存在梯度消失和梯度爆炸问题", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第四章 Transformer架构", styles['Heading1']))
    elements.append(Paragraph("Transformer是一种基于自注意力机制的神经网络架构，由Google在2017年提出。", styles['BodyText']))
    elements.append(Paragraph("Transformer的核心组件：", styles['BodyText']))
    elements.append(Paragraph("- 自注意力（Self-Attention）：计算序列中每个位置与其他位置的关系", styles['BodyText']))
    elements.append(Paragraph("- 多头注意力（Multi-Head Attention）：多个注意力头并行计算", styles['BodyText']))
    elements.append(Paragraph("- 位置编码（Positional Encoding）：为序列添加位置信息", styles['BodyText']))
    elements.append(Paragraph("- 前馈神经网络（Feed-Forward）：对每个位置独立处理", styles['BodyText']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("第五章 实战项目", styles['Heading1']))
    elements.append(Paragraph("深度学习实战项目包括：", styles['BodyText']))
    elements.append(Paragraph("- 图像分类：CIFAR-10、ImageNet", styles['BodyText']))
    elements.append(Paragraph("- 目标检测：YOLO、Faster R-CNN", styles['BodyText']))
    elements.append(Paragraph("- 语音识别：CTC、Attention-based", styles['BodyText']))
    elements.append(Paragraph("- 机器翻译：Transformer模型", styles['BodyText']))
    elements.append(Paragraph("- 文本生成：GPT系列模型", styles['BodyText']))

    doc.build(elements)
    print(f"生成: {pdf_path.name}")


if __name__ == "__main__":
    create_pdf_1()
    create_pdf_2()
    create_pdf_3()
    print("\n示例PDF文件生成完成！")