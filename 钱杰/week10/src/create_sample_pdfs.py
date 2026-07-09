import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdf"
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

sample_content = {
    "ai_basics.pdf": """人工智能基础入门

第一章：什么是人工智能

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。人工智能领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

人工智能的发展历程可以分为三个阶段：
1. 弱人工智能（Narrow AI）：只能执行特定任务的AI，如语音助手、图像识别等
2. 强人工智能（General AI）：具备人类智能水平的AI，可以执行任何智力任务
3. 超人工智能（Super AI）：超越人类智能的AI，目前仍处于理论阶段

第二章：机器学习概述

机器学习是人工智能的核心技术之一，它使计算机系统能够从数据中学习并改进性能，而无需明确编程。机器学习算法可以分为以下几类：

监督学习：使用标记数据进行训练，如分类和回归
无监督学习：从未标记数据中发现模式，如聚类和降维
强化学习：通过与环境交互获得奖励来学习，如游戏AI

第三章：深度学习基础

深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。

常用的深度学习架构包括：
- 卷积神经网络（CNN）：主要用于图像和视频处理
- 循环神经网络（RNN）：主要用于序列数据处理
- Transformer：主要用于自然语言处理，如GPT、BERT等

第四章：人工智能应用领域

人工智能已经广泛应用于多个领域：
1. 医疗健康：疾病诊断、药物研发、医疗影像分析
2. 金融科技：风险评估、投资分析、智能客服
3. 交通运输：自动驾驶、交通优化、物流管理
4. 教育领域：智能辅导、个性化学习、教育评估
5. 智能制造：质量检测、预测维护、生产优化

第五章：人工智能的未来展望

随着技术的不断进步，人工智能将在更多领域发挥重要作用。未来的发展趋势包括：
- 边缘计算与AI的结合
- 可解释AI的发展
- AI与物联网的融合
- 伦理和安全问题的关注

结语

人工智能正在改变我们的生活和工作方式，了解AI的基础知识有助于我们更好地适应这个快速变化的时代。""",
    
    "climate_change.pdf": """气候变化与环境保护

第一章：气候变化的基本概念

气候变化是指地球气候系统的长期变化，包括温度、降水、风向等要素的改变。近年来，全球气候变化日益明显，主要表现为全球变暖、极端天气事件增多等现象。

全球变暖的主要原因是人类活动导致的温室气体排放增加，特别是二氧化碳（CO2）、甲烷（CH4）和氧化亚氮（N2O）。

第二章：气候变化的影响

气候变化对地球生态系统和人类社会产生了广泛影响：

生态系统影响：
- 冰川融化和海平面上升
- 生物多样性减少
- 生态系统失衡

人类社会影响：
- 农业产量下降和粮食安全问题
- 水资源短缺
- 健康风险增加
- 自然灾害频发

第三章：应对气候变化的措施

为了应对气候变化，国际社会采取了一系列措施：

减少温室气体排放：
- 发展可再生能源（太阳能、风能、水能等）
- 提高能源效率
- 推广电动汽车和公共交通
- 实施碳定价和碳交易

适应气候变化：
- 建设防洪设施
- 发展抗旱作物
- 改善城市规划

国际合作：
- 巴黎协定：全球气候协议
- 碳达峰和碳中和目标
- 技术转让和资金支持

第四章：个人可以做什么

每个人都可以为应对气候变化做出贡献：
1. 节约能源：随手关灯、使用节能电器
2. 绿色出行：步行、骑行或乘坐公共交通
3. 减少浪费：垃圾分类、重复使用物品
4. 健康饮食：减少肉类消费、节约粮食
5. 关注环保：支持环保组织、传播环保知识

第五章：未来展望

气候变化是人类面临的共同挑战，需要全球合作和长期努力。通过技术创新和社会变革，我们有能力减缓气候变化的影响，保护我们共同的地球家园。""",
    
    "python_programming.pdf": """Python编程入门教程

第一章：Python简介

Python是一种高级、通用、解释型编程语言。它以简洁的语法和强大的功能著称，广泛应用于Web开发、数据分析、人工智能、科学计算等领域。

Python的特点：
- 语法简洁，易于学习和阅读
- 跨平台，支持多种操作系统
- 丰富的标准库和第三方库
- 面向对象和函数式编程支持
- 强大的社区支持

第二章：Python基础语法

变量和数据类型：
- 整数（int）：如 1, 2, 3
- 浮点数（float）：如 1.5, 2.0
- 字符串（str）：如 "Hello", 'World'
- 布尔值（bool）：True, False

基本运算符：
- 算术运算：+ - * / % ** //
- 比较运算：== != > < >= <=
- 逻辑运算：and or not

第三章：控制结构

条件语句：
if condition:
    statement
elif condition:
    statement
else:
    statement

循环语句：
for item in sequence:
    statement

while condition:
    statement

第四章：函数和模块

定义函数：
def function_name(parameters):
    statement
    return value

模块导入：
import module_name
from module_name import function_name

第五章：数据结构

列表（List）：有序、可变的元素集合
my_list = [1, 2, 3, 'hello']

字典（Dictionary）：键值对的无序集合
my_dict = {'name': 'Alice', 'age': 25}

集合（Set）：无序、不重复的元素集合
my_set = {1, 2, 3, 3, 4}

第六章：面向对象编程

类的定义：
class ClassName:
    def __init__(self, parameters):
        self.attribute = value
    
    def method(self, parameters):
        statement

对象创建：
obj = ClassName(arguments)

第七章：文件操作

读取文件：
with open('file.txt', 'r') as f:
    content = f.read()

写入文件：
with open('file.txt', 'w') as f:
    f.write(content)

第八章：异常处理

try:
    risky_operation()
except ExceptionType:
    handle_exception()
finally:
    cleanup()

结语

Python是一门功能强大且易于学习的编程语言，掌握Python可以为你的编程之旅打下坚实的基础。""",
    
    "space_exploration.pdf": """太空探索与航天技术

第一章：人类太空探索简史

人类对太空的探索始于20世纪中叶，经历了多个重要阶段：

1957年：苏联发射第一颗人造卫星斯普特尼克1号
1961年：苏联宇航员尤里·加加林成为第一个进入太空的人类
1969年：美国阿波罗11号成功登月，尼尔·阿姆斯特朗成为第一个踏上月球的人类
1971年：苏联发射第一个空间站礼炮1号
1981年：美国航天飞机哥伦比亚号首次发射
1998年：国际空间站开始建设
2003年：中国神舟五号载人飞船成功发射

第二章：主要航天技术

火箭技术：
- 化学火箭：使用液体或固体燃料
- 电推进：使用电能加速离子
- 可重复使用火箭：降低发射成本

卫星技术：
- 通信卫星：全球通信网络
- 气象卫星：天气预报和气候监测
- 导航卫星：GPS、北斗等导航系统
- 遥感卫星：地球资源监测和环境研究

空间站技术：
- 长期载人空间研究
- 微重力实验平台
- 国际合作项目

第三章：太空探索的科学价值

太空探索带来了许多科学发现：
- 太阳系形成和演化的研究
- 地球气候和环境的监测
- 生命起源和宇宙生物学研究
- 基础物理和化学实验

第四章：商业航天的兴起

近年来，商业航天发展迅速：
- SpaceX：可重复使用火箭和星链计划
- Blue Origin：亚轨道旅游和重型火箭
- 卫星互联网：全球覆盖的高速网络
- 太空旅游：普通人进入太空的机会

第五章：未来展望

未来的太空探索目标包括：
- 重返月球和建立月球基地
- 载人火星任务
- 小行星采矿
- 深空探测任务

结语

太空探索不仅扩展了人类的知识边界，也推动了科技的进步和国际合作。"""
}


def create_pdf(text: str, output_path: Path):
    from PyPDF2 import PdfWriter
    
    writer = PdfWriter()
    
    lines = text.split('\n')
    current_line = 0
    lines_per_page = 50
    
    while current_line < len(lines):
        page_text = '\n'.join(lines[current_line:current_line + lines_per_page])
        current_line += lines_per_page
        
        from io import StringIO
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        try:
            pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
            font_name = 'SimSun'
        except:
            font_name = 'Helvetica'
        
        buffer = StringIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        style = getSampleStyleSheet()['Normal']
        style.fontName = font_name
        style.fontSize = 12
        style.leading = 18
        
        elements = []
        for line in page_text.split('\n'):
            elements.append(Paragraph(line, style))
        
        doc.build(elements)
        
        buffer.seek(0)
        from PyPDF2 import PdfReader
        reader = PdfReader(buffer)
        for page in reader.pages:
            writer.add_page(page)
    
    with open(output_path, 'wb') as f:
        writer.write(f)


def create_text_pdf(text: str, output_path: Path):
    lines = text.split('\n')
    
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    try:
        pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
        font_name = 'SimSun'
    except:
        font_name = 'Helvetica'
    
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=A4)
    
    can.setFont(font_name, 12)
    
    x, y = 50, 750
    line_height = 18
    
    for line in lines:
        if y < 50:
            can.showPage()
            can.setFont(font_name, 12)
            y = 750
        
        if len(line) > 80:
            words = line
            while len(words) > 80:
                can.drawString(x, y, words[:80])
                y -= line_height
                words = words[80:]
            can.drawString(x, y, words)
        else:
            can.drawString(x, y, line)
        
        y -= line_height
    
    can.save()
    
    packet.seek(0)
    with open(output_path, 'wb') as f:
        f.write(packet.read())


def main():
    existing_pdfs = list(RAW_PDF_DIR.glob("*.pdf"))
    
    for pdf in existing_pdfs:
        if pdf.name in sample_content.keys():
            pdf.unlink()
    
    for filename, content in sample_content.items():
        output_path = RAW_PDF_DIR / filename
        try:
            create_text_pdf(content, output_path)
            logger.info(f"创建成功: {filename}")
        except Exception as e:
            logger.error(f"创建失败 {filename}: {e}")
            with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"已创建txt文件: {filename}.txt")
    
    logger.info("示例文件创建完成！")


if __name__ == "__main__":
    main()