# BERT-NER 中文命名实体识别

基于 BERT 的中文命名实体识别（NER）项目，支持识别人名（PER）、机构名（ORG）、地名（LOC）。

## 📋 环境要求

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+

## 🛠️ 安装依赖

```bash
pip install torch transformers seqeval pytorch-crf tqdm
```

## 🚀 快速开始

### 1. 训练模型

```bash
cd src
python train.py
```

### 2. 评估模型

```bash
cd src
python evaluate.py
```

### 3. 预测

```bash
python predict.py
```

## 📁 项目结构

```
├── src/              # 源代码
│   ├── train.py      # 训练脚本
│   ├── evaluate.py   # 评估脚本
│   ├── model.py      # 模型定义
│   └── dataset.py    # 数据集处理
├── data/             # 数据集
└── predict.py        # 预测脚本
```

## 📊 支持的实体类型

| 类型 | 说明 | 示例 |
|------|------|------|
| PER | 人名 | 李明 |
| ORG | 机构名 | 北京大学 |
| LOC | 地名 | 北京 |

## 📝 数据格式

数据集使用 JSON 格式，每行一个样本：

```json
{"text": "李明在北京大学工作", "labels": ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O"]}
```

## 📄 许可证

MIT License
