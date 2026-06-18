# work7 — 人民日报 NER 序列标注

在 `peoples_daily` 数据集上独立实现 BERT 序列标注训练，不修改上级 `src/` 代码。

## 数据集

- 路径：`../data/peoples_daily/`
- 格式：`{"tokens": [...], "ner_tags": ["O", "B-PER", ...]}`
- 实体类型：PER（人名）、ORG（机构）、LOC（地名），共 7 个 BIO 标签
- 规模：train 20864 / validation 2318 / test 4636（测试集有标签）

## 目录结构

```
work7/
├── dataset.py          # PeoplesDailyDataset + DataLoader
├── model.py            # BERT + Linear / CRF
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── compare_results.py  # Linear vs CRF 对比
├── outputs/
│   ├── checkpoints/    # best_linear.pt / best_crf.pt
│   └── logs/           # 训练与评估 JSON
└── README.md
```

## 运行步骤

```bash
cd work7

# 1. 训练 BERT + Linear
python train.py

# 2. 训练 BERT + CRF（可选对比）
python train.py --use_crf

# 3. 在测试集评估
python evaluate.py --split test
python evaluate.py --use_crf --split test

# 4. 汇总对比
python compare_results.py
```

显存不足时可加 `--batch_size 16`。

## 与 cluener 的区别

| 项目 | cluener（src/） | peoples_daily（work7/） |
|------|----------------|------------------------|
| 标注格式 | span（text + label 字典） | 已是 BIO（tokens + ner_tags） |
| 实体类型 | 10 类 | 3 类 |
| 标签数 | 21 | 7 |
| 测试集标签 | 无（只能用 validation） | 有（可直接算 test F1） |
