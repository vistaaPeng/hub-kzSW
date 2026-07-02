# work8 — LCQMC / BQ Corpus 文本匹配

在 **LCQMC** 与 **BQ Corpus** 两个数据集上独立实现 BERT 文本匹配训练，对比三种方法，不修改上级 `src/` 代码。

> 课堂演示已在 `src/` 上用 **AFQMC** 完成；本作业在另外两个数据集上复现实验并撰写分析报告。

## 作业目标

1. 在两个数据集上分别训练三种文本匹配方法
2. 在 **test 集**（有标签）上评估并对比 Accuracy / F1
3. 分析不同方法、不同数据集上的效果差异，写入 `RESULTS.md`

## 三种方法

| 方法 | 脚本 | 说明 |
|------|------|------|
| BiEncoder + CosineEmbeddingLoss | `train_biencoder.py --loss cosine` | 表示型，余弦相似度 + 阈值搜索 |
| BiEncoder + TripletLoss | `train_biencoder.py --loss triplet` | 表示型，三元组相对距离 |
| CrossEncoder + CrossEntropyLoss | `train_crossencoder.py` | 交互型，句对拼接后直接分类 |

## 数据集

| 数据集 | 路径 | 规模（train / val / test） | 领域 |
|--------|------|---------------------------|------|
| LCQMC | `../data/lcqmc/` | 238,766 / 8,802 / 12,500 | 通用问句匹配 |
| BQ Corpus | `../data/bq_corpus/` | 68,960 / 8,620 / 8,620 | 金融客服问句 |

数据格式与 AFQMC 相同：`{"sentence1", "sentence2", "label"}`（0=不相似，1=相似）。

**与 AFQMC 的区别**：LCQMC 和 BQ 的 **test 集有标签**，可直接在 test 上报告最终指标。

## 目录结构

```
work8/
├── dataset.py              # Pair / Triplet / CrossEncoder 数据集
├── model.py                # BiEncoder / CrossEncoder
├── train_biencoder.py      # 表示型训练
├── train_crossencoder.py   # 交互型训练
├── evaluate.py             # test/val 评估
├── compare_results.py      # 汇总对比表
├── run_all.py              # 一键跑全部实验
├── SETUP.md                # 环境准备
├── RESULTS.md              # 实验报告（跑完后填写）
└── outputs/
    ├── lcqmc/
    │   ├── checkpoints/
    │   └── logs/
    ├── bq_corpus/
    │   ├── checkpoints/
    │   └── logs/
    └── comparison_test.json
```

## 推荐运行顺序

### 1. 环境准备

见 [`SETUP.md`](SETUP.md)。

### 2. 建议先跑 BQ Corpus（数据量小，约 1~2 小时/GPU）

```powershell
cd E:\DeepLearning\week7\文本匹配项目\work8

# BiEncoder Cosine
python train_biencoder.py --dataset bq_corpus --loss cosine

# BiEncoder Triplet
python train_biencoder.py --dataset bq_corpus --loss triplet

# CrossEncoder
python train_crossencoder.py --dataset bq_corpus

# test 集评估
python evaluate.py --dataset bq_corpus --model_type biencoder --loss cosine --split test
python evaluate.py --dataset bq_corpus --model_type biencoder --loss triplet --split test
python evaluate.py --dataset bq_corpus --model_type crossencoder --split test
```

### 3. 再跑 LCQMC（训练集约 24 万条，耗时明显更长）

```powershell
python train_biencoder.py --dataset lcqmc --loss cosine
python train_biencoder.py --dataset lcqmc --loss triplet
python train_crossencoder.py --dataset lcqmc

python evaluate.py --dataset lcqmc --model_type biencoder --loss cosine --split test
python evaluate.py --dataset lcqmc --model_type biencoder --loss triplet --split test
python evaluate.py --dataset lcqmc --model_type crossencoder --split test
```

显存不足时加 `--batch_size 16`。

### 4. 汇总对比

```powershell
python compare_results.py --split test
```

### 5. 一键运行（可选）

```powershell
# 只跑 BQ（快速验证流程）
python run_all.py --dataset bq_corpus

# 两个数据集全部实验（LCQMC 需较长时间）
python run_all.py
```

## 默认训练配置

与 `src/` AFQMC 演示保持一致，便于横向对比：

- BERT：bert-base-chinese，**4 层**（`--num_hidden_layers 4`）
- Epochs：**3**
- Batch size：**32**
- BiEncoder max_length：**64**；CrossEncoder max_length：**128**
- 优化器：AdamW，lr=2e-5

## 报告撰写要点（RESULTS.md）

参考上周 work7 的 [`RESULTS.md`](../序列标注项目/work7/RESULTS.md)，建议包含：

1. 两个数据集 × 三种方法的 **test 集** 指标表
2. Cosine vs Triplet：LCQMC 数据量大时 Triplet 是否更有优势？
3. BiEncoder vs CrossEncoder：表示型与交互型的精度/速度权衡
4. BQ（金融）vs LCQMC（通用）：领域差异对模型的影响
5. 可选消融：`--pool cls/mean/max`、`--num_hidden_layers 4/12`

## 与 AFQMC（src/）的对应关系

| 项目 | AFQMC（src/） | work8 |
|------|--------------|-------|
| 用途 | 课堂演示 | 学生作业 |
| 数据集 | 蚂蚁金融问句 | LCQMC + BQ Corpus |
| test 标签 | 无（-1 占位） | **有**，可直接评估 |
| 代码位置 | `src/` | `work8/`（独立，不改 src） |
