# BQ Corpus 文本匹配训练实验报告

## 1. 实验背景

本实验将 BiEncoder 文本匹配训练数据集切换为 BQ Corpus，并沿用适配 Mac 笔记本的轻量训练配置。实验机器为 Apple Silicon Mac，内存 16GB；当前 `deep_learn` 环境中 MPS 实际不可用，因此本次训练使用 CPU 完成。

## 2. 数据集

本次使用 BQ Corpus 数据集：

| 划分 | 样本数 |
| --- | ---: |
| train | 68,960 |
| validation | 8,620 |
| test | 8,620 |

最终指标来自 validation 集。

## 3. 训练方式

本次训练使用前台 macOS Terminal 展示实时进度，便于直接观察 `tqdm` 训练条、loss 和预计剩余时间。

训练命令如下：

```bash
MPLCONFIGDIR=/Users/songqingbin/PycharmProjects/hub-kzSW/宋庆彬/week08/outputs/mpl_cache \
/Users/songqingbin/anaconda3/envs/deep_learn/bin/python src/train_biencoder.py \
  --bert_path /Users/songqingbin/PycharmProjects/rag_learn/integrated_qa_system/rag_qa/models/bert-base-chinese \
  --data_dir data/bq_corpus \
  --loss cosine \
  --epochs 1 \
  --num_hidden_layers 2 \
  --max_length 32 \
  --batch_size 16
```

核心配置：

| 参数 | 值 |
| --- | --- |
| 模型结构 | BiEncoder |
| 损失函数 | CosineEmbeddingLoss |
| Pooling | mean |
| BERT 层数 | 2 |
| Epoch | 1 |
| Batch size | 16 |
| 单句最大长度 | 32 |
| 训练设备 | CPU |
| 训练 batch 数 | 4,310 |
| 验证 batch 数 | 539 |

## 4. 实验结果

| 指标 | 结果 |
| --- | ---: |
| train_loss | 0.2372 |
| val_acc | 0.7829 |
| val_f1 | 0.7826 |
| 最优阈值 | 0.69 |
| 总耗时 | 407.40 秒 |

## 5. 结论

BQ Corpus 数据规模小于 LCQMC，在相同的 Mac 轻量配置下训练耗时明显更短。本次全量一轮训练完成后，验证集 F1 达到 0.7826，说明该配置可以在本机 CPU 环境下较快完成 BQ Corpus 的可复现实验。
