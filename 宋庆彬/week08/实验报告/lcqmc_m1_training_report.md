# LCQMC 文本匹配训练实验报告

## 1. 实验背景

本实验将原项目中的 BiEncoder 文本匹配训练流程从默认 AFQMC 数据集切换到 LCQMC 数据集，并针对本机 Mac 笔记本环境进行了轻量化适配。实验机器为 Apple Silicon Mac，内存 16GB；当前 `deep_learn` 环境中 PyTorch 能检测到 MPS 构建能力，但实际运行时 `mps` 不可用，因此本次训练使用 CPU 完成。

## 2. 数据集情况

项目中包含三个句对匹配数据集：

| 数据集 | train | validation | test | 说明 |
| --- | ---: | ---: | ---: | --- |
| AFQMC | 34,334 | 4,316 | 3,861 | 项目默认使用的数据集 |
| LCQMC | 238,766 | 8,802 | 12,500 | 本次实验使用的数据集 |
| BQ Corpus | 68,960 | 8,620 | 8,620 | 项目中提供但本次未训练 |

本次训练使用 LCQMC 的完整训练集和完整验证集：

- 训练集：238,766 条
- 验证集：8,802 条
- 测试集：12,500 条，仅构建 DataLoader，未作为最终指标来源

## 3. Mac 笔记本适配方式

为了让 LCQMC 在 16GB Mac 笔记本上能够稳定完成训练，对代码做了以下调整：

1. 在 `src/train_biencoder.py` 中增加 `--device` 参数，支持 `auto/cuda/mps/cpu`，自动选择可用设备。本机最终回退到 CPU。
2. 训练时降低模型和输入规模：使用 2 层 BERT、`max_length=32`、`batch_size=16`，减少 CPU 训练压力和内存占用。

其余数据加载和输出保存逻辑保持项目原有方式。

## 4. 训练配置

训练命令如下：

```bash
MPLCONFIGDIR=/Users/songqingbin/PycharmProjects/hub-kzSW/宋庆彬/week08/outputs/mpl_cache \
/Users/songqingbin/anaconda3/envs/deep_learn/bin/python src/train_biencoder.py \
  --bert_path /Users/songqingbin/PycharmProjects/rag_learn/integrated_qa_system/rag_qa/models/bert-base-chinese \
  --data_dir data/lcqmc \
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
| 预训练模型 | bert-base-chinese |
| 损失函数 | CosineEmbeddingLoss |
| Pooling | mean |
| BERT 层数 | 2 |
| Epoch | 1 |
| Batch size | 16 |
| 单句最大长度 | 32 |
| 训练设备 | CPU |
| 模型参数量 | 31.4M |
| 训练 batch 数 | 14,923 |
| 验证 batch 数 | 551 |

## 5. 实验结果

本次全量 LCQMC 一轮训练结果如下：

| 指标 | 结果 |
| --- | ---: |
| train_loss | 0.2007 |
| val_acc | 0.7261 |
| val_f1 | 0.7255 |
| 最优阈值 | 0.84 |
| 总耗时 | 4,038.47 秒 |

训练过程中 loss 从约 0.31 逐步下降到 0.2007，说明模型在完整 LCQMC 训练集上完成了有效学习。验证集 F1 达到 0.7255，作为 CPU 环境下的 2 层轻量 BERT 配置，结果具备可复现实验价值。

## 6. 结论

本次实验完成了从默认 AFQMC 到 LCQMC 的训练切换，并在 Mac 笔记本环境下完成 LCQMC 全量一轮训练。通过降低 BERT 层数、缩短输入长度、减小 batch size，并增加设备自动选择参数，训练流程能够在本机 CPU 环境下稳定运行。

如果后续要进一步提升指标，可以尝试增加 BERT 层数、训练更多 epoch，或在可用 MPS/CUDA 的环境中恢复更大的 batch size 和输入长度。
