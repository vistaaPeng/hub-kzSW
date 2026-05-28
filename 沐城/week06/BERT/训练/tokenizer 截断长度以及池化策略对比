完整参数说明：
python train.py \
  --pool cls          \  # 池化策略：cls / mean / max
  --epochs 3          \  # 训练轮数
  --batch_size 32     \  # batch 大小
  --max_length 128    \  # tokenizer 截断长度（建议 64 即可）
  --lr 2e-5           \  # BERT 层学习率
  --head_lr_mult 5.0  \  # 分类头 lr = lr × 5.0
  --dropout 0.1       \  # 分类头前的 dropout
  --warmup_ratio 0.1  \  # warmup 步数 = 总步数 × 0.1
  --grad_accum 1      \  # 梯度累积（显存不足时设 2 或 4）
  --use_class_weight     # 启用加权 loss


cls

max_length 128
python train.py   --pool cls            --epochs 3            --batch_size 32       --max_length 128      --lr 2e-5             --head_lr_mult 5.0    --dropout 0.1         --warmup_ratio 0.1    --grad_accum 1        --use_class_weight
训练日志：
[
  {
    "epoch": 1,
    "train_loss": 2.741431959529688,
    "train_acc": 0.07260119940029985,
    "val_acc": 0.091,
    "val_macro_f1": 0.01112129544760159,
    "elapsed_s": 195.45318794250488
  },
  {
    "epoch": 2,
    "train_loss": 2.6574357719078234,
    "train_acc": 0.09374062968515742,
    "val_acc": 0.1346,
    "val_macro_f1": 0.06102391777884207,
    "elapsed_s": 195.84259843826294
  },
  {
    "epoch": 3,
    "train_loss": 2.355340906526374,
    "train_acc": 0.16686656671664168,
    "val_acc": 0.2032,
    "val_macro_f1": 0.18151900392026887,
    "elapsed_s": 194.60894966125488
  }
]
loss比较高

max_length 64
python train.py   --pool cls            --epochs 3            --batch_size 32       --max_length 64      --lr 2e-5             --head_lr_mult 5.0    --dropout 0.1         --warmup_ratio 0.1    --grad_accum 1        --use_class_weight
进行数据分析发现大部分文本长度都不超过64
训练日志：
使用设备: cuda
类别数: 15
DataLoader 构建完成
  train: 53360 条, 1668 batch
  val  : 10000 条, 313 batch
  test : 10000 条, 313 batch
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 32894.56it/s]
模型参数量: 109.5M  (BERT: 109.5M, 分类头: 11.5K)
池化策略: cls
类别权重（用于加权 loss）：
   0 故事  : 3.202
   1 文化  : 0.872
   2 娱乐  : 0.715
   3 体育  : 0.891
   4 财经  : 0.684
   5 房产  : 1.688
   6 汽车  : 0.864
   7 教育  : 1.035
   8 科技  : 0.597
   9 军事  : 0.979
  10 旅游  : 1.056
  11 国际  : 0.733
  12 证券  : 13.842
  13 农业  : 1.233
  14 电竞  : 1.049
使用加权 CrossEntropyLoss
总训练步数: 5004, warmup: 500
Epoch 1/3 | train_loss=2.3380 train_acc=0.2165 | val_acc=0.3933 val_macro_f1=0.3805 | 122s                              
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_cls_weighted.pt  (val_acc=0.3933)
Epoch 2/3 | train_loss=1.7040 train_acc=0.4314 | val_acc=0.4529 val_macro_f1=0.4359 | 132s                              
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_cls_weighted.pt  (val_acc=0.4529)
                                                                                                                        
分类报告：
              precision    recall  f1-score   support

          故事       0.28      0.60      0.39       215
          文化       0.49      0.43      0.46       736
          娱乐       0.54      0.50      0.52       910
          体育       0.61      0.62      0.61       767
          财经       0.43      0.35      0.39       956
          房产       0.46      0.60      0.52       378
          汽车       0.61      0.57      0.59       791
          教育       0.47      0.58      0.52       646
          科技       0.52      0.37      0.43      1089
          军事       0.46      0.46      0.46       716
          旅游       0.37      0.48      0.42       693
          国际       0.47      0.40      0.43       905
          证券       0.17      0.58      0.26        45
          农业       0.43      0.47      0.45       494
          电竞       0.57      0.58      0.58       659

    accuracy                           0.48     10000
   macro avg       0.46      0.50      0.47     10000
weighted avg       0.49      0.48      0.48     10000

Epoch 3/3 | train_loss=1.5044 train_acc=0.4888 | val_acc=0.4814 val_macro_f1=0.4679 | 330s
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_cls_weighted.pt  (val_acc=0.4814)

训练完成。最优 val_acc=0.4814
训练日志 → /root/text_classification/outputs/train_log_cls_weighted.json

[
  {
    "epoch": 1,
    "train_loss": 2.3379546342761084,
    "train_acc": 0.21649175412293853,
    "val_acc": 0.3933,
    "val_macro_f1": 0.38050503114553125,
    "elapsed_s": 121.7118091583252
  },
  {
    "epoch": 2,
    "train_loss": 1.704041451957451,
    "train_acc": 0.431428035982009,
    "val_acc": 0.4529,
    "val_macro_f1": 0.43594870099183086,
    "elapsed_s": 132.0661816596985
  },
  {
    "epoch": 3,
    "train_loss": 1.5044338821471184,
    "train_acc": 0.4888305847076462,
    "val_acc": 0.4814,
    "val_macro_f1": 0.46789240193999526,
    "elapsed_s": 330.15163373947144
  }
]
发现合理的tokenizer确实很重要，过长或过短都不行。
过长导致很多无效token参与loss计算，对结果有影响。
过短会截断有效token，也会影响准确率。

mean
python train.py   --pool mean            --epochs 3            --batch_size 32       --max_length 64      --lr 2e-5             --head_lr_mult 5.0    --dropout 0.1         --warmup_ratio 0.1    --grad_accum 1        --use_class_weight
训练日志：
使用设备: cuda
类别数: 15
DataLoader 构建完成
  train: 53360 条, 1668 batch
  val  : 10000 条, 313 batch
  test : 10000 条, 313 batch
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 30522.43it/s]
模型参数量: 109.5M  (BERT: 109.5M, 分类头: 11.5K)
池化策略: mean
类别权重（用于加权 loss）：
   0 故事  : 3.202
   1 文化  : 0.872
   2 娱乐  : 0.715
   3 体育  : 0.891
   4 财经  : 0.684
   5 房产  : 1.688
   6 汽车  : 0.864
   7 教育  : 1.035
   8 科技  : 0.597
   9 军事  : 0.979
  10 旅游  : 1.056
  11 国际  : 0.733
  12 证券  : 13.842
  13 农业  : 1.233
  14 电竞  : 1.049
使用加权 CrossEntropyLoss
总训练步数: 5004, warmup: 500
Epoch 1/3 | train_loss=2.3425 train_acc=0.2058 | val_acc=0.3734 val_macro_f1=0.3626 | 317s                              
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_mean_weighted.pt  (val_acc=0.3734)
Epoch 2/3 | train_loss=1.7131 train_acc=0.4262 | val_acc=0.4671 val_macro_f1=0.4576 | 239s                              
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_mean_weighted.pt  (val_acc=0.4671)
                                                                                                                        
分类报告：
              precision    recall  f1-score   support

          故事       0.32      0.62      0.43       215
          文化       0.44      0.46      0.45       736
          娱乐       0.49      0.48      0.48       910
          体育       0.61      0.61      0.61       767
          财经       0.44      0.32      0.37       956
          房产       0.43      0.66      0.52       378
          汽车       0.62      0.60      0.61       791
          教育       0.53      0.57      0.55       646
          科技       0.49      0.38      0.43      1089
          军事       0.46      0.44      0.45       716
          旅游       0.41      0.38      0.39       693
          国际       0.44      0.45      0.45       905
          证券       0.19      0.51      0.27        45
          农业       0.43      0.46      0.44       494
          电竞       0.56      0.56      0.56       659

    accuracy                           0.48     10000
   macro avg       0.46      0.50      0.47     10000
weighted avg       0.49      0.48      0.48     10000

Epoch 3/3 | train_loss=1.4897 train_acc=0.4920 | val_acc=0.4804 val_macro_f1=0.4683 | 228s
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_mean_weighted.pt  (val_acc=0.4804)

训练完成。最优 val_acc=0.4804
训练日志 → /root/text_classification/outputs/train_log_mean_weighted.json

[
  {
    "epoch": 1,
    "train_loss": 2.342517378555424,
    "train_acc": 0.2057721139430285,
    "val_acc": 0.3734,
    "val_macro_f1": 0.36255205049894745,
    "elapsed_s": 317.4250328540802
  },
  {
    "epoch": 2,
    "train_loss": 1.713132866640677,
    "train_acc": 0.4261619190404798,
    "val_acc": 0.4671,
    "val_macro_f1": 0.4575547639223442,
    "elapsed_s": 239.27007675170898
  },
  {
    "epoch": 3,
    "train_loss": 1.4897112283392109,
    "train_acc": 0.49196026986506747,
    "val_acc": 0.4804,
    "val_macro_f1": 0.46828252286221234,
    "elapsed_s": 227.87852025032043
  }
]

max
python train.py   --pool max            --epochs 3            --batch_size 32       --max_length 64      --lr 2e-5             --head_lr_mult 5.0    --dropout 0.1         --warmup_ratio 0.1    --grad_accum 1        --use_class_weight
训练日志：
使用设备: cuda
类别数: 15
DataLoader 构建完成
  train: 53360 条, 1668 batch
  val  : 10000 条, 313 batch
  test : 10000 条, 313 batch
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 31703.82it/s]
模型参数量: 109.5M  (BERT: 109.5M, 分类头: 11.5K)
池化策略: max
类别权重（用于加权 loss）：
   0 故事  : 3.202
   1 文化  : 0.872
   2 娱乐  : 0.715
   3 体育  : 0.891
   4 财经  : 0.684
   5 房产  : 1.688
   6 汽车  : 0.864
   7 教育  : 1.035
   8 科技  : 0.597
   9 军事  : 0.979
  10 旅游  : 1.056
  11 国际  : 0.733
  12 证券  : 13.842
  13 农业  : 1.233
  14 电竞  : 1.049
使用加权 CrossEntropyLoss
总训练步数: 5004, warmup: 500
Epoch 1/3 | train_loss=2.4340 train_acc=0.1748 | val_acc=0.2840 val_macro_f1=0.2812 | 330s                              
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_max_weighted.pt  (val_acc=0.2840)
Epoch 2/3 | train_loss=1.7971 train_acc=0.3964 | val_acc=0.4249 val_macro_f1=0.4120 | 226s                              
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_max_weighted.pt  (val_acc=0.4249)
                                                                                                                        
分类报告：
              precision    recall  f1-score   support

          故事       0.29      0.63      0.39       215
          文化       0.42      0.44      0.43       736
          娱乐       0.54      0.45      0.49       910
          体育       0.62      0.61      0.61       767
          财经       0.45      0.24      0.31       956
          房产       0.44      0.59      0.51       378
          汽车       0.59      0.58      0.58       791
          教育       0.47      0.59      0.52       646
          科技       0.50      0.37      0.43      1089
          军事       0.37      0.63      0.47       716
          旅游       0.35      0.47      0.40       693
          国际       0.47      0.18      0.26       905
          证券       0.15      0.60      0.24        45
          农业       0.45      0.46      0.45       494
          电竞       0.57      0.54      0.56       659

    accuracy                           0.46     10000
   macro avg       0.44      0.49      0.44     10000
weighted avg       0.48      0.46      0.45     10000

Epoch 3/3 | train_loss=1.5723 train_acc=0.4690 | val_acc=0.4589 val_macro_f1=0.4436 | 207s
  ✓ 新最优模型已保存 → /root/text_classification/outputs/checkpoints/best_max_weighted.pt  (val_acc=0.4589)

训练完成。最优 val_acc=0.4589
训练日志 → /root/text_classification/outputs/train_log_max_weighted.json

[
  {
    "epoch": 1,
    "train_loss": 2.4339720866133248,
    "train_acc": 0.17481259370314842,
    "val_acc": 0.284,
    "val_macro_f1": 0.28119776131739893,
    "elapsed_s": 329.56671118736267
  },
  {
    "epoch": 2,
    "train_loss": 1.7970579309978227,
    "train_acc": 0.39636431784107945,
    "val_acc": 0.4249,
    "val_macro_f1": 0.41197092823138487,
    "elapsed_s": 226.3176872730255
  },
  {
    "epoch": 3,
    "train_loss": 1.5723379428240134,
    "train_acc": 0.4689655172413793,
    "val_acc": 0.4589,
    "val_macro_f1": 0.44360311915577283,
    "elapsed_s": 207.19975876808167
  }
]


对比：
对比不同池化策略：cls / mean / max的效果。
发现 cls 跟 mean 验证集准确率差别不大，cls略高一点点。
max 要比前两者低一点点。
