完整参数：
python train_sft.py \
  --num_train 5000    \  # 训练样本数，-1 用全部
  --epochs 3          \  # 训练轮数
  --batch_size 4      \  # 每步 batch 大小
  --grad_accum 4      \  # 梯度累积，等效 batch = 4×4 = 16
  --lr 2e-4           \  # 学习率；不填则自动选：LoRA=2e-4，全量=2e-5
  --lora_r 8          \  # LoRA rank（仅 LoRA 模式有效）
  --lora_alpha 16     \  # 缩放因子（仅 LoRA 模式有效）
  --full_ft              # 加此 flag 切换为全量微调

# lora微调：
python train_sft.py \
  --num_train 5000    \
  --epochs 3          \
  --batch_size 4      \
  --grad_accum 4      \
  --lr 2e-4           \
  --lora_r 8          \
  --lora_alpha 16
日志：
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
总训练步数: 937（batch=4, grad_accum=4, epochs=3, lr=0.0002）

Epoch 1/3 | train_loss=0.7318  val_loss=0.6580 | 135s                                                                   
  ✓ 最优LoRA adapter已保存 → /root/text_classification/outputs/sft_adapter  (val_loss=0.6580)
Epoch 2/3 | train_loss=0.6111  val_loss=0.6359 | 122s                                                                   
  ✓ 最优LoRA adapter已保存 → /root/text_classification/outputs/sft_adapter  (val_loss=0.6359)
Epoch 3/3 | train_loss=0.5176  val_loss=0.6497 | 137s                                                                   

训练完成。最优 val_loss=0.6359
训练日志 → /root/text_classification/outputs/train_log_sft.json
LoRA adapter → /root/text_classification/outputs/sft_adapter

下一步：运行 evaluate_sft.py 查看分类准确率与三方对比
Epoch 3/3 | train_loss=0.5176  val_loss=0.6497 | 137s                                                                   

训练完成。最优 val_loss=0.6359
训练日志 → /root/text_classification/outputs/train_log_sft.json
LoRA adapter → /root/text_classification/outputs/sft_adapter

下一步：运行 evaluate_sft.py 查看分类准确率与三方对比

500条样本的准确率：
python evaluate_sft.py --num_samples 500
============================================================
LLM SFT 分类结果
============================================================
  样本数    : 500
  准确率    : 284/500 = 0.5680
  无法解析  : 2 条 (0.4%)
  总耗时    : 22.1s，均值 0.04s/条

三方对比（val 集随机采样，seed=42）
  ┌──────────────────────────────────────────┬──────────┐
  │ 方法                                     │ 准确率   │
  ├──────────────────────────────────────────┼──────────┤
  │ BERT fine-tune（全部 53K 条，3 epochs）   │ ~0.57~62 │
  │ Qwen2-0.5B zero-shot                     │ 0.3300（200 条） │
  │ Qwen2-0.5B SFT（LoRA，500 条样本）    │ 0.5680   │
  └──────────────────────────────────────────┴──────────┘

思考题：
  1. SFT 相比 zero-shot 提升了多少？这符合你的预期吗？
  2. BERT 用了全部 53K 条，SFT 只用了 5K 条；如果数据量相同，谁更有优势？
  3. LoRA 参数量仅约 0.5%，效果损失有多大？
     对比实验：train_sft.py --lora_r 32，或换回全量微调（去掉 peft）。
  4. 生成式分类有 "无法解析" 的情况，判别式分类（BERT）没有。
     在生产系统中，这个差异如何处理？

结果已保存 → /root/text_classification/outputs/llm_sft_results.json

200条样本的准确率：
python evaluate_sft.py --num_samples 200
============================================================
LLM SFT 分类结果
============================================================
  样本数    : 200
  准确率    : 114/200 = 0.5700
  无法解析  : 2 条 (1.0%)
  总耗时    : 9.4s，均值 0.05s/条

三方对比（val 集随机采样，seed=42）
  ┌──────────────────────────────────────────┬──────────┐
  │ 方法                                     │ 准确率   │
  ├──────────────────────────────────────────┼──────────┤
  │ BERT fine-tune（全部 53K 条，3 epochs）   │ ~0.57~62 │
  │ Qwen2-0.5B zero-shot                     │ 0.3300（200 条） │
  │ Qwen2-0.5B SFT（LoRA，200 条样本）    │ 0.5700   │
  └──────────────────────────────────────────┴──────────┘

结果已保存 → /root/text_classification/outputs/llm_sft_results.json

思考题：
1. SFT 相比 zero-shot 提升了多少？这符合你的预期吗？提升了20%，符合预期。
2. BERT 用了全部 53K 条，SFT 只用了 5K 条；如果数据量相同，谁更有优势？SFT，因为SFT能够学到更多的语义特征，处理复杂语义文本的能力更强。
3. LoRA 参数量仅约 0.5%，效果损失有多大？
对比实验：train_sft.py --lora_r 32，或换回全量微调（去掉 peft）。
4. 生成式分类有 "无法解析" 的情况，判别式分类（BERT）没有。
在生产系统中，这个差异如何处理？

# 全量微调：
python train_sft.py \
  --num_train 5000    \
  --epochs 3          \
  --batch_size 4      \
  --grad_accum 4      \
  --lr 2e-5           \
  --full_ft
日志：
用设备: cuda  |  微调模式: 全量微调（Full Fine-Tuning）
训练集: 5000 条 | 验证集（前500条）: 500 条

加载 tokenizer: /root/autodl-tmp/pretrain_models/Qwen2-0.5B-Instruct
加载 base model: /root/autodl-tmp/pretrain_models/Qwen2-0.5B-Instruct
Loading weights: 100%|███████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 700.71it/s]
trainable params: 494,032,768 || all params: 494,032,768 || trainable%: 100.0000
总训练步数: 937（batch=4, grad_accum=4, epochs=3, lr=2e-05）

Epoch 1/3 | train_loss=0.7935  val_loss=0.6805 | 131s                                                                   
Writing model shards: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.82s/it]
  ✓ 最优完整模型已保存 → /root/text_classification/outputs/sft_full_ckpt  (val_loss=0.6805)
Epoch 2/3 | train_loss=0.3338  val_loss=0.8976 | 131s                                                                   
Epoch 3/3 | train_loss=0.1195  val_loss=1.1207 | 132s                                                                   

训练完成。最优 val_loss=0.6805
训练日志 → /root/text_classification/outputs/train_log_full_ft.json
完整模型 → /root/text_classification/outputs/sft_full_ckpt

下一步：运行 evaluate_sft.py 查看分类准确率与三方对比

200条样本的准确率：
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt --num_samples 200
============================================================
LLM SFT 分类结果
============================================================
  样本数    : 200
  准确率    : 110/200 = 0.5500
  无法解析  : 0 条 (0.0%)
  总耗时    : 8.5s，均值 0.04s/条

三方对比（val 集随机采样，seed=42）
  ┌──────────────────────────────────────────┬──────────┐
  │ 方法                                     │ 准确率   │
  ├──────────────────────────────────────────┼──────────┤
  │ BERT fine-tune（全部 53K 条，3 epochs）   │ ~0.57~62 │
  │ Qwen2-0.5B zero-shot                     │ 0.3300（200 条） │
  │ Qwen2-0.5B SFT（全量，200 条样本）    │ 0.5500   │
  └──────────────────────────────────────────┴──────────┘


全量微调准确率未必比LoRA效果好。

