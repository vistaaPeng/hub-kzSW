"""
Qwen2.5-0.5B-Instruct + LoRA NER 模块

基于 peoples_daily 数据集，使用 Qwen2.5-0.5B-Instruct 作为基座模型 + LoRA 微调

使用方式：
  python train_qwen.py                           # 训练
  python evaluate_qwen.py                        # 评估

依赖：
  pip install transformers peft torch seqeval tqdm
"""
