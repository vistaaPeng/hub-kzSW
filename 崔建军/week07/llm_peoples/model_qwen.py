"""
Qwen2.5-0.5B-Instruct + LoRA NER 模型

基于 Qwen2.5-0.5B-Instruct 基座模型，使用 LoRA 进行轻量级微调

教学重点：
  1. LoRA 原理：只在 Attention 的 Q/V 层注入低秩矩阵，大幅减少可训练参数
  2. Qwen2.5 的差异：使用 Qwen2Model，需要配置 tie_word_embeddings=False
  3. 标签预测：提取 Qwen 最后一层 hidden state，添加分类头预测 BIO 标签

依赖：
  pip install peft transformers torch

使用方式：
  from model_qwen import QwenLoraNER, build_model
  model = build_model(use_lora=True, model_name="Qwen/Qwen2.5-0.5B-Instruct", num_labels=7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, Qwen2Config
from peft import LoraConfig, get_peft_model, TaskType


class QwenLoraNER(nn.Module):
    """Qwen2.5 + LoRA + 线性分类头的 NER 模型。

    架构：
      Qwen2.5-ForCausalLM (冻结原模型)
            ↓
      LoRA Q/V 层（可训练）
            ↓
      最后一层 hidden_state (B, L, 896)
            ↓
      Dropout → Linear(896, num_labels)
            ↓
      logits (B, L, num_labels)

    损失：CrossEntropy，ignore_index=-100 跳过特殊token
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        qwen_config = Qwen2Config.from_pretrained(
            model_name,
            trust_remote_code=True,
            tie_word_embeddings=False,
        )

        self.qwen = Qwen2ForCausalLM.from_pretrained(
            model_name,
            config=qwen_config,
            trust_remote_code=True,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters()

        hidden_size = qwen_config.hidden_size
        
        dtype = torch.bfloat16
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        logits = self.classifier(self.dropout(hidden_states))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """生成响应（用于 SFT 评估）。"""
        with torch.no_grad():
            outputs = self.qwen.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                **kwargs,
            )
        return outputs


def build_model(
    use_lora: bool = True,
    model_name: str = r"D:\BaiduNetdiskDownload\AI\pretrain_models\Qwen2.5-0.5B-Instruct",
    num_labels: int = 7,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    dropout: float = 0.1,
) -> QwenLoraNER:
    """构建 Qwen + LoRA NER 模型。"""
    model = QwenLoraNER(
        model_name=model_name,
        num_labels=num_labels,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        dropout=dropout,
    )
    return model
