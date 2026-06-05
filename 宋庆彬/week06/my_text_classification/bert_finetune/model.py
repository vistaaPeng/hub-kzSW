"""BERT 文本分类模型：BertModel + 自定义分类头"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import BertModel

POOL_OPTIONS = ("cls", "mean", "max")


class BertClassifier(nn.Module):
    """BertModel → Pooling → Dropout → Linear"""

    def __init__(self, bert_model_name: str, num_labels: int, pool: str = "cls", dropout: float = 0.1):
        super().__init__()
        assert pool in POOL_OPTIONS, f"pool 必须是 {POOL_OPTIONS}"
        self.pool = pool
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        vec = self._pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(vec))

    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        if self.pool == "mean":
            return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.pool == "max":
            return (last_hidden + (1 - mask) * (-1e9)).max(dim=1).values
        raise ValueError(f"未知池化策略: {self.pool}")


def build_model(bert_model_name: str, num_labels: int, pool: str = "cls") -> BertClassifier:
    model = BertClassifier(bert_model_name, num_labels, pool=pool)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_head  = sum(p.numel() for p in model.classifier.parameters()) / 1e3
    print(f"模型: {n_total:.1f}M 参数 | 池化: {pool} | 分类头: {n_head:.1f}K")
    return model
