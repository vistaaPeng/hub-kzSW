"""work8 文本匹配模型：BiEncoder / CrossEncoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel


class BiEncoder(nn.Module):
    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        super().__init__()
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.pool = pool
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)
        vec = self.dropout(vec)
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, batch_a, batch_b):
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()
        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        masked = last_hidden + (1 - mask) * (-1e9)
        return masked.max(dim=1).values


class CrossEncoder(nn.Module):
    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=None):
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls_vec)


def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    model = BiEncoder(bert_path, pool=pool, dropout=dropout, num_hidden_layers=num_hidden_layers)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")
    return model


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    model = CrossEncoder(bert_path, dropout=dropout, num_hidden_layers=num_hidden_layers)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: CrossEncoder (layers={num_hidden_layers or 12})")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")
    return model
