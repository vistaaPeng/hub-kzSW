from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def get_last_hidden_state(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs.last_hidden_state


class BiEncoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)

    def _mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = get_last_hidden_state(outputs)
        pooled = self._mean_pooling(last_hidden_state, attention_mask)
        return self.dropout(pooled)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)
        return emb1, emb2


class CrossEncoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = get_last_hidden_state(outputs)
        pooled = last_hidden_state[:, 0, :]
        logits = self.classifier(pooled).squeeze(-1)
        return logits


class TextMatchingLoss:
    @staticmethod
    def cosine_loss(emb1, emb2, labels):
        target = torch.where(
            labels == 1,
            torch.ones_like(labels, dtype=torch.float32),
            torch.full_like(labels, -1.0, dtype=torch.float32)
        )
        return F.cosine_embedding_loss(emb1, emb2, target)

    @staticmethod
    def triplet_loss(emb1, emb2, labels):
        margin = 0.2
        loss_fn = nn.TripletMarginLoss(margin=margin)
        perm = torch.randperm(emb1.size(0), device=emb1.device)
        negative = emb2[perm]
        return loss_fn(emb1, emb2, negative)

    @staticmethod
    def cross_entropy_loss(logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels.float())
