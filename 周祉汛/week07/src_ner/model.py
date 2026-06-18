"""
BERT NER 模型：支持普通线性分类头和 CRF 层。
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertNER(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1, use_crf=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_out = outputs.last_hidden_state        # [B, L, H]
        seq_out = self.dropout(seq_out)
        logits = self.classifier(seq_out)          # [B, L, num_labels]

        if labels is not None:
            if self.use_crf:
                loss = -self.crf(logits, labels, mask=attention_mask.bool())
                return logits, loss
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return logits, loss
        else:
            if self.use_crf:
                return self.crf.decode(logits, mask=attention_mask.bool())
            else:
                return logits.argmax(dim=-1)

    @property
    def num_labels(self):
        return self.classifier.out_features


def build_model(bert_path, num_labels, use_crf=False, dropout=0.1):
    model = BertNER(bert_path, num_labels, dropout, use_crf)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total:.2f}M | CRF: {use_crf}")
    return model