import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchcrf import CRF

class BertNER(nn.Module):
    def __init__(self, bert_model_path, num_labels, use_crf=False):
        super(BertNER, self).__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        output = {}

        if self.use_crf:
            if labels is not None:
                mask = attention_mask.bool()

                # CRF 不允许 labels 中有负值，临时替换
                labels_for_crf = labels.clone()
                labels_for_crf[labels_for_crf == -100] = 0

                # mask 应该排除 -100 的位置
                # mask = mask & (labels != -100)

                # 关键修复：确保每个序列的第一个位置 mask=True
                # CRF 要求每个序列至少有一个有效 token
                # mask[:, 0] = True

                loss = -self.crf(logits, labels_for_crf, mask=mask, reduction='mean')
                output["loss"] = loss

            # 解码时也要确保第一个位置为 True
            decode_mask = attention_mask.bool()
            # decode_mask[:, 0] = True
            preds = self.crf.decode(logits, mask=decode_mask)
            output["preds"] = preds
        else:
            output["preds"] = torch.argmax(logits, dim=-1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                output["loss"] = loss

        return output

def build_model(bert_model_path, num_labels, use_crf=False):
    return BertNER(bert_model_path=bert_model_path, num_labels=num_labels, use_crf=use_crf)
