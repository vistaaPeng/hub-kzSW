import torch
import torch.nn as nn

from transformers import BertModel

try:
    from torchcrf import CRF
except ImportError:
    CRF = None


# =====================================================
# Base
# =====================================================

class BaseNERModel(nn.Module):

    def print_model_info(self):

        total_params = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        frozen_params = (
            total_params - trainable_params
        )

        model_size_mb = (
            total_params * 4
        ) / (1024 ** 2)

        print("\n" + "=" * 70)
        print("NER MODEL INFORMATION")
        print("=" * 70)

        print(
            f"Model Name        : {self.model_name}"
        )

        print(
            f"Backbone          : {self.bert.name_or_path}"
        )

        print(
            f"Hidden Size       : {self.bert.config.hidden_size}"
        )

        print(
            f"Num Labels        : {self.num_labels}"
        )

        print(
            f"Dropout           : {self.dropout.p}"
        )

        print(
            f"CRF Enabled       : {self.use_crf}"
        )

        print("-" * 70)

        print(
            f"Total Params      : {total_params:,}"
        )

        print(
            f"Trainable Params  : {trainable_params:,}"
        )

        print(
            f"Frozen Params     : {frozen_params:,}"
        )

        print(
            f"Estimated Size    : {model_size_mb:.2f} MB"
        )

        print("=" * 70)

    def summary(self):

        print("\nMODEL LAYERS")
        print("-" * 70)

        for name, module in self.named_children():

            params = sum(
                p.numel()
                for p in module.parameters()
            )

            print(
                f"{name:<20} {params:>15,}"
            )

        print("-" * 70)

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        print(
            f"{'Total':<20} {total:>15,}"
        )


# =====================================================
# Bert + Linear
# =====================================================

class BertNER(BaseNERModel):

    def __init__(
        self,
        bert_model_path,
        num_labels,
        dropout_prob=0.1
    ):
        super().__init__()

        self.model_name = "BERT-NER"

        self.use_crf = False

        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(
            bert_model_path,
            return_dict=True
        )

        self.dropout = nn.Dropout(
            dropout_prob
        )

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )

        self.print_model_info()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(
            sequence_output
        )

        logits = self.classifier(
            sequence_output
        )

        if labels is not None:

            loss = self.loss_fn(
                logits.view(
                    -1,
                    self.num_labels
                ),
                labels.view(-1)
            )

            return {
                "loss": loss,
                "logits": logits
            }

        preds = torch.argmax(
            logits,
            dim=-1
        )

        return {
            "preds": preds,
            "logits": logits
        }


# =====================================================
# Bert + CRF
# =====================================================

class BertCRFNER(BaseNERModel):

    def __init__(
        self,
        bert_model_path,
        num_labels,
        dropout_prob=0.1
    ):
        super().__init__()

        if CRF is None:
            raise ImportError(
                "请安装 pytorch-crf\n"
                "pip install pytorch-crf"
            )

        self.model_name = "BERT-CRF"

        self.use_crf = True

        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(
            bert_model_path,
            return_dict=True
        )

        self.dropout = nn.Dropout(
            dropout_prob
        )

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )

        self.crf = CRF(
            num_labels,
            batch_first=True
        )

        self.print_model_info()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(
            sequence_output
        )

        emissions = self.classifier(
            sequence_output
        )

        mask = attention_mask.bool()

        if labels is not None:

            labels = labels.clone()

            labels[labels == -100] = 0

            loss = -self.crf(
                emissions,
                labels,
                mask=mask,
                reduction="mean"
            )

            return {
                "loss": loss,
                "emissions": emissions
            }

        preds = self.crf.decode(
            emissions,
            mask=mask
        )

        return {
            "preds": preds,
            "emissions": emissions
        }


# =====================================================
# Factory
# =====================================================

def build_model(
    bert_model_path,
    num_labels,
    use_crf=False,
    dropout=0.1
):

    if use_crf:

        return BertCRFNER(
            bert_model_path=bert_model_path,
            num_labels=num_labels,
            dropout_prob=dropout
        )

    return BertNER(
        bert_model_path=bert_model_path,
        num_labels=num_labels,
        dropout_prob=dropout
    )


if __name__ == "__main__":

    model = build_model(
        bert_model_path="./pretrained_models/bert-base-chinese",
        num_labels=7,
        use_crf=False
    )

    model.summary()
