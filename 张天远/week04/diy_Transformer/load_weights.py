"""从 HuggingFace BERT state_dict 加载权重到 MyBert"""
import os
os.environ["HF_HOME"] = "M:/huggingface_cache"

import torch
from transformers import BertModel, BertConfig


def load_hf_bert(model_name: str = "bert-base-chinese", device="cpu"):
    """加载 HuggingFace BERT 模型和配置"""
    config = BertConfig.from_pretrained(model_name)
    hf_bert = BertModel.from_pretrained(model_name).to(device)
    hf_bert.eval()
    return hf_bert, config


def transfer_weights(my_bert: torch.nn.Module, hf_state_dict: dict):
    """将 HF state_dict 的权重复制到 MyBert"""
    sd = hf_state_dict

    emb = my_bert.embeddings
    emb.word_embeddings.weight.data.copy_(sd["embeddings.word_embeddings.weight"])
    emb.position_embeddings.weight.data.copy_(sd["embeddings.position_embeddings.weight"])
    emb.token_type_embeddings.weight.data.copy_(sd["embeddings.token_type_embeddings.weight"])
    emb.LayerNorm.weight.data.copy_(sd["embeddings.LayerNorm.weight"])
    emb.LayerNorm.bias.data.copy_(sd["embeddings.LayerNorm.bias"])

    for i in range(len(my_bert.encoder.layer)):
        p = f"encoder.layer.{i}"
        layer = my_bert.encoder.layer[i]
        layer.attention.query.weight.data.copy_(sd[f"{p}.attention.self.query.weight"])
        layer.attention.query.bias.data.copy_(sd[f"{p}.attention.self.query.bias"])
        layer.attention.key.weight.data.copy_(sd[f"{p}.attention.self.key.weight"])
        layer.attention.key.bias.data.copy_(sd[f"{p}.attention.self.key.bias"])
        layer.attention.value.weight.data.copy_(sd[f"{p}.attention.self.value.weight"])
        layer.attention.value.bias.data.copy_(sd[f"{p}.attention.self.value.bias"])
        layer.attention_output.dense.weight.data.copy_(sd[f"{p}.attention.output.dense.weight"])
        layer.attention_output.dense.bias.data.copy_(sd[f"{p}.attention.output.dense.bias"])
        layer.attention_output.LayerNorm.weight.data.copy_(sd[f"{p}.attention.output.LayerNorm.weight"])
        layer.attention_output.LayerNorm.bias.data.copy_(sd[f"{p}.attention.output.LayerNorm.bias"])
        layer.intermediate.dense.weight.data.copy_(sd[f"{p}.intermediate.dense.weight"])
        layer.intermediate.dense.bias.data.copy_(sd[f"{p}.intermediate.dense.bias"])
        layer.output.dense.weight.data.copy_(sd[f"{p}.output.dense.weight"])
        layer.output.dense.bias.data.copy_(sd[f"{p}.output.dense.bias"])
        layer.output.LayerNorm.weight.data.copy_(sd[f"{p}.output.LayerNorm.weight"])
        layer.output.LayerNorm.bias.data.copy_(sd[f"{p}.output.LayerNorm.bias"])

    my_bert.pooler.dense.weight.data.copy_(sd["pooler.dense.weight"])
    my_bert.pooler.dense.bias.data.copy_(sd["pooler.dense.bias"])


def load_weights_from_checkpoint(my_bert, checkpoint_path, map_location="cpu"):
    """从训练 checkpoint 加载微调后的 BERT 权重 + 返回 classifier/dropout 权重"""
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    full_sd = ckpt["model_state_dict"]

    # 提取 BERT 权重（去掉 "bert." 前缀）
    bert_sd = {}
    for k, v in full_sd.items():
        if k.startswith("bert."):
            bert_sd[k[5:]] = v  # strip "bert." prefix

    transfer_weights(my_bert, bert_sd)

    # 提取 classifier
    classifier_weight = full_sd.get("classifier.weight")
    classifier_bias = full_sd.get("classifier.bias")

    return classifier_weight, classifier_bias
