"""
推理脚本：对单条文本抽取实体，输出 JSON 格式实体列表。
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import BertTokenizer

from model import build_model


def extract_entities(pred_label_ids, tokens, id2label):
    """将 BIO 标签序列转为实体列表（基于 token 级别）"""
    entities = []
    current = None
    for i, label_id in enumerate(pred_label_ids):
        label = id2label.get(label_id, "O")
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": label[2:], "tokens": [tokens[i]]}
        elif label.startswith("I-") and current and current["type"] == label[2:]:
            current["tokens"].append(tokens[i])
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    # 合并 token 为原文子串（注意：如果是 subword，需要还原，这里简单拼接）
    return [{"text": "".join(e["tokens"]), "type": e["type"]} for e in entities]


def predict_ner(text, model, tokenizer, id2label, device, max_length=128, use_crf=False):
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

    with torch.no_grad():
        if use_crf:
            pred_ids = model(input_ids, attention_mask, token_type_ids)
            # pred_ids 是 list of list of ints
            pred_label_ids = pred_ids[0] if pred_ids else []
        else:
            logits = model(input_ids, attention_mask, token_type_ids)
            pred_label_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    # 获取每个 token 对应的原始字符（用于实体提取）
    tokens = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            tokens.append("[SPECIAL]")
        else:
            tokens.append(text[start:end])

    # 过滤掉 special tokens 的预测
    filtered_ids = []
    filtered_tokens = []
    for i, (token, label_id) in enumerate(zip(tokens, pred_label_ids)):
        if token != "[SPECIAL]":
            filtered_ids.append(label_id)
            filtered_tokens.append(token)
    entities = extract_entities(filtered_ids, filtered_tokens, id2label)
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--bert_path", default=str(Path(__file__).parent.parent.parent / "pretrain_models/bert-base-chinese"))
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_length", default=128, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    use_crf = ckpt.get("use_crf", False)
    id2label = ckpt["id2label"]
    num_labels = len(id2label)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = build_model(args.bert_path, num_labels, use_crf=use_crf)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    entities = predict_ner(args.text, model, tokenizer, id2label, device, args.max_length, use_crf)
    print(json.dumps(entities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()