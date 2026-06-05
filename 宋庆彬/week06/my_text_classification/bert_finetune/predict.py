"""BERT 单条 / 批量推理"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from common.config import DATA_DIR, CKPT_DIR, BERT_MODEL_NAME
from common.utils import get_device
from model import build_model


def predict_single(text, model, tokenizer, id2name, max_length, device, top_k=3):
    encoding = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        probs = F.softmax(model(input_ids, attention_mask, token_type_ids), dim=-1).squeeze(0)

    top_probs, top_ids = probs.topk(min(top_k, len(probs)))
    return [
        {"label": id2name[int(lid)], "prob": f"{float(p):.4f}"}
        for lid, p in zip(top_ids, top_probs)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    device = get_device()

    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

    model = build_model(BERT_MODEL_NAME, len(id2name), pool=args.pool)
    ckpt = torch.load(CKPT_DIR / f"best_{args.pool}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    if args.text:
        results = predict_single(args.text, model, tokenizer, id2name, args.max_length, device, args.top_k)
        print(f"\n文本: {args.text}")
        for i, r in enumerate(results):
            print(f"  Top-{i+1}: {r['label']} ({r['prob']})")
    else:
        examples = [
            "苹果发布新款iPhone，搭载A19芯片",
            "今日A股全线下跌，沪指跌幅超2%",
            "梅西世界波破门，阿根廷晋级决赛",
            "教育部出台新政，减轻学生课业负担",
        ]
        print("示例推理（无 --text 时的演示）：")
        for text in examples:
            r = predict_single(text, model, tokenizer, id2name, args.max_length, device, 3)
            print(f"\n  {text[:40]}")
            for rr in r:
                print(f"    {rr['label']:4s}  {rr['prob']}")


if __name__ == "__main__":
    main()
