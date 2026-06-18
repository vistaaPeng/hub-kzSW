"""
模型评估：实体级 Precision/Recall/F1，支持 CRF。
"""

import argparse
import json
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import BertTokenizer

from dataset import build_dataloaders
from model import build_model

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "ner"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs_ner" / "checkpoints"
OUTPUT_DIR = ROOT / "outputs_ner"


def evaluate_entity(model, loader, id2label, device, use_crf, print_report=False):
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model(input_ids, attention_mask, token_type_ids)
                # pred_ids_list is list of list of ints
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                for j in range(len(labels[i])):
                    if labels[i][j].item() != -100:
                        gold_seq.append(id2label[labels[i][j].item()])
                        if use_crf:
                            pred_id = pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0
                        else:
                            pred_id = pred_ids_list[i][j]
                        pred_seq.append(id2label.get(pred_id, "O"))
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    f1 = f1_score(all_golds, all_preds)
    prec = precision_score(all_golds, all_preds)
    rec = recall_score(all_golds, all_preds)
    if print_report:
        print("\n实体级分类报告：")
        print(classification_report(all_golds, all_preds))
    return f1, {"precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="checkpoint 路径，如 outputs_ner/checkpoints/best_crf.pt")
    parser.add_argument("--bert_path", default=str(BERT_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    use_crf = ckpt.get("use_crf", False)
    id2label = ckpt["id2label"]
    num_labels = len(id2label)

    # DataLoader
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    _, val_loader = build_dataloaders(
        data_dir, tokenizer, batch_size=args.batch_size,
        max_length=args.max_length
    )

    # 模型
    model = build_model(args.bert_path, num_labels, use_crf=use_crf)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    f1, metrics = evaluate_entity(model, val_loader, id2label, device, use_crf, print_report=True)
    print(f"\n实体级评估结果：Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()