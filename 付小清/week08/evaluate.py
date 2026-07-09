"""
work8 文本匹配评估

使用方式：
  cd work8
  python evaluate.py --dataset lcqmc --model_type biencoder --loss cosine --split test
  python evaluate.py --dataset bq_corpus --model_type crossencoder --split test
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset, resolve_data_dir
from model import build_biencoder, build_crossencoder

PROJECT_ROOT = Path(__file__).parent.parent
WORK8_ROOT = Path(__file__).parent
BERT_PATH = PROJECT_ROOT.parent / "pretrain_models" / "bert-base-chinese"


@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    model.eval()
    all_sims, all_labels = [], []

    for batch in loader:
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        all_sims.extend(sims)
        all_labels.extend(batch["label"].tolist())

    sims = np.array(all_sims)
    labels = np.array(all_labels)

    if find_threshold:
        threshold = _find_best_threshold(sims, labels)

    preds = (sims >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_pos = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "f1_pos": f1_pos,
        "threshold": threshold,
        "auc": auc,
    }


def _find_best_threshold(sims, labels):
    best_f1, best_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh)


@torch.no_grad()
def eval_crossencoder(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        ).cpu()
        all_logits.extend(logits.tolist())
        all_labels.extend(batch["label"].tolist())

    preds = np.argmax(all_logits, axis=1)
    labels = np.array(all_labels)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_pos = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)

    return {"accuracy": accuracy, "f1": f1, "f1_pos": f1_pos}


def ckpt_dir(dataset: str) -> Path:
    return WORK8_ROOT / "outputs" / dataset / "checkpoints"


def log_dir(dataset: str) -> Path:
    return WORK8_ROOT / "outputs" / dataset / "logs"


def default_ckpt(dataset: str, model_type: str, loss: str | None = None) -> Path:
    if model_type == "biencoder":
        return ckpt_dir(dataset) / f"biencoder_{loss}_best.pt"
    return ckpt_dir(dataset) / "crossencoder_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="work8 文本匹配评估")
    parser.add_argument("--dataset", required=True, choices=["lcqmc", "bq_corpus"])
    parser.add_argument("--model_type", required=True, choices=["biencoder", "crossencoder"])
    parser.add_argument("--loss", default="cosine", choices=["cosine", "triplet"],
                        help="BiEncoder 训练时使用的 loss 类型")
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint 路径（默认按 dataset+方法推断）")
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir(args.dataset)
    ckpt_path = Path(args.ckpt) if args.ckpt else default_ckpt(args.dataset, args.model_type, args.loss)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}，请先运行对应训练脚本")

    print(f"设备: {device}")
    print(f"数据集: {args.dataset}  划分: {args.split}")
    print(f"加载 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    data_path = data_dir / f"{args.split}.jsonl"

    if args.model_type == "biencoder":
        model = build_biencoder(
            bert_path=args.bert_path,
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        ds = PairDataset(data_path, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)
    else:
        model = build_crossencoder(
            bert_path=args.bert_path,
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    print(f"\n{'=' * 55}")
    print(f"{args.dataset} / {args.model_type} / {args.split}（{len(ds)} 条）")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  F1(weighted): {metrics['f1']:.4f}")
    print(f"  F1(正例)    : {metrics['f1_pos']:.4f}")
    if "threshold" in metrics:
        print(f"  阈值        : {metrics['threshold']:.2f}")
        print(f"  AUC         : {metrics['auc']:.4f}")

    out = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "loss": args.loss if args.model_type == "biencoder" else None,
        "split": args.split,
        "num_samples": len(ds),
        **metrics,
    }
    tag = f"eval_{args.model_type}"
    if args.model_type == "biencoder":
        tag = f"eval_biencoder_{args.loss}"
    out_path = log_dir(args.dataset) / f"{tag}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
