"""
peoples_daily NER 模型评估脚本（带 CRF BIO 硬约束）

与 evaluate_practice.py 的区别：
  对 CRF 模型在推理阶段施加 BIO 硬约束，将非法转移分数设为 -1e9，
  使 Viterbi 解码永远不会输出非法序列（如 O→I-X、B-PER→I-ORG）。

使用方式：
  python src/evaluate_practice_new.py                        # 评估 BERT+Linear
  python src/evaluate_practice_new.py --use_crf              # 评估 BERT+CRF（带硬约束）
  python src/evaluate_practice_new.py --use_crf --split test

依赖：
  pip install torch transformers seqeval tqdm
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)

from model import build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent.parent.parent / "pretrain_models" / "bert-base-chinese" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"

ENTITY_TYPES = ["PER", "ORG", "LOC"]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def _apply_bio_constraints(model: nn.Module, label2id: dict, id2label: dict) -> None:
    """对 CRF 层施加 BIO 硬约束，将非法转移分数设为 -1e9（推理阶段，无需重训）。"""
    num_labels = len(label2id)

    with torch.no_grad():
        # 1. 禁止以 I-X 开头
        for i in range(num_labels):
            if id2label[i].startswith("I-"):
                model.crf.start_transitions[i] = -1e9

        # 2. 禁止 O -> I-X（只允许 O -> O 或 O -> B-X）
        o_idx = label2id["O"]
        for i in range(num_labels):
            if id2label[i].startswith("I-"):
                model.crf.transitions[o_idx, i] = -1e9

        # 3. 禁止 B-X -> I-Y (X≠Y) 和 I-X -> I-Y (X≠Y)
        for i in range(num_labels):
            for j in range(num_labels):
                tag_i = id2label[i]
                tag_j = id2label[j]
                if tag_i.startswith(("B-", "I-")) and tag_j.startswith("I-"):
                    type_i = tag_i[2:]
                    type_j = tag_j[2:]
                    if type_i != type_j:
                        model.crf.transitions[i, j] = -1e9


class PeoplesDailyDataset(Dataset):
    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计非法 BIO 序列数量。"""
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev.startswith("B-") or prev.startswith("I-"):
                    prev_type = prev[2:]
                    if prev_type != curr_type:
                        stats["illegal_transition"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (all_preds, all_golds)。"""
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0, "O"))
                    else:
                        pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_peoples_daily_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}")
        print(f"请先运行：python src/train_practice.py {'--use_crf' if args.use_crf else ''}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels, label2id, id2label = build_label_schema()

    bert_path = str(args.bert_path.resolve()) if args.bert_path.exists() else "bert-base-chinese"
    if not args.bert_path.exists():
        print(f"本地模型路径不存在：{args.bert_path}")
        print(f"将自动从 HuggingFace Hub 下载：bert-base-chinese")

    model = build_model(
        use_crf=args.use_crf,
        bert_path=bert_path,
        num_labels=len(labels),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

    # ==================== CRF BIO 硬约束 ====================
    if args.use_crf:
        _apply_bio_constraints(model, label2id, id2label)
        print("已施加 BIO 硬约束：非法转移分数已设为 -1e9")
    # ======================================================

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=ckpt["args"].get("max_length", 128),
        data_dir=DATA_DIR,
    )
    loader = val_loader if args.split == "validation" else test_loader
    split_name = args.split

    print(f"\n正在在 [{split_name}] 集上推理...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    # seqeval entity-level 指标
    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}  |  评估集：{split_name}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))

    # 非法序列统计
    illegal_stats = count_illegal_sequences(all_preds)
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
    if args.use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  -> CRF Viterbi 解码：非法序列 0 条 (硬约束生效)")
        else:
            print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
            print(f"  → 提示：硬约束已施加，若仍出现非法序列请检查约束逻辑")
    else:
        print(f"  → 线性头约 {pct:.1f}% 的序列含非法转移，CRF+硬约束可完全消除")

    # 保存结果 JSON
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": split_name,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
    }
    out_path = LOG_DIR / f"eval_peoples_daily_{run_tag}_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="评估 peoples_daily BERT NER 模型（带 CRF BIO 硬约束）")
    parser.add_argument("--use_crf", action="store_true", help="评估 CRF 模型（否则评估 Linear）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", choices=["validation", "test"], default="test",
                        help="评估数据集（默认 test）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
