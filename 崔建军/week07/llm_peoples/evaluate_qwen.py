"""
在测试集上评估 Qwen2.5 + LoRA NER 模型，并统计非法序列

与 BERT 评估脚本的功能完全一致，但使用 Qwen2.5 + LoRA 模型：
  - 3 类实体（PER/ORG/LOC）
  - 数据格式为 peoples_daily 分词 token 列表 + BIO 标签列表

教学重点：
  1. seqeval 的 entity-level 评估
  2. 非法序列统计
  3. 逐类型 F1 分析

使用方式：
  python evaluate_qwen.py                        # 在测试集上评估
  python evaluate_qwen.py --split validation     # 在验证集上评估
  python evaluate_qwen.py --batch_size 8        # 自定义批次大小
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path

import torch
from transformers import Qwen2Tokenizer
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)

from dataset_qwen import build_label_schema, build_dataloaders, MODEL_NAME
from model_qwen import build_model


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints_qwen"
LOG_DIR = ROOT / "outputs" / "logs_qwen"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计非法 BIO 序列数量。

    非法类型：
      - illegal_start：序列以 I-X 开头
      - illegal_transition：B-X 或 I-X 后面跟 I-Y（X≠Y）
    """
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

    total_illegal = stats["illegal_start"] + stats["illegal_transition"]
    stats["total_illegal"] = total_illegal
    return stats


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (pred_seqs, gold_seqs)。"""
    model.eval()
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().tolist()

            logits, _ = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []

                for j, gold_id in enumerate(labels[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if j < len(preds[i]):
                        pred_seq.append(id2label.get(preds[i][j], "O"))
                    else:
                        pred_seq.append("O")

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")
    print(f"实体类型：{labels[1:]}")

    print(f"\n正在加载 Qwen2.5-0.5B-Instruct 分词器...")
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"\n正在加载模型...")
    model = build_model(
        use_lora=True,
        model_name=MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    ckpt_path = CKPT_DIR / "best_qwen_lora_peoples.pt"
    if not ckpt_path.exists():
        print(f"错误：找不到 checkpoint 文件 {ckpt_path}")
        print("请先运行训练脚本：python train_qwen.py")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )
    print(f"数据集规模：训练={len(train_loader.dataset)}，验证={len(val_loader.dataset)}，测试={len(test_loader.dataset)}")

    print(f"\n正在在 [{args.split}] 集上推理...")

    if args.split == "validation":
        loader = val_loader
    elif args.split == "train":
        loader = train_loader
    else:
        loader = test_loader

    pred_seqs, gold_seqs = run_inference(model, loader, id2label, device)

    entity_precision = precision_score(pred_seqs, gold_seqs)
    entity_recall = recall_score(pred_seqs, gold_seqs)
    entity_f1 = f1_score(pred_seqs, gold_seqs)

    illegal_stats = count_illegal_sequences(pred_seqs)

    print("=" * 70)
    print(f"模型：Qwen2.5-0.5B + LoRA  |  评估集：{args.split}")
    print("=" * 70)
    print(f"Entity-level Precision: {entity_precision:.4f}")
    print(f"Entity-level Recall:    {entity_recall:.4f}")
    print(f"Entity-level F1:        {entity_f1:.4f}")

    entity_types = list(set(et for seq in gold_seqs for et in seq if et != "O"))
    entity_types = [et[2:] if et.startswith("B-") or et.startswith("I-") else et for et in entity_types]
    entity_types = sorted(set(entity_types))

    print("\n【逐类型 F1】")
    report = seqeval_report(gold_seqs, pred_seqs, digits=4, zero_division=0)
    lines = report.split("\n")
    for line in lines:
        if line.strip() and not line.startswith(" ") and not line.startswith("support"):
            print(f"  {line}")

    print("\n【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    if illegal_stats['total_seqs'] > 0:
        illegal_pct = 100 * illegal_stats['total_illegal'] / illegal_stats['total_seqs']
        print(f"  → 非法序列占比：{illegal_pct:.1f}%")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    eval_log_path = LOG_DIR / f"eval_qwen_lora_peoples_{args.split}.json"
    eval_result = {
        "split": args.split,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "illegal_stats": illegal_stats,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_f1": ckpt["val_entity_f1"],
    }
    with open(eval_log_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {eval_log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="评估 Qwen2.5 + LoRA NER 模型（peoples_daily 数据集）")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    main()
