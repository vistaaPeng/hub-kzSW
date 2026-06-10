"""
peoples_daily NER 训练脚本（支持从已有 checkpoint 继续训练）

使用方式：
  # 从头训练
  python src/train_practice_new.py

  # 从已有 checkpoint 继续训练（默认再训到 epoch 5）
  python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_linear.pt --epochs 5
  python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_crf.pt --epochs 5 --use_crf

  # 继续训练时降低学习率（推荐，防止破坏已收敛参数）
  python src/train_practice_new.py --resume outputs/checkpoints/best_peoples_daily_linear.pt --epochs 5 --lr 5e-6

教学重点：
  1. 模型恢复：加载 state_dict + 标签映射 + 历史最佳 F1
  2. 日志追加：读取已有日志，在新 epoch 后追加记录
  3. 学习率调整：继续训练时建议降低 lr（默认 resume_lr_mult=0.1）
  4. Warmup 重置：为新训练阶段重新计算 scheduler
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

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
    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()
            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i]
                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        if j < len(pred_ids):
                            pred_seq.append(id2label.get(pred_ids[j], "O"))
                        else:
                            pred_seq.append("O")
                    else:
                        pred_seq.append(id2label.get(pred_ids[j], "O"))
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

    # Tokenizer
    bert_path = str(args.bert_path.resolve()) if args.bert_path.exists() else "bert-base-chinese"
    if not args.bert_path.exists():
        print(f"本地模型路径不存在：{args.bert_path}")
        print(f"将自动从 HuggingFace Hub 下载：bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # DataLoader
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )

    # 模型
    model = build_model(
        use_crf=args.use_crf,
        bert_path=bert_path,
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    # ==================== Resume 逻辑 ====================
    start_epoch = 1
    best_f1 = 0.0
    log_records = []
    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_peoples_daily_{run_tag}.pt"
    log_path = LOG_DIR / f"train_peoples_daily_{run_tag}.json"

    if args.resume and args.resume.exists():
        print(f"\n🔄 从 checkpoint 恢复训练：{args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # 1. 加载模型权重
        model.load_state_dict(ckpt["state_dict"])
        print(f"  ✓ 模型权重已加载")

        # 2. 校验关键参数一致性
        ckpt_use_crf = ckpt.get("use_crf", False)
        if ckpt_use_crf != args.use_crf:
            raise ValueError(
                f"checkpoint 的 use_crf={ckpt_use_crf} 与当前 args.use_crf={args.use_crf} 不一致"
            )
        ckpt_label2id = ckpt.get("label2id", {})
        if ckpt_label2id != label2id:
            print("  ⚠ 警告：checkpoint 中的标签体系与当前配置不一致")

        # 3. 恢复历史最佳 F1 和起始 epoch
        best_f1 = ckpt.get("val_entity_f1", 0.0)
        trained_epochs = ckpt.get("epoch", 0)
        start_epoch = trained_epochs + 1
        print(f"  ✓ 已训练 epoch：{trained_epochs}，历史最佳 F1：{best_f1:.4f}")

        # 4. 读取已有日志（追加模式）
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                log_records = json.load(f)
            print(f"  ✓ 已加载历史日志，共 {len(log_records)} 条记录")

        print(f"  → 将从 epoch {start_epoch} 继续训练到 epoch {args.epochs}")
    else:
        if args.resume:
            print(f"  ⚠ 指定的 resume 路径不存在：{args.resume}，将从头训练")
        else:
            print("\n🆕 从头开始训练")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== Optimizer & Scheduler ====================
    # 计算实际学习率：若 resume 且用户未显式指定 lr，可自动降低
    effective_lr = args.lr
    if args.resume and args.resume.exists() and args.auto_decay_lr:
        effective_lr = args.lr * args.resume_lr_mult
        print(f"\n📉 自动降低学习率：{args.lr} → {effective_lr}（resume_lr_mult={args.resume_lr_mult}）")
        print(f"   提示：继续训练时降低 lr 可防止破坏已收敛参数")

    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": effective_lr},
            {"params": head_params, "lr": effective_lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    # 本次需要训练的 epoch 数
    remaining_epochs = max(0, args.epochs - start_epoch + 1)
    total_steps = len(train_loader) * remaining_epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n本次训练：epoch {start_epoch} ~ {args.epochs}（共 {remaining_epochs} 轮）")
    print(f"训练步数：{total_steps}，预热步数：{warmup_steps}（warmup_ratio={args.warmup_ratio}）")

    # ==================== 训练循环 ====================
    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型（peoples_daily，支持 resume）")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3, help="目标总 epoch 数（resume 时将从已训 epoch+1 继续到此值）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Resume 相关参数
    parser.add_argument("--resume", type=Path, default=None, help="从已有 checkpoint 继续训练（权重+日志恢复）")
    parser.add_argument("--auto_decay_lr", action="store_true", default=True, help="resume 时自动降低学习率（lr *= resume_lr_mult）")
    parser.add_argument("--resume_lr_mult", type=float, default=0.1, help="resume 时的学习率衰减倍数（默认 0.1）")

    return parser.parse_args()


if __name__ == "__main__":
    main()
