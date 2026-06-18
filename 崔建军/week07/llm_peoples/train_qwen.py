"""
Qwen2.5-0.5B-Instruct + LoRA NER 训练脚本

基于 peoples_daily 数据集，使用 Qwen2.5-0.5B-Instruct 作为基座模型 + LoRA 微调

与 BERT 训练脚本的关键区别：
  1. 模型：Qwen2.5-ForCausalLM + LoRA，不是 BERT + Linear/CRF
  2. 分词器：Qwen2Tokenizer，不是 BertTokenizer
  3. 可训练参数：LoRA 参数（约 0.1%），不是全部参数
  4. 学习率：通常更低（1e-4 ~ 5e-4），因为 LoRA 已经限制了参数空间

教学重点：
  1. LoRA 配置参数：rank、alpha、dropout
  2. 分层学习率：LoRA 层用较高学习率，原模型用很低学习率（冻结）
  3. PEFT 库的使用：get_peft_model、print_trainable_parameters

使用方式：
  python train_qwen.py                           # 默认配置训练
  python train_qwen.py --epochs 5 --lr 1e-4     # 自定义超参数
  python train_qwen.py --batch_size 4            # 更小批次（Qwen 占用更多显存）

依赖：
  pip install transformers peft torch seqeval tqdm
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import Qwen2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_qwen import build_label_schema, build_dataloaders, MODEL_NAME
from model_qwen import build_model


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints_qwen"
LOG_DIR = ROOT / "outputs" / "logs_qwen"


def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (avg_loss, entity_f1)。"""
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().tolist()
            labels_np = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []

                for j, gold_id in enumerate(labels_np[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if j < len(preds[i]):
                        pred_seq.append(id2label.get(preds[i][j], "O"))
                    else:
                        pred_seq.append("O")

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_preds, all_golds)
    return avg_loss, entity_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int = 4,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, loss = model(input_ids, attention_mask, labels)
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

    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")
    print(f"实体类型：{labels[1:]}")

    print(f"\n正在加载 Qwen2.5-0.5B-Instruct 分词器...")
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"\n正在构建数据集...")
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )
    print(f"数据集规模：训练={len(train_loader.dataset)}，验证={len(val_loader.dataset)}")

    print(f"\n正在构建 Qwen + LoRA 模型...")
    model = build_model(
        use_lora=True,
        model_name=MODEL_NAME,
        num_labels=num_labels,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dropout=args.dropout,
    ).to(device)

    print(f"模型：Qwen2.5-0.5B + LoRA")
    print(f"  标签数：{num_labels}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "qwen" in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": other_params, "lr": args.lr * 10},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "best_qwen_lora_peoples.pt"
    log_path = LOG_DIR / "train_qwen_lora_peoples.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n开始训练（Qwen2.5-0.5B + LoRA）— peoples_daily 数据集...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device)
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
    print(f"\n下一步：python evaluate_qwen.py")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Qwen2.5 + LoRA NER 模型（peoples_daily 数据集）")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
