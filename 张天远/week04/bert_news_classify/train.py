"""
BERT 微调训练脚本
"""
import os
import sys
import json
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

from config import *
from preprocess import BertDataset


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"bert_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger("bert_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    logger.info(f"日志: {log_file}")
    return logger


class BertClassifier(nn.Module):
    """BERT + 分类头"""

    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # (batch, hidden)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def create_dataloaders():
    train_ds = BertDataset(TRAIN_X, TRAIN_MASK, TRAIN_Y)
    val_ds = BertDataset(VAL_X, VAL_MASK, VAL_Y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, val_loader


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for x, mask, y in val_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def train(logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        label_meta = json.load(f)
    idx2label = label_meta["idx2label"]

    train_loader, val_loader = create_dataloaders()
    logger.info(f"训练 batch: {len(train_loader)}, 验证 batch: {len(val_loader)}")

    model = BertClassifier(MODEL_NAME, NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数: {total_params:,} (可训练: {trainable:,})")

    criterion = nn.CrossEntropyLoss()

    # 分组学习率：BERT 主体用低 LR，分类头用稍高 LR
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and not any(nd in n for nd in no_decay)],
         "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and any(nd in n for nd in no_decay)],
         "lr": LEARNING_RATE, "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n],
         "lr": LEARNING_RATE * 5, "weight_decay": WEIGHT_DECAY},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped)

    total_steps = len(train_loader) // GRADIENT_ACCUMULATION * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    logger.info("\n" + "=" * 60)
    logger.info(f"BERT 微调 | MODEL={MODEL_NAME} | BATCH={BATCH_SIZE}x{GRADIENT_ACCUMULATION}")
    logger.info(f"LR={LEARNING_RATE} | EPOCHS={EPOCHS} | MAX_LEN={MAX_LEN}")
    logger.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (x, mask, y) in enumerate(train_loader):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss = criterion(logits, y) / GRADIENT_ACCUMULATION
            loss.backward()
            train_loss += loss.item() * GRADIENT_ACCUMULATION * x.size(0)

            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % (len(train_loader) // 4) == 0:
                logger.info(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item() * GRADIENT_ACCUMULATION:.4f}"
                )

        # 处理最后不足 accumulation 的残余梯度
        if len(train_loader) % GRADIENT_ACCUMULATION != 0:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"\n{'='*60}\n"
            f"Epoch {epoch}/{EPOCHS} | Time: {epoch_time:.1f}s\n"
            f"  Train Loss: {avg_train_loss:.4f}\n"
            f"  Val Loss:   {val_loss:.4f}\n"
            f"  Val Acc:    {val_acc:.4f} ({val_acc*100:.2f}%)\n"
            f"  Val F1:     {val_f1:.4f}\n"
            f"{'='*60}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            ckpt = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_f1": val_f1, "val_acc": val_acc,
                "model_name": MODEL_NAME, "num_classes": NUM_CLASSES,
                "idx2label": idx2label,
            }, ckpt)
            logger.info(f"  >> 最佳模型已保存 (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  F1 未提升 ({patience_counter}/{EARLY_STOP_PATIENCE}), 最佳: {best_val_f1:.4f} @ epoch {best_epoch}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"\n早停！最佳 F1: {best_val_f1:.4f} @ epoch {best_epoch}")
            break

    logger.info(f"\n训练完成！最佳 F1={best_val_f1:.4f} @ epoch {best_epoch}")


def main():
    logger = setup_logging()
    for f in [TRAIN_X, TRAIN_MASK, TRAIN_Y]:
        if not os.path.exists(f):
            logger.error(f"缺少预编码文件: {f}\n请先运行: python preprocess.py")
            sys.exit(1)
    train(logger)


if __name__ == "__main__":
    main()
