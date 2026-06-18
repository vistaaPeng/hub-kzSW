import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from transformers import (
    get_linear_schedule_with_warmup
)

from tqdm.auto import tqdm

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from dataset import (
    create_dataloader,
    tokenizer
)

from model import build_model


# =====================================================
# Seed
# =====================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# EarlyStopping
# =====================================================

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# =====================================================
# Args
# =====================================================

parser = argparse.ArgumentParser()
parser.add_argument("--use_crf", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=3e-5)

parser.add_argument("--warmup_ratio", type=float, default=0.1)
args = parser.parse_args()

set_seed(42)

# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

MODEL_DIR = ROOT.parent.parent / "pretrained_models" / "bert-base-chinese"

# =====================================================
# Labels
# =====================================================

with open(DATA_DIR / "label_names.json", "r", encoding="utf-8") as f:
    LABELS = json.load(f)

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# =====================================================
# DataLoaders
# =====================================================

train_loader = create_dataloader(
    DATA_DIR / "train.json",
    LABEL2ID,
    tokenizer,
    batch_size=args.batch_size,
    shuffle=True
)

val_loader = create_dataloader(
    DATA_DIR / "validation.json",
    LABEL2ID,
    tokenizer,
    batch_size=args.batch_size
)

test_loader = create_dataloader(
    DATA_DIR / "test.json",
    LABEL2ID,
    tokenizer,
    batch_size=args.batch_size
)

print(f"Train: {len(train_loader.dataset)}")
print(f"Val:   {len(val_loader.dataset)}")
print(f"Test:  {len(test_loader.dataset)}")

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# Model
# =====================================================

model = build_model(
    bert_model_path=str(MODEL_DIR),
    num_labels=NUM_LABELS,
    use_crf=args.use_crf
)
model.to(device)

# =====================================================
# Optimizer & Scheduler
# =====================================================

optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_loader) * args.epochs
warmup_steps = int(total_steps * args.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup_steps,
    total_steps
)

# =====================================================
# TensorBoard
# =====================================================

writer = SummaryWriter(log_dir="runs/bert_ner")

# =====================================================
# Eval function
# =====================================================

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)

            if args.use_crf:
                preds = outputs["preds"]
                for i in range(len(preds)):
                    pred_seq = preds[i]
                    label_seq = labels[i].cpu().tolist()
                    true_tags, pred_tags = [], []
                    pred_idx = 0
                    for lab in label_seq:
                        if lab == -100:
                            continue
                        true_tags.append(ID2LABEL[lab])
                        pred_tags.append(ID2LABEL[pred_seq[pred_idx]])
                        pred_idx += 1
                    y_true.append(true_tags)
                    y_pred.append(pred_tags)
            else:
                preds = outputs["preds"].cpu()
                for pred_seq, label_seq in zip(preds, labels):
                    true_tags, pred_tags = [], []
                    for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                        if l == -100:
                            continue
                        true_tags.append(ID2LABEL[l])
                        pred_tags.append(ID2LABEL[p])
                    y_true.append(true_tags)
                    y_pred.append(pred_tags)

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    return p, r, f1, report

# =====================================================
# Train
# =====================================================

best_f1 = 0
loss_history, f1_history = [], []

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask, labels)
        loss = output["loss"]

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    p, r, f1, report = evaluate(model, val_loader)

    # TensorBoard logging
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Metric/Precision", p, epoch)
    writer.add_scalar("Metric/Recall", r, epoch)
    writer.add_scalar("Metric/F1", f1, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    loss_history.append(train_loss)
    f1_history.append(f1)

    print(f"Epoch {epoch:02d} | Loss={train_loss:.4f} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved best model with F1 =", best_f1)

writer.close()

# =====================================================
# Test
# =====================================================

model.load_state_dict(torch.load("best_model.pt", map_location=device))
p, r, f1, report = evaluate(model, test_loader)

print("\n===== TEST =====")
print(report)
print(f"P={p:.4f} R={r:.4f} F1={f1:.4f}")

# =====================================================
# Plot curves
# =====================================================

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Train Loss")
plt.savefig(PLOT_DIR / "loss_curve.png")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(f1_history)
plt.title("Val F1")
plt.savefig(PLOT_DIR / "f1_curve.png")
plt.show()

print(f"\nBest Val F1 = {best_f1:.4f}")
