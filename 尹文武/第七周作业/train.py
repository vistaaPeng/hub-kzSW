import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

from dataset import create_dataloader, tokenizer
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
# Path
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

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)


# =====================================================
# Data
# =====================================================
train_loader = create_dataloader(DATA_DIR / "train.json", LABEL2ID, tokenizer, batch_size=args.batch_size, shuffle=True)
val_loader = create_dataloader(DATA_DIR / "validation.json", LABEL2ID, tokenizer, batch_size=args.batch_size)
test_loader = create_dataloader(DATA_DIR / "test.json", LABEL2ID, tokenizer, batch_size=args.batch_size)


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
).to(device)


# =====================================================
# Optimizer
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
# Eval
# =====================================================
def evaluate():
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask, labels=None)

            preds = outputs["preds"]

            # CRF: list[list]
            # if args.use_crf:
            #     for i, pred_seq in enumerate(preds):
            #         true_seq = labels[i].tolist()

            #         t_tags, p_tags = [], []
            #         j = 0

            #         for lab in true_seq:
            #             if lab == -100:
            #                 continue
            #             t_tags.append(ID2LABEL[lab])
            #             p_tags.append(ID2LABEL[pred_seq[j]])
            #             j += 1

            #         y_true.append(t_tags)
            #         y_pred.append(p_tags)
            if args.use_crf:
                for i, pred_seq in enumerate(preds):
                    true_seq = labels[i].tolist()

                    # CRF decode 返回的 preds 包含所有 mask=True 位置的预测
                    # 包括我们强制设为 True 的第一个位置 ([CLS])
                    # 但我们在评估时应该排除 [CLS] 位置（它的 label 是 -100）

                    t_tags, p_tags = [], []
                    pred_idx = 0

                    for idx, lab in enumerate(true_seq):
                        if lab == -100:
                            # 如果是第一个位置 [CLS]，preds 中也有对应的预测，跳过它
                            if idx == 0 and pred_idx < len(pred_seq):
                                pred_idx += 1  # 跳过 CRF 对 [CLS] 的预测
                            continue

                        t_tags.append(ID2LABEL[lab])
                        if pred_idx < len(pred_seq):
                            p_tags.append(ID2LABEL[pred_seq[pred_idx]])
                            pred_idx += 1

                    y_true.append(t_tags)
                    y_pred.append(p_tags)

            # Softmax
            else:
                preds = preds.cpu()

                for p_seq, l_seq in zip(preds, labels):
                    t_tags, p_tags = [], []

                    for p, l in zip(p_seq.tolist(), l_seq.tolist()):
                        if l == -100:
                            continue
                        t_tags.append(ID2LABEL[l])
                        p_tags.append(ID2LABEL[p])

                    y_true.append(t_tags)
                    y_pred.append(p_tags)

    return (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )


# =====================================================
# Train
# =====================================================
best_f1 = 0
loss_list, f1_list = [], []

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, labels)

        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    p, r, f1 = evaluate()

    loss_list.append(avg_loss)
    f1_list.append(f1)

    print(
        f"\nEpoch {epoch:02d} | "
        f"Loss={avg_loss:.4f} | "
        f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved best model:", best_f1)


# =====================================================
# Test
# =====================================================
model.load_state_dict(torch.load("best_model.pt", map_location=device))

p, r, f1 = evaluate()

print("\n===== TEST =====")
print(f"P={p:.4f} R={r:.4f} F1={f1:.4f}")


# =====================================================
# Plot
# =====================================================
plt.figure()
plt.plot(loss_list)
plt.title("Train Loss")
plt.savefig(PLOT_DIR / "loss.png")
plt.close()

plt.figure()
plt.plot(f1_list)
plt.title("Val F1")
plt.savefig(PLOT_DIR / "f1.png")
plt.show()

print("Done.")
