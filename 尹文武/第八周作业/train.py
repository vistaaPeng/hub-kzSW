from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import BiEncoder, CrossEncoder, TextMatchingLoss
from text_matching_utils import build_dataloader, compute_metrics, ensure_dir, get_tokenizer, load_samples, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT.parent.parent / "pretrained_models" / "bert-base-chinese"
DEFAULT_DATASET = ROOT / "data" / "bq_corpus" / "train.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text matching model.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--model_type", type=str, default="biencoder", choices=["biencoder", "crossencoder"])
    parser.add_argument("--loss_type", type=str, default="cosine", choices=["cosine", "triplet", "cross_entropy"])
    parser.add_argument("--model_name_or_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs"))
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, model_type, loss_type):
    model.train()
    total_loss = 0.0
    preds = []
    labels = []

    for batch_idx, batch in enumerate(loader):
        if model_type == "biencoder":
            input_ids_1 = batch["input_ids_1"].to(device)
            attention_mask_1 = batch["attention_mask_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attention_mask_2 = batch["attention_mask_2"].to(device)
            labels_batch = batch["labels"].to(device)

            emb1, emb2 = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            if loss_type == "cosine":
                loss = TextMatchingLoss.cosine_loss(emb1, emb2, labels_batch)
            else:
                loss = TextMatchingLoss.triplet_loss(emb1, emb2, labels_batch)

            cos = torch.nn.functional.cosine_similarity(emb1, emb2)
            preds_batch = (cos > 0.5).long().detach().cpu().tolist()
            labels_batch_cpu = labels_batch.detach().cpu().tolist()
            preds.extend(preds_batch)
            labels.extend(labels_batch_cpu)

        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = TextMatchingLoss.cross_entropy_loss(logits, labels_batch)
            preds_batch = (torch.sigmoid(logits) > 0.5).long().detach().cpu().tolist()
            labels_batch_cpu = labels_batch.detach().cpu().tolist()
            preds.extend(preds_batch)
            labels.extend(labels_batch_cpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            metrics = compute_metrics(labels, preds)
            print(
                f"batch={batch_idx:03d} loss={loss.item():.4f} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}"
            )

    avg_loss = total_loss / max(1, len(loader))
    metrics = compute_metrics(labels, preds)
    return avg_loss, metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "reports")

    samples = load_samples(data_path)
    if not samples:
        raise ValueError(f"No valid training samples found in {data_path}")

    tokenizer = get_tokenizer(args.model_name_or_path)
    train_loader = build_dataloader(
        samples,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cross_encoder=(args.model_type == "crossencoder"),
    )

    if args.model_type == "biencoder":
        model = BiEncoder(args.model_name_or_path).to(device)
    else:
        model = CrossEncoder(args.model_name_or_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = {"loss": [], "accuracy": [], "f1": []}
    best_f1 = -1.0
    best_path = output_dir / "checkpoints" / "best_model.pt"
    if args.model_type == "biencoder":
        best_path = output_dir / "checkpoints" / "best_biencoder_model.pt"
    else:
        best_path = output_dir / "checkpoints" / "best_crossencoder_model.pt"


    train_log_path = output_dir / "logs" / f"train_{args.model_type}_{args.loss_type}.log"
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,accuracy,f1\n")

    for epoch in range(args.epochs):
        start = time.time()
        loss, metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.model_type,
            args.loss_type,
        )
        history["loss"].append(loss)
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"avg_loss={loss:.4f} accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} elapsed={time.time() - start:.1f}s"
        )

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1},{loss:.6f},{metrics['accuracy']:.6f},{metrics['f1']:.6f}\n"
            )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model_type": args.model_type,
                    "loss_type": args.loss_type,
                    "state_dict": model.state_dict(),
                },
                best_path,
            )

    # save curves
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(history["loss"], marker="o")
    axes[0].set_title("Loss")
    axes[1].plot(history["accuracy"], marker="o", color="#2ecc71")
    axes[1].set_title("Accuracy")
    axes[2].plot(history["f1"], marker="o", color="#f39c12")
    axes[2].set_title("F1")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "logs" / "train_metrics.png", dpi=150)
    plt.close(fig)

    save_json(
        {
            "model_type": args.model_type,
            "loss_type": args.loss_type,
            "best_checkpoint": str(best_path),
            "best_f1": best_f1,
            "history": history,
        },
        output_dir / "logs" / "train_summary.json",
    )

    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
