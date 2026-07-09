from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT.parent.parent / "pretrained_models" / "bert-base-chinese"


class PairDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "text1": sample["text1"],
            "text2": sample["text2"],
            "label": int(sample.get("label", 0)),
        }


class PairCollator:
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts1 = [item["text1"] for item in batch]
        texts2 = [item["text2"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoded1 = self.tokenizer(
            texts1,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded2 = self.tokenizer(
            texts2,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids_1": encoded1["input_ids"],
            "attention_mask_1": encoded1["attention_mask"],
            "input_ids_2": encoded2["input_ids"],
            "attention_mask_2": encoded2["attention_mask"],
            "labels": labels,
            "raw_texts": batch,
        }


class CrossEncoderCollator:
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts1 = [item["text1"] for item in batch]
        texts2 = [item["text2"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

        encoded = self.tokenizer(
            texts1,
            texts2,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "raw_texts": batch,
        }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_csv_or_tsv(path: Path) -> List[Dict]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [row for row in reader]


def inspect_sample_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() in {".csv", ".tsv"}:
        return "table"
    return "unknown"


def normalize_sample(row: Dict) -> Optional[Dict]:
    if not isinstance(row, dict):
        return None

    text1 = (
        row.get("sentence1")
        or row.get("text1")
        or row.get("query")
        or row.get("left")
        or row.get("premise")
    )
    text2 = (
        row.get("sentence2")
        or row.get("text2")
        or row.get("doc")
        or row.get("right")
        or row.get("hypothesis")
    )
    label = row.get("label")

    if text1 is None or text2 is None:
        return None
    if label is None:
        label = row.get("similar")
    if label is None:
        return None

    return {
        "text1": str(text1),
        "text2": str(text2),
        "label": int(label),
    }


def load_samples(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    fmt = inspect_sample_format(path)
    if fmt == "jsonl":
        rows = read_jsonl(path)
    elif fmt == "table":
        rows = read_csv_or_tsv(path)
    else:
        rows = []

    samples = []
    for row in rows:
        sample = normalize_sample(row)
        if sample is not None:
            samples.append(sample)
    return samples


def build_dataloader(samples: List[Dict], tokenizer, batch_size: int = 16, max_length: int = 128, cross_encoder: bool = False):
    if cross_encoder:
        collator = CrossEncoderCollator(tokenizer, max_length=max_length)
    else:
        collator = PairCollator(tokenizer, max_length=max_length)
    return DataLoader(
        PairDataset(samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )


def get_tokenizer(model_name_or_path: Optional[str] = None):
    path = model_name_or_path if model_name_or_path else str(DEFAULT_MODEL_PATH)
    return AutoTokenizer.from_pretrained(path)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_json(data: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
