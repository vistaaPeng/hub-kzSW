"""TNEWS Dataset 封装"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class TNEWSDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer: BertTokenizer, max_length: int = 64):
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["sentence"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def build_dataloaders(data_dir: Path, tokenizer: BertTokenizer,
                      max_length: int = 64, batch_size: int = 32,
                      num_train: int | None = None, val_subset: int | None = None):
    """构建 train/val/test DataLoader。num_train/val_subset 可限制数据量加速。"""
    train_ds = TNEWSDataset(data_dir / "train.json", tokenizer, max_length)
    val_ds   = TNEWSDataset(data_dir / "val.json", tokenizer, max_length)
    test_ds  = TNEWSDataset(data_dir / "test.json", tokenizer, max_length)

    if num_train is not None and num_train < len(train_ds):
        train_ds.data = train_ds.data[:num_train]
    if val_subset is not None and val_subset < len(val_ds):
        val_ds.data = val_ds.data[:val_subset]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"DataLoader: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_loader, val_loader, test_loader
