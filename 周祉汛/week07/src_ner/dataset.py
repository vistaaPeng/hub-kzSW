"""
NER 数据集封装，支持字符级标签到 subword token 的对齐。
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        char_labels = item["labels"]   # list of int, length = len(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoding["offset_mapping"][0]  # (L, 2)
        token_labels = [-100] * self.max_length

        for i, (start, end) in enumerate(offset_mapping):
            if i >= self.max_length:
                break
            # [CLS] / [SEP] / [PAD] 的 offset 为 (0,0)
            if start == 0 and end == 0:
                continue
            if start < len(char_labels):
                token_labels[i] = char_labels[start]
            else:
                # 理论上不会发生，以防万一
                token_labels[i] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(token_labels, dtype=torch.long),
        }


def build_dataloaders(data_dir, tokenizer, batch_size=32, max_length=128, num_workers=0):
    train_ds = NERDataset(data_dir / "train.json", tokenizer, max_length)
    val_ds   = NERDataset(data_dir / "val.json",   tokenizer, max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"DataLoader 构建完成")
    print(f"  train: {len(train_ds)} 条, {len(train_loader)} batch")
    print(f"  val  : {len(val_ds)} 条, {len(val_loader)} batch")
    return train_loader, val_loader
