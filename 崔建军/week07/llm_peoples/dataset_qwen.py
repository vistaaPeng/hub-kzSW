"""
Qwen2.5-0.5B NER 数据集处理模块

基于 peoples_daily 数据集，使用 Qwen2.5-0.5B-Instruct 作为基座模型 + LoRA 微调

与 BERT 模型的关键区别：
  1. 分词器：Qwen2.5 使用 Qwen2Tokenizer，需要处理 chat template
  2. 子词对齐：Qwen 是字节级 BPE，需要用 word_ids() 对齐 token 与标签
  3. 数据格式：Qwen2.5-Instruct 使用 chat template 格式
  4. LoRA 微调：只训练 LoRA 参数，不更新原模型参数

使用方式：
  from dataset_qwen import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2Tokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

ENTITY_TYPES = ["PER", "ORG", "LOC"]

MODEL_NAME = Path(r"D:\BaiduNetdiskDownload\AI\pretrain_models\Qwen2.5-0.5B-Instruct")


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class QwenNERDataset(Dataset):
    """Qwen2.5 NER 的 PyTorch Dataset。

    数据格式：
      {
        "tokens": ["海", "钓", "比", "赛", "地", "点", ...],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", ...]
      }

    处理流程：
      1. 构建 prompt：角色扮演 + 任务说明 + 输入文本
      2. Qwen2 tokenizer 处理（is_split_into_words=True）
      3. 用 word_ids() 对齐 token 与标签
      4. 返回 input_ids / attention_mask / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: Qwen2Tokenizer,
        label2id: dict,
        max_length: int = 256,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list = row["tokens"]
        ner_tags: list = row["ner_tags"]

        tag_ids = [self.label2id[t] for t in ner_tags]

        text = "".join(tokens)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        word_ids = encoding.word_ids() if hasattr(encoding, 'word_ids') else None

        labels = torch.full((self.max_length,), -100, dtype=torch.long)

        if word_ids is not None:
            previous_word_idx = None
            label_idx = 0
            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    if label_idx < len(tag_ids):
                        labels[i] = tag_ids[label_idx]
                        label_idx += 1
                    else:
                        labels[i] = self.label2id["O"]
                else:
                    if label_idx > 0:
                        labels[i] = tag_ids[min(label_idx - 1, len(tag_ids) - 1)]
                previous_word_idx = word_idx
        else:
            text_len = min(len(tag_ids), self.max_length - 2)
            labels[1:text_len+1] = torch.tensor(tag_ids[:text_len], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_jsonl(path: Path) -> list:
    """加载 JSON 数据文件（支持标准 JSON 数组格式）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        return [data]


def build_dataloaders(
    tokenizer: Qwen2Tokenizer,
    label2id: dict,
    batch_size: int = 8,
    max_length: int = 256,
    data_dir: Path = DATA_DIR,
    split: str = "train",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader。"""
    train_records = load_jsonl(data_dir / "train.json")
    val_records = load_jsonl(data_dir / "validation.json")
    test_records = load_jsonl(data_dir / "test.json")

    train_dataset = QwenNERDataset(train_records, tokenizer, label2id, max_length)
    val_dataset = QwenNERDataset(val_records, tokenizer, label2id, max_length)
    test_dataset = QwenNERDataset(test_records, tokenizer, label2id, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
        },
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
        },
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
        },
    )

    return train_loader, val_loader, test_loader
