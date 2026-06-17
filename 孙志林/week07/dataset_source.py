"""
人民日报 NER 数据集类

与 cluener 数据集的核心区别：
  - cluener：text(str) + label(span dict) → 需要 span_to_bio() 转换
  - 人民日报：tokens(list) + labels(BIO list) → 直接使用，无需转换

数据格式示例：
  {
    "tokens": ["海", "钓", "比", "赛", "地", "点", "在", ...],
    "labels": ["O", "O", "O", "O", "O", "O", "O", ...]
  }

标签体系（7 个）：O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent
DATA_DIR = ROOT.parent / "week7 序列标注问题" / "序列标注项目" / "data" / "peoples_daily"


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从 label_names.json 构建标签体系，返回 (labels, label2id, id2label)。"""
    with open(DATA_DIR / "label_names.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 数据集。

    数据已预分词且标签已是 BIO 格式，处理流程：
      tokens(list[str]) → BERT tokenize(is_split_into_words=True)
                        → word_ids() 子词对齐（非首子词设为 -100）
                        → 返回 input_ids / attention_mask / token_type_ids / labels
    """

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
        label_strs: list[str] = row["ner_tags"]

        # 1. BIO 字符串 → label id（数据已是 BIO，直接转换）
        char_labels = [self.label2id.get(l, 0) for l in label_strs]

        # 2. BERT tokenize（is_split_into_words=True 保持 word_ids 与 tokens 对齐）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：非首子词和特殊 token 标记为 -100（cross_entropy 的 ignore_index）
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                aligned_labels.append(char_labels[wid] if wid < len(char_labels) else -100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_records(split: str) -> list:
    with open(DATA_DIR / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train")
    val_records = load_records("validation")
    test_records = load_records("test")

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
