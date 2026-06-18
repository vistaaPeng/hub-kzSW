"""
人民日报 NER 数据集处理模块

数据集特点：
  - 已经是 token 级别（字符分词）+ BIO 格式
  - 无需 span → BIO 转换
  - 标签体系：O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC（共7类）

与 cluener 数据集的区别：
  - cluener：span 格式，需要 span_to_bio() 转换
  - peoples_daily：已经是 token + BIO 格式，直接使用

依赖：
  pip install torch transformers seqeval scikit-learn tqdm
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"


# 人民日报 NER 标签体系
ENTITY_TYPES = ["PER", "ORG", "LOC"]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    数据格式（已预处理为 token + BIO）：
      {
        "tokens": ["海", "钓", "比", "赛", ...],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", ...]
      }

    处理流程：
      tokens → BertTokenizer (is_split_into_words=True)
             → 用 word_ids() 对齐子词标签
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
        tokens: list = row["tokens"]           # 已经是字符列表
        ner_tags: list = row["ner_tags"]        # 已经是 BIO 标签列表

        # 1. 将 BIO 标签转为 id
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. 调用 tokenizer（is_split_into_words=True 与字符对齐）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：取每个 token 对应的字符索引
        #    非首子词、特殊token 标记为 -100
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载指定分割的数据。"""
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 快速测试数据加载
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("../pretrain_models/bert-base-chinese")
    labels, label2id, id2label = build_label_schema()

    print(f"标签体系：{labels}")
    print(f"标签数：{len(labels)}")

    train_loader, val_loader, test_loader = build_dataloaders(tokenizer, label2id)

    # 测试一条数据
    batch = next(iter(train_loader))
    print(f"\n批次形状：")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # 反解第一条的标签
    labels_sample = batch["labels"][0].tolist()
    print(f"\n第一条样本的标签（部分）：")
    print(f"  {labels_sample[:20]}")

    # 还原标签名
    tag_names = [id2label[l] if l != -100 else "PAD" for l in labels_sample[:20]]
    print(f"  标签名：{tag_names}")
