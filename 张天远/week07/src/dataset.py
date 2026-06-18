"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. peoples_daily 已是 BIO 标签格式，直接对齐即可
  3. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  4. DataLoader 工厂函数统一封装，--dataset 参数一键切换

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent

# ── 实体类型定义 ─────────────────────────────────────────────────────────

CLUENER_ENTITY_TYPES = [
    "address", "book", "company", "game",
    "government", "movie", "name", "organization",
    "position", "scene",
]

PEOPLES_DAILY_ENTITY_TYPES = ["PER", "ORG", "LOC"]


def get_entity_types(dataset: str = "cluener2020") -> list[str]:
    """返回指定数据集的实体类型列表。"""
    if dataset == "peoples_daily":
        return PEOPLES_DAILY_ENTITY_TYPES
    return CLUENER_ENTITY_TYPES


def get_data_dir(dataset: str = "cluener2020") -> Path:
    """返回指定数据集的数据目录。cluener2020 映射到 data/cluener/。"""
    dir_name = "cluener" if dataset == "cluener2020" else dataset
    return ROOT / "data" / dir_name


# ── 标签体系 ─────────────────────────────────────────────────────────────

def build_label_schema(
    dataset: str = "cluener2020",
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    entity_types = get_entity_types(dataset)
    labels = ["O"]
    for etype in entity_types:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


# ── span→BIO 转换（仅 cluener2020 使用）─────────────────────────────────

def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    """将 cluener2020 的 span 格式标注转为逐字符 BIO 标签 id 列表。

    教学要点：先全部初始化为 O，再按 span 位置填入 B/I。
    若存在嵌套实体（本数据集极少），外层实体覆盖内层。
    """
    n = len(text)
    bio = ["O"] * n

    if not label_dict:
        return [label2id[t] for t in bio]

    for etype, spans in label_dict.items():
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        for surface, positions in spans.items():
            for start, end in positions:
                if start >= n or end >= n:
                    continue
                bio[start] = b_tag
                for idx in range(start + 1, end + 1):
                    bio[idx] = i_tag

    return [label2id.get(t, 0) for t in bio]


# ── CluenerDataset（cluener2020，span 格式）──────────────────────────────

class CluenerDataset(Dataset):
    """cluener2020 的 PyTorch Dataset。

    教学流程：
      text → span_to_bio → 字符级 BIO ids
           → BertTokenizer (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
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
        text: str = row["text"]
        label_dict: dict = row.get("label") or {}

        # 1. span → 字符级 BIO id 列表
        char_labels = span_to_bio(text, label_dict, self.label2id)

        # 2. 将文本拆为字符列表，传入 tokenizer
        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


# ── PeoplesDailyDataset（人民日报 NER，BIO 标签格式）────────────────────

class PeoplesDailyDataset(Dataset):
    """peoples_daily 的 PyTorch Dataset。

    与 CluenerDataset 的关键区别：
      - 数据已是 BIO 标签列表（ner_tags 字段），不需要 span_to_bio()
      - tokens 是字符列表，可直接传入 tokenizer(is_split_into_words=True)
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
        tokens = row["tokens"]            # ["海","钓","比","赛",...]
        ner_tags = row["ner_tags"]        # ["O","O","O","O","B-LOC",...]

        # 1. BIO 标签字符串 → id（不需要 span_to_bio）
        char_labels = [self.label2id.get(t, 0) for t in ner_tags]

        # 2. Tokenize（与 CluenerDataset 相同：逐字符输入）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐（与 CluenerDataset 完全相同的逻辑）
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


# ── 数据加载 + DataLoader 工厂 ──────────────────────────────────────────

def load_records(split: str, data_dir: Path) -> list:
    """从 JSON 文件加载记录列表。"""
    with open(data_dir / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    dataset: str = "cluener2020",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader。

    Parameters
    ----------
    dataset : str
        "cluener2020" 或 "peoples_daily"，决定使用哪个 Dataset 类和数据目录。

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    data_dir = get_data_dir(dataset)
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    DatasetClass = PeoplesDailyDataset if dataset == "peoples_daily" else CluenerDataset

    train_ds = DatasetClass(train_records, tokenizer, label2id, max_length)
    val_ds = DatasetClass(val_records, tokenizer, label2id, max_length)
    test_ds = DatasetClass(test_records, tokenizer, label2id, max_length)

    print(f"数据集：{dataset} | "
          f"训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)} | "
          f"标签数={len(label2id)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
