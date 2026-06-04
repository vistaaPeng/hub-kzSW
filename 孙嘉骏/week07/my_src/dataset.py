"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"


# ==================== 预定义实体类型（仅类型名，不含 B-/I- 前缀） ====================
ENTITY_TYPES = ["PER", "ORG", "LOC"]   # 根据实际数据添加或修改

# ==================== 构建标签模式（自动添加 BIO 前缀） ====================
def build_label_schema() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    构建 BIO 标签体系，返回 (labels, label2id, id2label)。
    实体类型从预定义的 ENTITY_TYPES 生成，自动添加 B- 和 I- 前缀及 O。
    """
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


# ----------------------------- CluenerDataset (固定长度 padding) -----------------------------
class CluenerDataset(Dataset):
    """
    用于 BERT 等编码器模型的 NER 序列标注数据集。
    输入: data (样本列表), tokenizer, label2id, max_length
    输出: input_ids, attention_mask, token_type_ids, labels (所有序列长度均为 max_length)
    """
    def __init__(self,
                 data: List[Dict],
                 tokenizer: BertTokenizer,
                 label2id: Dict[str, int],
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

        # 预处理所有样本，直接固定长度 padding
        self.examples = []
        for sample in data:
            tokens = sample['tokens']
            tags = sample['ner_tags']
            encoded = self._encode_and_align_fixed(tokens, tags)
            self.examples.append(encoded)

    def _encode_and_align_fixed(self, tokens: List[str], tags: List[str]) -> Dict[str, torch.Tensor]:
        """
        使用 padding='max_length' 将序列填充到 max_length，并完成标签对齐。
        返回的 input_ids, attention_mask, token_type_ids, labels 均为长度 max_length 的 tensor。
        """
        # 1. 对原始 tokens 进行编码，固定长度 padding
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',           # 固定长度 padding
            return_word_ids=True,
            return_tensors='pt'             # 直接返回 PyTorch tensors
        )
        # encoding 中已经包含 input_ids, attention_mask, token_type_ids (如果有)
        # shape 是 (1, max_length)，去掉 batch 维度
        input_ids = encoding['input_ids'][0]               # (max_length,)
        attention_mask = encoding['attention_mask'][0]     # (max_length,)
        token_type_ids = encoding.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[0]
        else:
            token_type_ids = torch.zeros(self.max_length, dtype=torch.long)

        # 2. 对齐标签到 subword 级别，并填充到 max_length
        word_ids = encoding.word_ids(batch_index=0)        # list of length max_length
        label_ids = [self.label2id[tag] for tag in tags]
        aligned_labels = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)                # 特殊 token
            elif word_idx != prev_word_idx:
                aligned_labels.append(label_ids[word_idx]) # 该 token 的首个 subword
            else:
                aligned_labels.append(-100)                # 非首 subword
            prev_word_idx = word_idx
        # aligned_labels 长度已经是 max_length
        labels = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
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

    train_ds = CluenerDataset(train_records, tokenizer, label2id, max_length)
    val_ds = CluenerDataset(val_records, tokenizer, label2id, max_length)
    test_ds = CluenerDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
