"""
BERT 预处理：用 bert-base-chinese tokenizer 编码所有数据并保存为 .pt
"""
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import *


def build_label_map():
    labels = set()
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                labels.add(parts[0])
    label_list = sorted(labels)
    label2idx = {l: i for i, l in enumerate(label_list)}
    idx2label = {i: l for l, i in label2idx.items()}
    print(f"类别: {label_list}")
    return label2idx, idx2label


def process_file(file_path, tokenizer, label2idx, max_len):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = parts[0], parts[1]

            encoded = tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(encoded["input_ids"].squeeze(0))
            attention_mask_list.append(encoded["attention_mask"].squeeze(0))
            labels_list.append(label2idx[label])

    return (
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.tensor(labels_list, dtype=torch.long),
    )


class BertDataset(Dataset):
    def __init__(self, x_path, mask_path, y_path):
        self.x = torch.load(x_path, weights_only=True)
        self.mask = torch.load(mask_path, weights_only=True)
        self.y = torch.load(y_path, weights_only=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]


def main():
    print("=" * 60)
    print("BERT 预处理：tokenize 所有数据")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"\n[1/3] 加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n[2/3] 构建标签映射...")
    label2idx, idx2label = build_label_map()
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, ensure_ascii=False)

    print("\n[3/3] 编码数据 (max_len={})...".format(MAX_LEN))
    for split, file_path, x_path, m_path, y_path in [
        ("训练集", TRAIN_FILE, TRAIN_X, TRAIN_MASK, TRAIN_Y),
        ("验证集", VAL_FILE,   VAL_X,   VAL_MASK,   VAL_Y),
        ("测试集", TEST_FILE,  TEST_X,  TEST_MASK,  TEST_Y),
    ]:
        print(f"  处理 {split}...")
        x, mask, y = process_file(file_path, tokenizer, label2idx, MAX_LEN)
        print(f"    x: {x.shape}, mask: {mask.shape}, y: {y.shape}")
        torch.save(x, x_path)
        torch.save(mask, m_path)
        torch.save(y, y_path)

    print("\n" + "=" * 60)
    print("BERT 预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
