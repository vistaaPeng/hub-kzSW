from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

# =========================
# 配置路径
# =========================
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
LABEL_FILE = DATA_DIR / "label_names.json"
MAX_LEN = 256
BATCH_SIZE = 32

# =========================
# 标签加载
# =========================
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# =========================
# Tokenizer
# =========================
MODEL_DIR = ROOT.parent.parent / "pretrained_models" / "bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR), local_files_only=True)

print("ROOT =", ROOT)
print("MODEL_DIR =", MODEL_DIR)
print("exists =", MODEL_DIR.exists())
print("tokenizer type =", type(tokenizer))

# =========================
# Dataset
# =========================
class NERDataset(Dataset):
    def __init__(self, json_file, label2id, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        with open(json_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]["tokens"]
        tags = self.samples[idx]["ner_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(self.label2id[tags[word_idx]])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# =========================
# DataLoader
# =========================
def create_dataloader(json_file, label2id, tokenizer, batch_size=BATCH_SIZE, shuffle=False):
    dataset = NERDataset(json_file, label2id, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# =========================
# 示例
# =========================
if __name__ == "__main__":
    train_loader = create_dataloader(DATA_DIR / "train.json", LABEL2ID, tokenizer, shuffle=True)
    for batch in train_loader:
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)
        break
