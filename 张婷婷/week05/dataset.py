
import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, text, seq_len, char2idx):
        self.seq_len = seq_len
        self.char2idx = char2idx
        self.tokens = [char2idx[c] for c in text]

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_dataloader(file_path, seq_len, batch_size):
    # 读取文本
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    # 构建词表
    unique_chars = sorted(list(set(full_text)))
    vocab_size = len(unique_chars)
    char2idx = {c:i for i,c in enumerate(unique_chars)}
    idx2char = {i:c for i,c in enumerate(unique_chars)}
    # 数据集 & DataLoader
    ds = CharDataset(full_text, seq_len, char2idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, vocab_size, char2idx, idx2char