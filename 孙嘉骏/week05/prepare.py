import glob
import torch
from torch.utils.data import Dataset

# ─────────────────────────── 数据 ───────────────────────────
def load_corpus(pattern="*.txt"):
    """
    功能：
        加载语料库
    参数：
        pattern：文件名模式
    返回值：
        语料库字符串
    使用方法：
        text = load_corpus("循环神经网络语言模型\\*.txt")
    """
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)

def build_vocab(text):
    """
    功能：
        构建字符索引映射表
    参数：
        text：语料库字符串
    返回值：
        字符索引映射表
    使用方法：
        char2idx, idx2char = build_vocab(text)
    """
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    """
    数据集类
    """
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y











