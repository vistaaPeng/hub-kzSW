"""数据探索：类别分布 + 文本长度 + Token 长度"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer

from common.config import DATA_DIR, FIG_DIR, BERT_MODEL_NAME, LABEL_NAMES
from common.utils import setup_plot_font


def plot_label_distribution(data, save_path):
    label_counts = {}
    for item in data:
        name = LABEL_NAMES[item["label"]]
        label_counts[name] = label_counts.get(name, 0) + 1

    names = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(10, 5))
    colors = plt.cm.Set3(range(len(names)))
    bars = plt.bar(names, counts, color=colors)
    for bar, c in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(c), ha="center", fontsize=8)
    plt.title("训练集类别分布")
    plt.xlabel("类别")
    plt.ylabel("样本数")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  类别分布图 → {save_path}")


def plot_text_length(data, save_path):
    char_lens = [len(item["sentence"]) for item in data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(char_lens, bins=50, edgecolor="black")
    axes[0].set_title("文本字符长度分布")
    axes[0].set_xlabel("字符数")
    axes[0].set_ylabel("样本数")
    axes[0].axvline(sum(char_lens) / len(char_lens), color="red", linestyle="--", label=f"均值: {sum(char_lens)/len(char_lens):.1f}")

    axes[1].boxplot(char_lens, vert=True)
    axes[1].set_title("文本长度箱线图")
    axes[1].set_ylabel("字符数")
    p99 = sorted(char_lens)[int(len(char_lens) * 0.99)]
    axes[1].axhline(p99, color="red", linestyle="--", label=f"P99: {p99}")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  文本长度图 → {save_path}")


def plot_token_length(data, tokenizer, save_path):
    token_lens = []
    for item in data:
        tokens = tokenizer.encode(item["sentence"], add_special_tokens=True)
        token_lens.append(len(tokens))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(token_lens, bins=50, edgecolor="black", color="orange")
    axes[0].set_title("Token 长度分布")
    axes[0].set_xlabel("Token 数")
    axes[0].set_ylabel("样本数")
    axes[0].axvline(sum(token_lens) / len(token_lens), color="red", linestyle="--", label=f"均值: {sum(token_lens)/len(token_lens):.1f}")

    axes[1].scatter(char_lens := [len(item["sentence"]) for item in data], token_lens, alpha=0.3, s=1)
    axes[1].set_xlabel("字符数")
    axes[1].set_ylabel("Token 数")
    axes[1].set_title("字符数 vs Token 数")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Token 长度图 → {save_path}")


def main():
    setup_plot_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("加载训练数据 ...")
    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  训练集: {len(train_data)} 条")

    plot_label_distribution(train_data, FIG_DIR / "label_distribution.png")
    plot_text_length(train_data, FIG_DIR / "text_length.png")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    plot_token_length(train_data, tokenizer, FIG_DIR / "token_length.png")

    char_lens = [len(item["sentence"]) for item in train_data]
    print(f"\n统计摘要:")
    print(f"  字符: 均值={sum(char_lens)/len(char_lens):.1f}, P99={sorted(char_lens)[int(len(char_lens)*0.99)]}")
    print(f"  Token/字比: {sum(len(tokenizer.encode(item['sentence'], add_special_tokens=True)) for item in train_data) / sum(char_lens):.2f}")


if __name__ == "__main__":
    main()
