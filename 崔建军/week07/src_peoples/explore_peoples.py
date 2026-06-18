"""
人民日报 NER 数据集探索与可视化

与 cluener 的关键差异：
  1. 数据格式：peoples_daily 已经是分词后的 token 列表 + BIO 标签列表
     - cluener: {"text": "...", "label": {"name": {"叶老桂": [[0,2]]}}} (span格式)
     - peoples_daily: {"tokens": ["海","钓",...], "ner_tags": ["O","O","B-LOC",...]} (BIO格式)
  2. 实体类型：只有 3 类（PER/ORG/LOC），共 7 个标签
  3. 统计方式：需要从 BIO 标签序列中提取实体边界

教学重点：
  1. BIO 标签序列的解析方法（如何从标签序列提取实体）
  2. 各实体类型的分布差异
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布

使用方式：
  python explore_peoples.py

输出：
  outputs/figures/entity_distribution_peoples.png   各类实体频次分布
  outputs/figures/text_length_distribution_peoples.png  文本长度分布
  outputs/figures/entity_length_distribution_peoples.png 实体长度分布
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "figures"


def load_split(split: str) -> list:
    """加载数据集（train/validation/test）"""
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bio_tags(tokens: list, ner_tags: list) -> list:
    """从 BIO 标签序列中提取实体列表。
    
    参数：
      tokens: 分词后的 token 列表
      ner_tags: 对应的 BIO 标签列表
    
    返回：
      实体列表，每个实体为 {"type": "PER", "surface": "张三", "length": 2}
    """
    entities = []
    current_entity = None
    
    for i, tag in enumerate(ner_tags):
        if tag.startswith("B-"):
            # 新实体开始
            if current_entity:
                entities.append(current_entity)
            etype = tag[2:]
            current_entity = {
                "type": etype,
                "surface": tokens[i],
                "length": 1,
            }
        elif tag.startswith("I-"):
            # 当前实体继续
            if current_entity and current_entity["type"] == tag[2:]:
                current_entity["surface"] += tokens[i]
                current_entity["length"] += 1
            else:
                # 非法转移：I-X 没有对应的 B-X，忽略
                current_entity = None
        else:
            # O 标签，结束当前实体
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # 处理句子末尾的实体
    if current_entity:
        entities.append(current_entity)
    
    return entities


def collect_stats(records: list) -> dict:
    """收集数据集统计信息"""
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}
    all_tokens = []
    
    for row in records:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        text_lengths.append(len(tokens))
        all_tokens.append(tokens)
        
        entities = parse_bio_tags(tokens, ner_tags)
        entity_per_sentence.append(len(entities))
        
        for entity in entities:
            etype = entity["type"]
            entity_type_counts[etype] += 1
            entity_lengths.append(entity["length"])
            
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(entity["surface"])
    
    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
        "all_tokens": all_tokens,
    }


def print_summary(stats_train: dict, stats_val: dict):
    """打印统计摘要"""
    print("=" * 70)
    print("人民日报 NER 数据集统计摘要")
    print("=" * 70)
    
    print("\n【训练集】")
    print(f"  样本数：{len(stats_train['text_lengths'])} 条")
    print(f"  文本平均长度：{sum(stats_train['text_lengths']) / len(stats_train['text_lengths']):.1f} token")
    print(f"  文本最大长度：{max(stats_train['text_lengths'])} token")
    max_len = max(stats_train['text_lengths'])
    max_idx = stats_train['text_lengths'].index(max_len)
    max_tokens = stats_train['all_tokens'][max_idx]
    print(f"  最长文本示例（前50 token）：{''.join(max_tokens[:50])}...")

    print(f"  文本长度中位数：{sorted(stats_train['text_lengths'])[len(stats_train['text_lengths'])//2]} token")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / len(stats_train['entity_per_sentence']):.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    if stats_train['entity_lengths']:
        print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} token")
    
    print("\n【各类实体频次（训练集）】")
    et_label = {
        "PER": "人名",
        "ORG": "组织机构",
        "LOC": "地名",
    }
    total = sum(stats_train["entity_type_counts"].values())
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {etype:6s} ({cn:8s}) : {cnt:6d} 条 ({pct:.1f}%)")
    
    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = et_label.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:6s} ({cn}) : {' | '.join(examples)}")
    
    print("\n【验证集】")
    print(f"  样本数：{len(stats_val['text_lengths'])} 条")
    print(f"  实体总数：{sum(stats_val['entity_type_counts'].values())}")
    
    print()


def plot_entity_distribution(stats_train: dict):
    """绘制实体类型分布柱状图"""
    et_label = {
        "PER": "人名",
        "ORG": "机构",
        "LOC": "地名",
    }
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k,k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_title("人民日报 NER 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "entity_distribution_peoples.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_distribution_peoples.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict):
    """绘制文本长度分布图"""
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title("文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("token 数量")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution_peoples.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'text_length_distribution_peoples.png'}")
    plt.close()
    print(f"  P95 文本长度={p95} token，建议 max_length=128")


def plot_entity_length_distribution(stats_train: dict):
    """绘制实体长度分布图"""
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:15]], ys[:15], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title("实体长度分布（训练集，前15）", fontsize=14)
    ax.set_xlabel("实体 token 数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "entity_length_distribution_peoples.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_length_distribution_peoples.png'}")
    plt.close()
    
    if stats_train['entity_lengths']:
        avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
        print(f"  实体平均长度={avg_len:.1f} token，CRF 对短实体边界识别优势更明显")


def main():
    parse_args()
    
    train_records = load_split("train")
    val_records = load_split("validation")
    
    print(f"加载数据：训练集 {len(train_records)} 条，验证集 {len(val_records)} 条")
    
    stats_train = collect_stats(train_records)
    
    stats_val = collect_stats(val_records)
    
    print_summary(stats_train, stats_val)
    
    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)
    
    print("\n探索完成！图表已保存到 outputs/figures/")
    print("下一步：python train_peoples.py               # 训练 BERT+Linear")
    print("         python train_peoples.py --use_crf    # 训练 BERT+CRF")


def parse_args():
    parser = argparse.ArgumentParser(description="探索人民日报 NER 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()