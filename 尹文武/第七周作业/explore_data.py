import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 配置
# ==========================

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

sns.set_style("whitegrid")

# ==========================
# 标签加载
# ==========================

label_path = os.path.join(DATA_DIR, "label_names.json")
print("label_path", label_path)
with open(label_path, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

ENTITY_TYPES = sorted(
    list(
        {
            tag[2:]
            for tag in LABELS
            if tag != "O"
        }
    )
)

print("标签集:")
print(LABELS)

print("\n实体类型:")
print(ENTITY_TYPES)

# ==========================
# BIO解析
# ==========================

def extract_entities(tokens, tags):

    entities = []

    start = None
    entity_type = None

    for i, tag in enumerate(tags):

        if tag.startswith("B-"):

            if start is not None:
                entities.append({
                    "text": "".join(tokens[start:i]),
                    "type": entity_type,
                    "start": start,
                    "end": i - 1,
                    "length": i - start
                })

            start = i
            entity_type = tag[2:]

        elif tag.startswith("I-"):

            if start is None:
                start = i
                entity_type = tag[2:]

        else:

            if start is not None:
                entities.append({
                    "text": "".join(tokens[start:i]),
                    "type": entity_type,
                    "start": start,
                    "end": i - 1,
                    "length": i - start
                })

            start = None
            entity_type = None

    if start is not None:
        entities.append({
            "text": "".join(tokens[start:]),
            "type": entity_type,
            "start": start,
            "end": len(tags)-1,
            "length": len(tags)-start
        })

    return entities

# ==========================
# BIO合法性检查
# ==========================

def bio_check(tags):

    errors = []

    for i, tag in enumerate(tags):

        if not tag.startswith("I-"):
            continue

        if i == 0:
            errors.append(i)
            continue

        prev = tags[i - 1]

        if prev == "O":
            errors.append(i)

        elif prev[2:] != tag[2:]:
            errors.append(i)

    return errors

# ==========================
# 读取所有数据集
# ==========================

datasets = {}

for file in Path(DATA_DIR).glob("*.json"):

    if file.name == "label_names.json":
        continue

    with open(file, "r", encoding="utf-8") as f:
        datasets[file.stem] = json.load(f)

print("\n发现数据集:")
for name in datasets:
    print(name)

# ==========================
# 全局统计
# ==========================

sentence_lengths = []

entity_counter = Counter()

entity_text_counter = Counter()

tag_counter = Counter()

entity_lengths = []

position_distribution = []

bio_error_count = 0

entity_len_by_type = defaultdict(list)

co_occurrence = defaultdict(Counter)

dataset_statistics = {}

# ==========================
# 遍历数据集
# ==========================

for dataset_name, data in datasets.items():

    sample_count = len(data)

    token_count = 0

    dataset_entity_count = 0

    for sample in data:

        tokens = sample["tokens"]
        tags = sample["ner_tags"]

        token_count += len(tokens)

        sentence_lengths.append(len(tokens))

        tag_counter.update(tags)

        errors = bio_check(tags)

        bio_error_count += len(errors)

        entities = extract_entities(tokens, tags)

        dataset_entity_count += len(entities)

        current_types = set()

        for ent in entities:

            entity_counter[ent["type"]] += 1

            entity_text_counter[ent["text"]] += 1

            entity_lengths.append(ent["length"])

            entity_len_by_type[ent["type"]].append(
                ent["length"]
            )

            current_types.add(ent["type"])

            position_distribution.append(
                ent["start"] / len(tokens)
            )

        for t1 in current_types:
            for t2 in current_types:
                if t1 != t2:
                    co_occurrence[t1][t2] += 1

    dataset_statistics[dataset_name] = {
        "samples": sample_count,
        "tokens": token_count,
        "entities": dataset_entity_count
    }

# ==========================
# 汇总统计
# ==========================

summary = {
    "datasets": dataset_statistics,
    "entity_types": ENTITY_TYPES,
    "label_count": len(LABELS),
    "total_entities": sum(entity_counter.values()),
    "unique_entities": len(entity_text_counter),
    "bio_errors": bio_error_count,
    "avg_sentence_length": round(
        np.mean(sentence_lengths), 2
    ),
    "max_sentence_length": int(
        np.max(sentence_lengths)
    ),
    "min_sentence_length": int(
        np.min(sentence_lengths)
    )
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "dataset_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )

# ==========================
# 图1 标签分布
# ==========================

plt.figure(figsize=(12, 6))

counts = [
    tag_counter.get(tag, 0)
    for tag in LABELS
]

sns.barplot(
    x=LABELS,
    y=counts
)

plt.title("Tag Distribution")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_tag_distribution.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 图2 实体类别分布
# ==========================

plt.figure(figsize=(8, 5))

counts = [
    entity_counter.get(t, 0)
    for t in ENTITY_TYPES
]

sns.barplot(
    x=ENTITY_TYPES,
    y=counts
)

plt.title("Entity Type Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_entity_distribution.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 图3 句长分布
# ==========================

plt.figure(figsize=(10, 6))

sns.histplot(
    sentence_lengths,
    bins=50,
    kde=True
)

plt.title("Sentence Length Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_sentence_length_distribution.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 图4 实体长度分布
# ==========================

plt.figure(figsize=(10, 6))

sns.histplot(
    entity_lengths,
    bins=20,
    kde=True
)

plt.title("Entity Length Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_entity_length_distribution.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 图5 高频实体TOP30
# ==========================

top_entities = entity_text_counter.most_common(30)

if len(top_entities):

    names = [x[0] for x in top_entities]
    counts = [x[1] for x in top_entities]

    plt.figure(figsize=(12, 8))

    sns.barplot(
        x=counts,
        y=names
    )

    plt.title("Top 30 Frequent Entities")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "05_top_entities.png"
        ),
        dpi=300
    )

    plt.close()

# ==========================
# 图6 实体位置分布
# ==========================

plt.figure(figsize=(10, 6))

sns.histplot(
    position_distribution,
    bins=20,
    kde=True
)

plt.title(
    "Entity Relative Position Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_entity_position_distribution.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 图7 各类别长度箱线图
# ==========================

rows = []

for t, lengths in entity_len_by_type.items():

    for l in lengths:
        rows.append({
            "type": t,
            "length": l
        })

if rows:

    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="type",
        y="length"
    )

    plt.title(
        "Entity Length by Type"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "07_entity_length_boxplot.png"
        ),
        dpi=300
    )

    plt.close()

# ==========================
# 图8 长尾分析
# ==========================

freqs = list(
    entity_text_counter.values()
)

if freqs:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        freqs,
        bins=50,
        log_scale=True
    )

    plt.title(
        "Long Tail Entity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "08_long_tail_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# ==========================
# 图9 实体共现热力图
# ==========================

matrix = []

for t1 in ENTITY_TYPES:

    row = []

    for t2 in ENTITY_TYPES:

        row.append(
            co_occurrence[t1][t2]
        )

    matrix.append(row)

plt.figure(figsize=(8, 6))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    xticklabels=ENTITY_TYPES,
    yticklabels=ENTITY_TYPES
)

plt.title(
    "Entity Co-occurrence Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_entity_cooccurrence.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 图10 标签转移矩阵
# ==========================

transition = np.zeros(
    (
        len(LABELS),
        len(LABELS)
    ),
    dtype=np.int64
)

label2id = {
    x: i
    for i, x in enumerate(LABELS)
}

for dataset_name, data in datasets.items():

    for sample in data:

        tags = sample["ner_tags"]

        for i in range(
            len(tags)-1
        ):

            transition[
                label2id[tags[i]],
                label2id[tags[i+1]]
            ] += 1

plt.figure(figsize=(12, 10))

sns.heatmap(
    transition,
    xticklabels=LABELS,
    yticklabels=LABELS,
    cmap="Blues"
)

plt.title(
    "BIO Transition Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "10_transition_matrix.png"
    ),
    dpi=300
)

plt.close()

# ==========================
# 输出文本报告
# ==========================

with open(
    os.path.join(
        OUTPUT_DIR,
        "report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

print("\n分析完成")
print(f"输出目录: {OUTPUT_DIR}")
