"""数据分析脚本：分析peoples_daily数据集的整体情况"""
import json
from collections import Counter, defaultdict

# 加载数据
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_data("data/peoples_daily/train.json")
val_data = load_data("data/peoples_daily/validation.json")
test_data = load_data("data/peoples_daily/test.json")

print("=" * 60)
print("人民日报NER数据集分析报告")
print("=" * 60)

# 1. 数据量统计
print("\n【1. 数据量统计】")
print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")
print(f"测试集样本数: {len(test_data)}")

# 2. 数据格式分析
print("\n【2. 数据格式分析】")
sample = train_data[0]
print(f"单条数据类型: {type(sample)}")
if isinstance(sample, dict):
    print(f"单条数据键: {list(sample.keys())}")
print(f"示例数据: {json.dumps(sample, ensure_ascii=False)[:500]}")

# 3. 标签分布统计
label_names = json.load(open("data/peoples_daily/label_names.json", "r", encoding="utf-8"))
print(f"\n【3. 标签类别】")
print(f"标签列表: {label_names}")
print(f"标签数量: {len(label_names)}")

def analyze_labels(data, dataset_name):
    label_counter = Counter()
    entity_counter = Counter()
    text_lengths = []

    for item in data:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        labels = item.get("labels", item.get("label", item.get("entities", None)))

        text_lengths.append(len(text))

        if isinstance(labels, list):
            if len(labels) > 0 and isinstance(labels[0], dict):
                for ent in labels:
                    ent_type = ent.get("entity", ent.get("type", "UNKNOWN"))
                    entity_counter[ent_type] += 1
            elif len(labels) > 0 and isinstance(labels[0], str):
                for label in labels:
                    label_counter[label] += 1
                    if label.startswith("B-"):
                        entity_type = label[2:]
                        entity_counter[entity_type] += 1

    return label_counter, entity_counter, text_lengths

# 分析各数据集
for name, data in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
    label_counter, entity_counter, text_lengths = analyze_labels(data, name)
    print(f"\n--- {name} ---")
    print(f"  样本数: {len(data)}")
    print(f"  平均文本长度: {sum(text_lengths)/len(text_lengths):.1f}")
    print(f"  最大文本长度: {max(text_lengths)}")
    print(f"  最小文本长度: {min(text_lengths)}")
    if entity_counter:
        print(f"  实体类型分布:")
        for ent_type, count in entity_counter.most_common():
            print(f"    {ent_type}: {count}")
    if label_counter:
        print(f"  标签分布:")
        for label, count in label_counter.most_common():
            print(f"    {label}: {count}")

# 4. 查看更多示例
print("\n【4. 数据示例】")
for i in range(min(3, len(train_data))):
    item = train_data[i]
    print(f"\n示例 {i+1}:")
    if isinstance(item, dict):
        for k, v in item.items():
            if isinstance(v, str):
                print(f"  {k}: {v[:100]}")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                print(f"  {k}: {json.dumps(v[:3], ensure_ascii=False)}")
            else:
                print(f"  {k}: {str(v)[:200]}")
    else:
        print(f"  {str(item)[:200]}")
