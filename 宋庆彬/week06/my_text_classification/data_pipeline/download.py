"""下载 TNEWS 数据集，保存为本地 JSON"""

import json
import sys
from pathlib import Path

# 将项目根目录加入 path，方便从子目录直接运行
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from common.config import DATA_DIR, LABEL_NAMES, NUM_LABELS

LABEL_CODE_TO_NAME = {
    "100": "故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经",
    "106": "房产", "107": "汽车", "108": "教育", "109": "科技", "110": "军事",
    "112": "旅游", "113": "国际", "114": "证券", "115": "农业", "116": "电竞",
}


def build_label_map(features):
    label_names = features["label"].names
    id2code = {i: code for i, code in enumerate(label_names)}
    id2name = {i: LABEL_CODE_TO_NAME[code] for i, code in id2code.items()}
    return {
        "id2code": id2code,
        "id2name": id2name,
        "code2id": {v: k for k, v in id2code.items()},
        "name2id": {v: k for k, v in id2name.items()},
        "num_labels": len(label_names),
    }


def save_split(dataset_split, path, split_name):
    records = [{"idx": item["idx"], "sentence": item["sentence"], "label": item["label"]} for item in dataset_split]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  {split_name}: {len(records)} 条 → {path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("正在下载 TNEWS ...")
    ds = load_dataset("clue", "tnews")

    label_map = build_label_map(ds["train"].features)
    with open(DATA_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"label_map → {DATA_DIR / 'label_map.json'}，类别数: {label_map['num_labels']}")
    for i in range(NUM_LABELS):
        print(f"  {i:2d} | {label_map['id2code'][i]} | {LABEL_NAMES[i]}")

    print("\n保存数据集分割 ...")
    save_split(ds["train"], DATA_DIR / "train.json", "train")
    save_split(ds["validation"], DATA_DIR / "val.json", "val")
    save_split(ds["test"], DATA_DIR / "test.json", "test")
    print("\n下载完成。")


if __name__ == "__main__":
    main()
