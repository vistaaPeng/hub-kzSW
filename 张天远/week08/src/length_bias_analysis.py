"""
Length Bias 分析 — 检测模型是否用句子长度差当"相似"捷径

教学重点：
  1. 如果正例对长度差显著小于负例对 → 数据存在 length bias
  2. 按长度差分桶评估 → 如果 F1 随长度差增大而下降 → 模型偷学捷径
  3. AFQMC 无 bias（对照），LCQMC 有 bias（验证目标）

使用方式：
  python src/length_bias_analysis.py
  python src/length_bias_analysis.py --data_dir data/lcqmc
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import f1_score, accuracy_score

from model import BiEncoder
from dataset import PairDataset, load_jsonl

ROOT      = Path(__file__).parent.parent
BERT_PATH = "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
LOG_DIR   = ROOT / "outputs" / "logs"


def evaluate_by_length_bucket(model, tokenizer, data_dir, device, max_length=64):
    """按 |len(s1)-len(s2)| 分桶评估 F1"""
    val_rows = load_jsonl(data_dir / "validation.jsonl")

    # 分桶
    buckets = defaultdict(list)  # (min_diff, max_diff) -> [(sim, label), ...]
    for r in val_rows:
        diff = abs(len(r["sentence1"]) - len(r["sentence2"]))
        # 桶边界：0-1, 2-3, 4-6, 7+
        if diff <= 1:
            key = (0, 1)
        elif diff <= 3:
            key = (2, 3)
        elif diff <= 6:
            key = (4, 6)
        else:
            key = (7, 999)

        enc_a = tokenizer(r["sentence1"], max_length=max_length, truncation=True,
                          padding="max_length", return_tensors="pt")
        enc_b = tokenizer(r["sentence2"], max_length=max_length, truncation=True,
                          padding="max_length", return_tensors="pt")

        with torch.no_grad():
            emb_a = model.encode(
                enc_a["input_ids"].to(device),
                enc_a["attention_mask"].to(device),
                enc_a["token_type_ids"].to(device))
            emb_b = model.encode(
                enc_b["input_ids"].to(device),
                enc_b["attention_mask"].to(device),
                enc_b["token_type_ids"].to(device))
            sim = F.cosine_similarity(emb_a, emb_b, dim=-1).item()

        buckets[key].append({"sim": sim, "label": r["label"],
                             "diff": diff,
                             "s1": r["sentence1"], "s2": r["sentence2"]})

    return dict(buckets)


def find_best_threshold(samples):
    """网格搜索最优阈值"""
    sims = np.array([s["sim"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    best_f1, best_thr = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr), best_f1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    ds_name = data_dir.name

    # 加载对应数据集 checkpoint
    ckpt_candidates = [
        CKPT_DIR / f"biencoder_cosine_{ds_name}_best.pt",
        CKPT_DIR / "biencoder_cosine_best.pt",
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        print(f"❌ 未找到 {ds_name} 的 BiEncoder checkpoint")
        return

    print(f"数据集: {ds_name}")
    print(f"Checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BiEncoder(BERT_PATH, pool=ckpt.get("args", {}).get("pool", "mean"),
                      num_hidden_layers=ckpt.get("args", {}).get("num_hidden_layers", 4)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    print("\n按长度差分桶评估...")
    buckets = evaluate_by_length_bucket(model, tokenizer, data_dir, device)

    # ── 输出 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Length Bias 分析 — {ds_name}")
    print(f"{'='*60}")
    print(f"{'长度差桶':<15} {'样本数':>7} {'F1(weighted)':>14} {'阈值':>8}")
    print(f"{'-'*48}")

    results = []
    for (lo, hi) in sorted(buckets.keys()):
        samples = buckets[(lo, hi)]
        n = len(samples)
        thr, f1 = find_best_threshold(samples)
        range_str = f"{lo}-{hi}" if hi < 999 else f"7+"
        print(f"  {range_str:<13} {n:>7,} {f1:>14.4f} {thr:>8.2f}")
        results.append({"range": range_str, "n": n, "f1": round(f1, 4),
                        "threshold": round(thr, 2)})

    # 趋势判断
    f1s = [r["f1"] for r in results]
    delta = f1s[-1] - f1s[0] if len(f1s) >= 2 else 0
    print(f"\n  F1(最短差) - F1(最长差) = {f1s[0]:.4f} - {f1s[-1]:.4f} = {delta:+.4f}")
    if delta < -0.03:
        print("  ⚠️ 存在 length bias：长度差越大 F1 越低，模型在偷学捷径")
    else:
        print("  ✅ 无明显 length bias：F1 不随长度差波动")

    # 保存
    log_path = LOG_DIR / f"length_bias_{ds_name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": ds_name, "buckets": results,
                   "delta_f1": round(delta, 4)}, f, ensure_ascii=False, indent=2)
    print(f"\n结果 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Length Bias 分析")
    p.add_argument("--data_dir", default="data/lcqmc", type=str,
                   help="数据目录（默认: data/lcqmc）")
    return p.parse_args()


if __name__ == "__main__":
    main()
