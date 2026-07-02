"""
错误分析：BiEncoder Cosine 在 AFQMC val 上的全面诊断
=====================================================
输出：
  1. 混淆矩阵 + FP/FN 数量
  2. 按相似度分桶的错误率（哪个区间最易错？）
  3. Top FP / Top FN 案例抽样
  4. 按长度差分桶的错误率
  5. 与 CrossEncoder 的错误重叠分析

用法：
  python src/error_analysis.py
  python src/error_analysis.py --data_dir data/bq_corpus
"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report

from model import build_biencoder, build_crossencoder
from dataset import PairDataset, load_jsonl
from evaluate import eval_biencoder, eval_crossencoder, _find_best_threshold

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"
BERT_PATH  = "bert-base-chinese"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
LOG_DIR    = ROOT / "outputs" / "logs"
FIG_DIR    = ROOT / "outputs" / "figures"


def analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    data_path = Path(args.data_dir) / "validation.jsonl"
    rows = load_jsonl(data_path)
    print(f"数据: {len(rows):,} 对")

    # ── 加载 BiEncoder ───────────────────────────────────────────────────
    bi_ckpt = torch.load(CKPT_DIR / "biencoder_cosine_best.pt", map_location=device, weights_only=False)
    bi_model = build_biencoder(args.bert_path,
                               pool=bi_ckpt.get("args", {}).get("pool", "mean"),
                               num_hidden_layers=bi_ckpt.get("args", {}).get("num_hidden_layers", 4)).to(device)
    bi_model.load_state_dict(bi_ckpt["state_dict"])
    bi_model.eval()

    # ── BiEncoder 评估 ───────────────────────────────────────────────────
    ds = PairDataset(data_path, tokenizer, max_length=64)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    metrics = eval_biencoder(bi_model, loader, device)
    sims = np.array(metrics["similarities"])
    labels = np.array(metrics["labels"])
    threshold = metrics["threshold"]

    preds = (sims >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"\n{'='*55}")
    print(f"BiEncoder Cosine 错误分析")
    print(f"  阈值: {threshold:.3f}  |  F1: {metrics['f1']:.4f}  |  Acc: {metrics['accuracy']:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  正例数: {labels.sum()}  负例数: {(labels==0).sum()}")
    print(f"  FP 率 (误报): {fp/(fp+tn)*100:.1f}%   FN 率 (漏报): {fn/(fn+tp)*100:.1f}%")

    # ── 1. 按相似度分桶的错误率 ──────────────────────────────────────────
    print(f"\n── 1. 按相似度分桶的错误率 ──")
    buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0)]
    rows_all = list(zip(sims, labels, preds, [r["sentence1"] for r in rows], [r["sentence2"] for r in rows]))
    for lo, hi in buckets:
        in_bucket = [r for r in rows_all if lo <= r[0] < hi]
        if not in_bucket: continue
        errs = [r for r in in_bucket if r[1] != r[2]]
        err_rate = len(errs) / len(in_bucket) * 100
        bar = "█" * int(err_rate / 2)
        print(f"  [{lo:.1f}, {hi:.1f}): {len(in_bucket):5d} 对  错误 {len(errs):4d} ({err_rate:5.1f}%) {bar}")

    # ── 2. Top FP / FN 案例 ──────────────────────────────────────────────
    print(f"\n── 2. Top 误报 (FP) ──")
    fp_cases = [(s, s1, s2) for s, l, p, s1, s2 in rows_all if l == 0 and p == 1]
    fp_cases.sort(key=lambda x: x[0], reverse=True)  # 最高相似度的 FP
    for i, (sim, s1, s2) in enumerate(fp_cases[:5]):
        print(f"  #{i+1} sim={sim:.3f}")
        print(f"    s1: {s1[:60]}")
        print(f"    s2: {s2[:60]}")

    print(f"\n── 2. Top 漏报 (FN) ──")
    fn_cases = [(s, s1, s2) for s, l, p, s1, s2 in rows_all if l == 1 and p == 0]
    fn_cases.sort(key=lambda x: x[0])  # 最低相似度的 FN
    for i, (sim, s1, s2) in enumerate(fn_cases[:5]):
        print(f"  #{i+1} sim={sim:.3f}")
        print(f"    s1: {s1[:60]}")
        print(f"    s2: {s2[:60]}")

    # ── 3. 按长度差分桶的错误率 ──────────────────────────────────────────
    print(f"\n── 3. 按长度差分桶的错误率 ──")
    len_buckets = [(0, 1), (2, 3), (4, 6), (7, 99)]
    for lo, hi in len_buckets:
        in_bucket = [r for r in rows_all if lo <= abs(len(r[3]) - len(r[4])) <= hi]
        if not in_bucket: continue
        errs = [r for r in in_bucket if r[1] != r[2]]
        err_rate = len(errs) / len(in_bucket) * 100
        label = f"{lo}-{hi}字" if hi < 99 else f"{lo}+字"
        print(f"  {label:8s}: {len(in_bucket):5d} 对  错误 {len(errs):4d} ({err_rate:5.1f}%)")

    # ── 4. 与 CrossEncoder 错误重叠（如有） ──────────────────────────────
    ce_ckpt_path = CKPT_DIR / "crossencoder_best.pt"
    if ce_ckpt_path.exists():
        print(f"\n── 4. Bi vs CrossEncoder 错误重叠 ──")
        ce_ckpt = torch.load(ce_ckpt_path, map_location=device, weights_only=False)
        ce_model = build_crossencoder(args.bert_path,
                                      num_hidden_layers=ce_ckpt.get("args", {}).get("num_hidden_layers", 4)).to(device)
        ce_model.load_state_dict(ce_ckpt["state_dict"])
        ce_model.eval()

        bi_errs = set(i for i, r in enumerate(rows_all) if r[1] != r[2])
        ce_preds_list = []
        for i in range(0, len(rows_all), args.batch_size):
            batch = rows_all[i:i+args.batch_size]
            s1s = [r[3] for r in batch]
            s2s = [r[4] for r in batch]
            enc = tokenizer(s1s, s2s, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = ce_model(**enc)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            ce_preds_list.extend(preds)

        ce_errs = set(i for i, (r, p) in enumerate(zip(rows_all, ce_preds_list)) if r[1] != p)
        both_err = bi_errs & ce_errs
        bi_only = bi_errs - ce_errs
        ce_only = ce_errs - bi_errs

        print(f"  Bi 错误: {len(bi_errs)}   CE 错误: {len(ce_errs)}")
        print(f"  两者都错: {len(both_err)}  仅 Bi 错: {len(bi_only)}  仅 CE 错: {len(ce_only)}")
        if len(bi_errs) > 0:
            print(f"  CE 纠正了 Bi 的 {len(bi_only)}/{len(bi_errs)} ({len(bi_only)/len(bi_errs)*100:.0f}%) 个错误")
    else:
        print(f"\n── 4. CrossEncoder checkpoint 不存在，跳过错误重叠分析 ──")

    # ── 保存 ─────────────────────────────────────────────────────────────
    log_path = LOG_DIR / f"error_analysis_{Path(args.data_dir).name}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(args.data_dir),
            "threshold": float(threshold),
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
            "fp_rate": round(fp/(fp+tn)*100, 1) if (fp+tn) > 0 else 0,
            "fn_rate": round(fn/(fn+tp)*100, 1) if (fn+tp) > 0 else 0,
            "top_fp": [{"sim": round(float(s), 3), "s1": s1, "s2": s2} for s, s1, s2 in fp_cases[:10]],
            "top_fn": [{"sim": round(float(s), 3), "s1": s1, "s2": s2} for s, s1, s2 in fn_cases[:10]],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bert_path",  default=BERT_PATH)
    p.add_argument("--data_dir",   default=str(DATA_DIR))
    p.add_argument("--batch_size", default=64, type=int)
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
