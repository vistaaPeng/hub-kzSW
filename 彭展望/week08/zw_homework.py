"""
第八周作业：在 LCQMC 和 BQ Corpus 数据集上比较三种文本匹配方法

对比方法：
  1. BiEncoder + CosineEmbeddingLoss（表示型，直接优化余弦相似度）
  2. BiEncoder + TripletLoss（表示型，三元组相对距离约束）
  3. CrossEncoder + CrossEntropyLoss（交互型，两句完全交互）

数据集说明：
  LCQMC    : 生活口语问答相似性，238K 训练 / 8.8K 验证
  BQ Corpus: 银行金融领域问句匹配，68.9K 训练 / 4.5K 验证

使用方式：
  python zw_homework.py                              # 训练全部两个数据集
  python zw_homework.py --datasets lcqmc             # 只训练 LCQMC
  python zw_homework.py --datasets bq_corpus         # 只训练 BQ Corpus
  python zw_homework.py --max_samples 10000          # 限制训练集大小（快速验证）
  python zw_homework.py --epochs 1 --max_samples 5000
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import (
    PairDataset,
    TripletDataset,
    CrossEncoderDataset,
)
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder

# ── 路径 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

def _find_bert_path():
    ai_root = ROOT.parent.parent.parent  # 2_AI大模型学习
    candidates = [
        ROOT.parent.parent / "pretrain_models" / "bert-base-chinese",
        ai_root / "week4语言模型(1)" / "bert-base-chinese",
        ai_root / "pretrain_models" / "bert-base-chinese",
    ]
    for p in candidates:
        if (p / "vocab.txt").exists():
            return p
    raise FileNotFoundError(
        "找不到 bert-base-chinese，请用 --bert_path 指定路径\n"
        f"已搜索: {candidates}"
    )

BERT_PATH = _find_bert_path()


# ── 工具：限制训练集大小 ───────────────────────────────────────────────────

def maybe_subset(dataset, max_samples):
    if max_samples > 0 and len(dataset) > max_samples:
        indices = random.sample(range(len(dataset)), max_samples)
        return Subset(dataset, indices)
    return dataset


# ── BiEncoder 训练（CosineEmbeddingLoss） ─────────────────────────────────

def train_biencoder_cosine(data_dir, ckpt_path, log_path, args, device, tokenizer, bert_path=None):
    print(f"\n{'='*60}")
    print(f"[BiEncoder-Cosine] 数据集: {Path(data_dir).name}")
    print(f"{'='*60}")

    train_ds = PairDataset(Path(data_dir) / "train.jsonl", tokenizer, args.max_length)
    val_ds   = PairDataset(Path(data_dir) / "validation.jsonl", tokenizer, args.max_length)
    train_ds = maybe_subset(train_ds, args.max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  train: {len(train_ds):,}条  val: {len(val_ds):,}条")

    model = build_biencoder(str(bert_path or BERT_PATH), pool="mean", num_hidden_layers=args.layers).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_f1, log_records = 0.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            ba = {k.replace("_a", ""): v.to(device) for k, v in batch.items() if k.endswith("_a")}
            bb = {k.replace("_b", ""): v.to(device) for k, v in batch.items() if k.endswith("_b")}
            labels = batch["label"].to(device)
            emb_a, emb_b = model(ba, bb)
            loss = F.cosine_embedding_loss(emb_a, emb_b, labels.float() * 2 - 1, margin=0.3)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item() * labels.size(0); n += labels.size(0)

        val_m = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | loss={total_loss/n:.4f} | "
              f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f} "
              f"threshold={val_m['threshold']:.2f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": total_loss/n,
                             "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                             "threshold": val_m["threshold"], "elapsed_s": elapsed})
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({"state_dict": model.state_dict(), "threshold": val_m["threshold"],
                        "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                        "args": {"pool": "mean", "num_hidden_layers": args.layers,
                                 "max_length": args.max_length}}, ckpt_path)
            print(f"  ✓ 保存最优 → val_f1={best_f1:.4f}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"  最优 val_f1={best_f1:.4f}  checkpoint → {ckpt_path}")
    return best_f1


# ── BiEncoder 训练（TripletLoss） ─────────────────────────────────────────

def train_biencoder_triplet(data_dir, ckpt_path, log_path, args, device, tokenizer, bert_path=None):
    print(f"\n{'='*60}")
    print(f"[BiEncoder-Triplet] 数据集: {Path(data_dir).name}")
    print(f"{'='*60}")

    train_ds = TripletDataset(Path(data_dir) / "train.jsonl", tokenizer, args.max_length)
    val_ds   = PairDataset(Path(data_dir) / "validation.jsonl", tokenizer, args.max_length)
    train_ds = maybe_subset(train_ds, args.max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  train(triplet): {len(train_ds):,}条  val: {len(val_ds):,}条")

    model = build_biencoder(str(bert_path or BERT_PATH), pool="mean", num_hidden_layers=args.layers).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_f1, log_records = 0.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            ea = {k.replace("_a", ""): v.to(device) for k, v in batch.items() if k.endswith("_a")}
            ep = {k.replace("_p", ""): v.to(device) for k, v in batch.items() if k.endswith("_p")}
            en = {k.replace("_n", ""): v.to(device) for k, v in batch.items() if k.endswith("_n")}
            emb_a = model.encode(**ea)
            emb_p = model.encode(**ep)
            emb_n = model.encode(**en)
            loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=0.3)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item() * emb_a.size(0); n += emb_a.size(0)

        val_m = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | loss={total_loss/n:.4f} | "
              f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f} "
              f"threshold={val_m['threshold']:.2f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": total_loss/n,
                             "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                             "threshold": val_m["threshold"], "elapsed_s": elapsed})
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({"state_dict": model.state_dict(), "threshold": val_m["threshold"],
                        "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                        "args": {"pool": "mean", "num_hidden_layers": args.layers,
                                 "max_length": args.max_length}}, ckpt_path)
            print(f"  ✓ 保存最优 → val_f1={best_f1:.4f}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"  最优 val_f1={best_f1:.4f}  checkpoint → {ckpt_path}")
    return best_f1


# ── CrossEncoder 训练 ─────────────────────────────────────────────────────

def train_crossencoder(data_dir, ckpt_path, log_path, args, device, tokenizer, bert_path=None):
    print(f"\n{'='*60}")
    print(f"[CrossEncoder] 数据集: {Path(data_dir).name}")
    print(f"{'='*60}")

    train_ds = CrossEncoderDataset(Path(data_dir) / "train.jsonl", tokenizer, 128)
    val_ds   = CrossEncoderDataset(Path(data_dir) / "validation.jsonl", tokenizer, 128)
    train_ds = maybe_subset(train_ds, args.max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  train: {len(train_ds):,}条  val: {len(val_ds):,}条")

    model = build_crossencoder(str(bert_path or BERT_PATH), num_hidden_layers=args.layers).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_f1, log_records = 0.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, n = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss    += loss.item() * labels.size(0)
            total_correct += (logits.argmax(-1) == labels).sum().item()
            n             += labels.size(0)

        val_m = eval_crossencoder(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} | loss={total_loss/n:.4f} train_acc={total_correct/n:.4f} | "
              f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": total_loss/n,
                             "train_acc": total_correct/n,
                             "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                             "elapsed_s": elapsed})
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({"state_dict": model.state_dict(),
                        "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                        "args": {"num_hidden_layers": args.layers}}, ckpt_path)
            print(f"  ✓ 保存最优 → val_f1={best_f1:.4f}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"  最优 val_f1={best_f1:.4f}  checkpoint → {ckpt_path}")
    return best_f1


# ── 单数据集实验（训练三种方法并对比） ────────────────────────────────────

def run_dataset(dataset_name, args, device, tokenizer, bert_path=None):
    data_dir  = ROOT / "data" / dataset_name
    ckpt_dir  = ROOT / "outputs" / "checkpoints" / dataset_name
    log_dir   = ROOT / "outputs" / "logs" / dataset_name
    fig_dir   = ROOT / "outputs" / "figures" / dataset_name
    for d in (ckpt_dir, log_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. BiEncoder + CosineEmbeddingLoss
    f1 = train_biencoder_cosine(
        data_dir,
        ckpt_dir / "biencoder_cosine_best.pt",
        log_dir  / "biencoder_cosine_log.json",
        args, device, tokenizer, bert_path,
    )
    results.append({
        "key": "biencoder_cosine",
        "label": "BiEncoder\n(CosineEmbeddingLoss)",
        "color": "#2196F3",
        "type": "biencoder",
        "val_f1": f1,
    })

    # 2. BiEncoder + TripletLoss
    f1 = train_biencoder_triplet(
        data_dir,
        ckpt_dir / "biencoder_triplet_best.pt",
        log_dir  / "biencoder_triplet_log.json",
        args, device, tokenizer, bert_path,
    )
    results.append({
        "key": "biencoder_triplet",
        "label": "BiEncoder\n(TripletLoss)",
        "color": "#4CAF50",
        "type": "biencoder",
        "val_f1": f1,
    })

    # 3. CrossEncoder + CrossEntropyLoss
    f1 = train_crossencoder(
        data_dir,
        ckpt_dir / "crossencoder_best.pt",
        log_dir  / "crossencoder_log.json",
        args, device, tokenizer, bert_path,
    )
    results.append({
        "key": "crossencoder",
        "label": "CrossEncoder\n(CrossEntropyLoss)",
        "color": "#FF9800",
        "type": "crossencoder",
        "val_f1": f1,
    })

    # ── 对比汇总 ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"【{dataset_name} 结果汇总】")
    print(f"{'方法':<35} {'val_F1':>8}")
    print(f"{'-'*45}")
    for r in results:
        print(f"  {r['key']:<33} {r['val_f1']:>8.4f}")

    best = max(results, key=lambda x: x["val_f1"])
    print(f"\n  最高 F1: {best['key']} ({best['val_f1']:.4f})")

    bi = [r for r in results if r["type"] == "biencoder"]
    if len(bi) == 2:
        delta = bi[1]["val_f1"] - bi[0]["val_f1"]
        tag = "TripletLoss 更优" if delta > 0.005 else (
              "CosineEmbeddingLoss 更优" if delta < -0.005 else "两种 Loss 差距不大")
        print(f"  Cosine vs Triplet Δ={delta:+.4f}  → {tag}")

    # 保存汇总
    log_path = log_dir / "comparison.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  汇总日志 → {log_path}")

    # 柱状图
    _plot_bar(results, dataset_name,
              fig_dir / "method_comparison_bar.png", args)

    return results


# ── 可视化 ────────────────────────────────────────────────────────────────

def _plot_bar(results, dataset_name, save_path, args):
    names  = [r["label"]   for r in results]
    f1s    = [r["val_f1"]  for r in results]
    colors = [r["color"]   for r in results]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, f1s, 0.5, color=colors, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("val F1 (weighted)")
    ax.set_title(f"Method Comparison on {dataset_name}\n"
                 f"(4-layer BERT, {args.epochs} epoch, max_samples={args.max_samples})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表 → {save_path}")


def _plot_cross_dataset(all_results, save_path):
    """跨数据集对比：同一方法在不同数据集上的 F1"""
    methods = ["biencoder_cosine", "biencoder_triplet", "crossencoder"]
    colors  = {"biencoder_cosine": "#2196F3",
               "biencoder_triplet": "#4CAF50",
               "crossencoder": "#FF9800"}

    datasets = list(all_results.keys())
    x = np.arange(len(datasets))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(methods):
        f1s = [next((r["val_f1"] for r in all_results[ds] if r["key"] == method), 0)
               for ds in datasets]
        bars = ax.bar(x + (i - 1) * w, f1s, w, label=method, color=colors[method], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("val F1 (weighted)")
    ax.set_title("Cross-Dataset Method Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  跨数据集对比图 → {save_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="第八周作业：多数据集文本匹配方法比较")
    parser.add_argument("--datasets",     nargs="+",
                        default=["lcqmc", "bq_corpus"],
                        help="要实验的数据集名（空格分隔），可选: lcqmc bq_corpus")
    parser.add_argument("--epochs",       default=1,     type=int,
                        help="每个方法训练 epoch 数（1 epoch 已能反映差异）")
    parser.add_argument("--batch_size",   default=64,    type=int)
    parser.add_argument("--max_length",   default=64,    type=int,
                        help="BiEncoder 单句最大 token 数")
    parser.add_argument("--layers",       default=4,     type=int,
                        help="BERT 层数（4 层快速验证；12 层全量精度）")
    parser.add_argument("--lr",           default=2e-5,  type=float)
    parser.add_argument("--max_samples",  default=-1,    type=int,
                        help="限制训练集大小（-1 = 全量，>0 = 随机抽样）")
    parser.add_argument("--bert_path",    default=None,  type=str,
                        help="bert-base-chinese 路径（默认自动查找）")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"设备: {device}")
    print(f"BERT 层数: {args.layers}  Epochs: {args.epochs}  "
          f"batch_size: {args.batch_size}  max_samples: {args.max_samples}")

    bert_path = Path(args.bert_path) if args.bert_path else BERT_PATH
    print(f"BERT 路径: {bert_path}")
    tokenizer = BertTokenizer.from_pretrained(str(bert_path))

    all_results = {}
    for ds_name in args.datasets:
        data_path = ROOT / "data" / ds_name
        if not data_path.exists():
            print(f"\n[跳过] 数据集目录不存在: {data_path}")
            continue
        t_start = time.time()
        print(f"\n\n{'#'*65}")
        print(f"# 数据集: {ds_name}")
        print(f"{'#'*65}")
        results = run_dataset(ds_name, args, device, tokenizer, bert_path)
        all_results[ds_name] = results
        print(f"\n[{ds_name}] 全部训练完成，耗时 {(time.time()-t_start)/60:.1f} 分钟")

    # 跨数据集对比图（仅当训练了多个数据集时）
    if len(all_results) >= 2:
        fig_dir = ROOT / "outputs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_cross_dataset(all_results, fig_dir / "cross_dataset_comparison.png")

    # 最终汇总
    print(f"\n\n{'='*65}")
    print("【全部实验结果汇总】")
    print(f"{'='*65}")
    for ds_name, results in all_results.items():
        print(f"\n{ds_name}:")
        for r in results:
            print(f"  {r['key']:<35} val_F1 = {r['val_f1']:.4f}")

    print("\n作业完成！图表保存在 outputs/figures/ 目录下。")


if __name__ == "__main__":
    main()
