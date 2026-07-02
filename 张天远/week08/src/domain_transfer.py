"""
领域迁移评估 — AFQMC ↔ LCQMC 跨数据集泛化能力

教学重点：
  1. 模型是在"学语义"还是"背领域"——看跨数据集 F1 下降多少
  2. 金融领域(AFQMC)训练 → 开放域(LCQMC)预测 → 预期大幅下降
  3. 反过来看 LCQMC→AFQMC 的迁移能力

实验设计：
  加载 AFQMC checkpoint → 评估 LCQMC val
  加载 LCQMC checkpoint → 评估 AFQMC val
  对比同数据集内的 F1（作为上界）

使用方式：
  python src/domain_transfer.py
  python src/domain_transfer.py --datasets afqmc lcqmc
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import BertTokenizer

from model import BiEncoder
from dataset import PairDataset
from evaluate import eval_biencoder

ROOT      = Path(__file__).parent.parent
BERT_PATH = "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
LOG_DIR   = ROOT / "outputs" / "logs"


def load_biencoder(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = BiEncoder(BERT_PATH, pool=saved_args.get("pool", "mean"),
                      num_hidden_layers=saved_args.get("num_hidden_layers", 4)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def evaluate_cross_dataset(model, tokenizer, data_path, device, batch_size=64):
    ds = PairDataset(data_path, tokenizer, max_length=64)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = eval_biencoder(model, loader, device)
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    print(f"设备: {device}")
    print(f"数据集: {args.datasets}")
    print(f"{'='*55}")

    results = []

    # 对每对 (源数据集, 目标数据集)
    for src in args.datasets:
        for tgt in args.datasets:
            ckpt_name = f"biencoder_cosine_{src}_best.pt" if src != "afqmc" else "biencoder_cosine_best.pt"
            ckpt_paths = [
                CKPT_DIR / ckpt_name,
                CKPT_DIR / f"biencoder_cosine_{src}_core.pt",
                CKPT_DIR / f"biencoder_cosine_{src}_best.pt",
            ]

            ckpt_path = None
            for p in ckpt_paths:
                if p.exists():
                    ckpt_path = p
                    break
            if ckpt_path is None:
                print(f"\n  [SKIP] {src} checkpoint 不存在")
                continue

            data_path = ROOT / "data" / tgt / "validation.jsonl"
            if not data_path.exists():
                print(f"\n  [SKIP] {tgt} val 数据不存在: {data_path}")
                continue

            tag = "同域" if src == tgt else "跨域"
            print(f"\n  [{tag}] {src} → {tgt}")
            print(f"    checkpoint: {ckpt_path.name}")

            model = load_biencoder(ckpt_path, device)
            metrics = evaluate_cross_dataset(model, tokenizer, data_path, device, args.batch_size)

            print(f"    Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
                  f"threshold={metrics['threshold']:.2f}")

            results.append({
                "train_dataset": src, "eval_dataset": tgt,
                "cross_domain": src != tgt,
                "accuracy": round(metrics['accuracy'], 4),
                "f1": round(metrics['f1'], 4),
                "threshold": round(metrics['threshold'], 2),
            })

    # ── 对比表 ──────────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*55}")
        print("领域迁移 对比矩阵")
        print(f"{'Train ↓ / Eval →':<20} ", end="")
        for ds in args.datasets:
            print(f"{ds:>15}", end="")
        print()
        print(f"{'-'*50}")
        for src in args.datasets:
            print(f"{src:<20} ", end="")
            for tgt in args.datasets:
                vals = [r for r in results if r["train_dataset"] == src and r["eval_dataset"] == tgt]
                if vals:
                    print(f"{vals[0]['f1']:>15.4f}", end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()

        # 计算迁移损失
        same_domain = [r for r in results if not r["cross_domain"]]
        cross_domain = [r for r in results if r["cross_domain"]]
        if same_domain and cross_domain:
            avg_same = sum(r["f1"] for r in same_domain) / len(same_domain)
            avg_cross = sum(r["f1"] for r in cross_domain) / len(cross_domain)
            print(f"\n  同域 F1 均值: {avg_same:.4f}")
            print(f"  跨域 F1 均值: {avg_cross:.4f}")
            print(f"  迁移损失 (Δ): {avg_same - avg_cross:.4f}")

        # 保存
        log_path = LOG_DIR / "domain_transfer.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="领域迁移评估")
    p.add_argument("--datasets", nargs="+", default=["afqmc", "lcqmc"],
                   help="数据集列表（默认: afqmc lcqmc）")
    p.add_argument("--batch_size", default=64, type=int)
    return p.parse_args()


if __name__ == "__main__":
    main()
