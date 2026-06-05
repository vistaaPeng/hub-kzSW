#!/usr/bin/env python3
"""三方方案对比：终端表格 + 柱状图"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt

from common.config import OUTPUT_DIR, FIG_DIR
from common.utils import setup_plot_font


def load_bert_acc():
    log = OUTPUT_DIR / "train_log_cls.json"
    if not log.exists():
        return None
    with open(log, encoding="utf-8") as f:
        records = json.load(f)
    return max(r["val_acc"] for r in records)


def load_zero_shot_acc():
    p = OUTPUT_DIR / "llm_zero_shot_results.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["accuracy"]


def load_sft_acc():
    p = OUTPUT_DIR / "llm_sft_results.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["accuracy"]


def main():
    setup_plot_font()

    results = {
        "BERT Fine-tune": load_bert_acc(),
        "LLM Zero-shot": load_zero_shot_acc(),
        "LLM SFT (LoRA)": load_sft_acc(),
    }

    # 终端表格
    print(f"\n{'='*50}")
    print(f"{'方案':<25} {'准确率':>8}")
    print(f"{'-'*50}")
    for name, acc in results.items():
        val_str = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"{name:<25} {val_str:>8}")
    print(f"{'='*50}")

    # 柱状图
    valid = {k: v for k, v in results.items() if v is not None}
    if len(valid) >= 2:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        bars = plt.bar(valid.keys(), valid.values(), color=["#4C72B0", "#DD8452", "#55A868"][:len(valid)])
        for bar, val in zip(bars, valid.values()):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
        plt.title("TNEWS 文本分类 — 三方案准确率对比")
        plt.ylabel("Accuracy")
        plt.ylim(0, max(valid.values()) * 1.2)
        plt.tight_layout()
        chart_path = FIG_DIR / "compare_accuracy.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\n对比柱状图 → {chart_path}")
    else:
        print("\n提示: 至少需要运行两个方案才能生成柱状图。")
        print("  BERT:    cd bert_finetune && python train.py")
        print("  Zero-shot: cd llm_zero_shot && python classify.py")
        print("  SFT:     cd llm_sft && python train.py && python evaluate.py")


if __name__ == "__main__":
    main()
