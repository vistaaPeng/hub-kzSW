"""
第六周作业：对比文本分类不同训练方法效果

对比方案：
  1. BERT fine-tune（CLS 池化，普通 loss）
  2. BERT fine-tune（CLS 池化，加权 loss，处理类别不均衡）
  3. LLM Zero-shot（Qwen2-0.5B，无需训练）
  4. LLM SFT + LoRA（Qwen2-0.5B 指令微调）

使用方式：
  python compare_methods.py

结果读取自项目 outputs/ 目录，无需额外训练。
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

matplotlib.rcParams["axes.unicode_minus"] = False


def _find_chinese_font():
    candidates = ["PingFang SC", "STHeiti", "Hiragino Sans GB",
                  "Arial Unicode MS", "SimHei", "Microsoft YaHei",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT

PROJ_DIR = Path(__file__).parent.parent / "week6 文本分类问题" / "text_classification项目"
OUT_DIR  = PROJ_DIR / "outputs"
FIG_DIR  = OUT_DIR / "figures"


def load_bert_log(name: str) -> list[dict]:
    path = OUT_DIR / f"train_log_{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_llm_results(name: str) -> dict:
    path = OUT_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_summary(bert_cls, bert_weighted, zero_shot, sft):
    print("=" * 70)
    print(f"{'第六周作业：不同训练方法效果对比':^70}")
    print("=" * 70)

    # ── BERT 训练曲线摘要 ──────────────────────────────────────────────────────
    print("\n【BERT fine-tune 训练曲线（val 集）】")
    print(f"{'Epoch':<8} {'CLS-普通 acc':>14} {'CLS-普通 F1':>13} "
          f"{'CLS-加权 acc':>14} {'CLS-加权 F1':>13}")
    print("-" * 70)
    for a, b in zip(bert_cls, bert_weighted):
        print(f"  {a['epoch']:<6} {a['val_acc']:>14.4f} {a['val_macro_f1']:>13.4f} "
              f"{b['val_acc']:>14.4f} {b['val_macro_f1']:>13.4f}")

    # ── 最终对比表 ─────────────────────────────────────────────────────────────
    best_cls      = max(bert_cls,      key=lambda x: x["val_acc"])
    best_weighted = max(bert_weighted, key=lambda x: x["val_acc"])

    print("\n\n【四种方法最终准确率汇总】")
    print("=" * 70)
    rows = [
        ("BERT fine-tune（普通 loss）",
         f"{best_cls['val_acc']:.2%}",
         f"{best_cls['val_macro_f1']:.2%}",
         "全量 53K，3 epoch"),
        ("BERT fine-tune（加权 loss）",
         f"{best_weighted['val_acc']:.2%}",
         f"{best_weighted['val_macro_f1']:.2%}",
         "全量 53K，3 epoch，处理不均衡"),
        ("LLM Zero-shot（Qwen2-0.5B）",
         f"{zero_shot['accuracy']:.2%}",
         "—",
         "200 条，无需训练"),
        ("LLM SFT + LoRA（Qwen2-0.5B）",
         f"{sft['accuracy']:.2%}",
         "—",
         "5K 条，LoRA r=8，3 epoch"),
    ]
    print(f"  {'方法':<28} {'val_acc':>8} {'macro_F1':>10} {'备注'}")
    print("  " + "-" * 65)
    for method, acc, f1, note in rows:
        print(f"  {method:<28} {acc:>8} {f1:>10}  {note}")

    # ── 关键洞察 ──────────────────────────────────────────────────────────────
    print("\n\n【关键洞察】")
    print("  1. 加权 loss vs 普通 loss：")
    delta_acc = best_weighted["val_acc"] - best_cls["val_acc"]
    delta_f1  = best_weighted["val_macro_f1"] - best_cls["val_macro_f1"]
    print(f"     - val_acc 差值：{delta_acc:+.4f}（加权 loss 略低）")
    print(f"     - macro F1 差值：{delta_f1:+.4f}（加权 loss 提升少数类覆盖率）")
    print("     → 加权 loss 牺牲少量整体准确率，换取对少数类（如证券）更好的召回")

    unparseable_rate = zero_shot["unparseable"] / zero_shot["total"]
    print(f"\n  2. LLM Zero-shot 无需训练，但准确率仅 {zero_shot['accuracy']:.0%}，")
    print(f"     且有 {unparseable_rate:.0%} 的回答无法解析（输出了类别以外的词汇）。")
    print("     → 适合快速原型、无标注数据场景，但精度有限")

    sft_unparseable_rate = sft["unparseable"] / sft["total"]
    print(f"\n  3. LLM SFT 仅用 5K 条数据（BERT 全量的 9.4%）就达到 {sft['accuracy']:.0%}，")
    print(f"     输出不可解析率降至 {sft_unparseable_rate:.0%}。")
    print("     → LoRA 指令微调能高效迁移大模型预训练知识，少量标注即可见效")

    print(f"\n  4. 综合来看，LLM SFT ≈ BERT 全量，但训练数据少 90%；")
    print("     有大量标注数据时 BERT fine-tune 仍是精度最稳的选择。")
    print()


def plot_comparison(bert_cls, bert_weighted, zero_shot, sft, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("第六周作业：文本分类不同训练方法效果对比", fontsize=14, y=1.01)

    epochs = [r["epoch"] for r in bert_cls]

    # ── 图1：BERT 训练曲线（val acc）────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, [r["val_acc"] for r in bert_cls],
            marker="o", label="BERT 普通 Loss")
    ax.plot(epochs, [r["val_acc"] for r in bert_weighted],
            marker="s", label="BERT 加权 Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("BERT fine-tune 训练曲线（val acc）")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.50, 0.62)

    # ── 图2：BERT 训练曲线（macro F1）──────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, [r["val_macro_f1"] for r in bert_cls],
            marker="o", label="BERT 普通 Loss")
    ax.plot(epochs, [r["val_macro_f1"] for r in bert_weighted],
            marker="s", label="BERT 加权 Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Macro F1")
    ax.set_title("BERT fine-tune 训练曲线（macro F1）")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.50, 0.60)

    # ── 图3：四方法准确率对比柱状图 ───────────────────────────────────────────
    ax = axes[2]
    methods = [
        "BERT\n普通 Loss",
        "BERT\n加权 Loss",
        "LLM\nZero-shot",
        "LLM\nSFT+LoRA",
    ]
    best_cls      = max(bert_cls,      key=lambda x: x["val_acc"])
    best_weighted = max(bert_weighted, key=lambda x: x["val_acc"])
    accs = [
        best_cls["val_acc"],
        best_weighted["val_acc"],
        zero_shot["accuracy"],
        sft["accuracy"],
    ]
    colors = ["#4C72B0", "#4C72B0", "#C44E52", "#2ca02c"]
    hatches = ["", "///", "", ""]
    bars = ax.bar(methods, accs, color=colors, hatch=hatches,
                  alpha=0.85, width=0.55, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("四种方法最终准确率对比")
    ax.set_ylim(0, 0.72)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="50% 基线")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 标注评测样本规模差异
    notes = ["val 10K", "val 10K", "200条", "200条"]
    for bar, note in zip(bars, notes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                0.01, note, ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    save_path = save_dir / "compare_methods.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存 → {save_path}")


def plot_training_loss(bert_cls, bert_weighted, sft_log, save_dir: Path):
    """训练 loss 下降曲线（3种有训练过程的方法）"""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_bert = [r["epoch"] for r in bert_cls]
    epochs_sft  = [r["epoch"] for r in sft_log]

    ax.plot(epochs_bert, [r["train_loss"] for r in bert_cls],
            marker="o", label="BERT 普通 Loss（train loss）")
    ax.plot(epochs_bert, [r["train_loss"] for r in bert_weighted],
            marker="s", label="BERT 加权 Loss（train loss）")
    ax.plot(epochs_sft,  [r["train_loss"] for r in sft_log],
            marker="^", linestyle="--", label="LLM SFT（train loss）")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("各方法训练 Loss 下降对比")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 3])

    plt.tight_layout()
    save_path = save_dir / "compare_train_loss.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练 loss 曲线已保存 → {save_path}")


def main():
    bert_cls      = load_bert_log("cls")
    bert_weighted = load_bert_log("cls_weighted")
    zero_shot     = load_llm_results("llm_zero_shot_results")
    sft           = load_llm_results("llm_sft_results")
    sft_log       = load_bert_log("sft")

    print_summary(bert_cls, bert_weighted, zero_shot, sft)
    plot_comparison(bert_cls, bert_weighted, zero_shot, sft, FIG_DIR)
    plot_training_loss(bert_cls, bert_weighted, sft_log, FIG_DIR)

    print("\n完成！所有对比图已保存到：")
    print(f"  {FIG_DIR}/compare_methods.png")
    print(f"  {FIG_DIR}/compare_train_loss.png")


if __name__ == "__main__":
    main()
