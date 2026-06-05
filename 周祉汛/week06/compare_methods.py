"""
第六周任务：对比文本分类不同训练方法效果

对比策略：
  1. BERT 微调（CLS 池化，标准交叉熵）
  2. BERT 微调（CLS 池化，类别加权交叉熵，应对样本不平衡）
  3. 零样本（Qwen2-0.5B）
  4. 指令微调 + LoRA（Qwen2-0.5B）
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

matplotlib.rcParams["axes.unicode_minus"] = False


def _locate_chinese_typeface():
    """寻找系统中可用的中文字体"""
    candidates = ["PingFang SC", "STHeiti", "Hiragino Sans GB",
                  "Arial Unicode MS", "SimHei", "Microsoft YaHei",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CN_FONT = _locate_chinese_typeface()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT

# 项目路径定义
PROJ_DIR = Path(__file__).parent.parent / "week6 文本分类问题" / "text_classification项目"
OUTPUT_DIR = PROJ_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def _read_bert_log(mode: str) -> list[dict]:
    """读取 BERT 训练日志 JSON 文件"""
    log_path = OUTPUT_DIR / f"train_log_{mode}.json"
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)


def _read_llm_result(kind: str) -> dict:
    """读取大模型评测结果 JSON 文件"""
    result_path = OUTPUT_DIR / f"{kind}.json"
    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def _display_summary(data_cls, data_w, data_zs, data_sft):
    """在控制台输出各项指标的对比汇总"""
    print("=" * 70)
    print(f"{'第六周作业：不同训练策略效果对比报告':^70}")
    print("=" * 70)

    # BERT 验证曲线关键数据
    print("\n【BERT 微调验证曲线（验证集）】")
    print(f"{'轮次':<8} {'普通-准确率':>14} {'普通-宏F1':>13} "
          f"{'加权-准确率':>14} {'加权-宏F1':>13}")
    print("-" * 70)
    for item_cls, item_w in zip(data_cls, data_w):
        print(f"  {item_cls['epoch']:<6} {item_cls['val_acc']:>14.4f} {item_cls['val_macro_f1']:>13.4f} "
              f"{item_w['val_acc']:>14.4f} {item_w['val_macro_f1']:>13.4f}")

    # 各方法最优指标
    best_cls = max(data_cls, key=lambda x: x["val_acc"])
    best_w = max(data_w, key=lambda x: x["val_acc"])

    print("\n\n【四种策略最终准确率总览】")
    print("=" * 70)
    rows = [
        ("BERT 微调（普通损失）",
         f"{best_cls['val_acc']:.2%}",
         f"{best_cls['val_macro_f1']:.2%}",
         "全量 53K，3 轮"),
        ("BERT 微调（加权损失）",
         f"{best_w['val_acc']:.2%}",
         f"{best_w['val_macro_f1']:.2%}",
         "全量 53K，3 轮，类别加权"),
        ("大模型零样本（Qwen2-0.5B）",
         f"{data_zs['accuracy']:.2%}",
         "—",
         "200 条样本，无训练"),
        ("大模型 SFT+LoRA（Qwen2-0.5B）",
         f"{data_sft['accuracy']:.2%}",
         "—",
         "5K 条样本，LoRA r=8，3 轮"),
    ]
    print(f"  {'方法':<28} {'验证准确率':>10} {'宏F1':>10} {'备注'}")
    print("  " + "-" * 65)
    for method, acc, f1, note in rows:
        print(f"  {method:<28} {acc:>10} {f1:>10}  {note}")

    # 核心观察
    print("\n\n【核心观察】")
    print("  1. 加权损失 vs 普通损失：")
    delta_acc = best_w["val_acc"] - best_cls["val_acc"]
    delta_f1 = best_w["val_macro_f1"] - best_cls["val_macro_f1"]
    print(f"     - 验证准确率差异：{delta_acc:+.4f}（加权损失略低）")
    print(f"     - 宏F1 差异：{delta_f1:+.4f}（加权损失提高少数类覆盖）")
    print("     → 加权损失以微弱准确率牺牲，提升了对稀有类别（如证券）的识别能力。")

    unparse_rate_zs = data_zs["unparseable"] / data_zs["total"]
    print(f"\n  2. 大模型零样本无需训练，但准确率仅为 {data_zs['accuracy']:.0%}，")
    print(f"     且输出无法解析的比例为 {unparse_rate_zs:.0%}（输出内容超出预定类别）。")
    print("     → 适用于快速验证、无标注数据场景，但精度有限。")

    unparse_rate_sft = data_sft["unparseable"] / data_sft["total"]
    print(f"\n  3. 大模型 SFT 仅使用 5K 条数据（BERT 全量的 9.4%）即达到 {data_sft['accuracy']:.0%}，")
    print(f"     输出不可解析率降至 {unparse_rate_sft:.0%}。")
    print("     → LoRA 指令微调能高效激发大模型预训练知识，小样本即可见效。")

    print(f"\n  4. 整体而言，大模型 SFT 性能 ≈ BERT 全量微调，但数据需求减少 90%；")
    print("     若拥有充足标注数据，BERT 微调依然是精度最可靠的选择。")
    print()


def _draw_comparison_figure(data_cls, data_w, data_zs, data_sft, save_dir: Path):
    """绘制三种对比子图：BERT 准确率曲线、BERT F1 曲线、四方法柱状图"""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("第六周作业：文本分类不同训练策略效果对比", fontsize=14, y=1.01)

    epochs = [r["epoch"] for r in data_cls]

    # 子图1：BERT 验证准确率曲线
    ax = axes[0]
    ax.plot(epochs, [r["val_acc"] for r in data_cls],
            marker="o", label="BERT 普通损失")
    ax.plot(epochs, [r["val_acc"] for r in data_w],
            marker="s", label="BERT 加权损失")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证准确率")
    ax.set_title("BERT 微调验证准确率变化")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.50, 0.62)

    # 子图2：BERT 验证宏F1曲线
    ax = axes[1]
    ax.plot(epochs, [r["val_macro_f1"] for r in data_cls],
            marker="o", label="BERT 普通损失")
    ax.plot(epochs, [r["val_macro_f1"] for r in data_w],
            marker="s", label="BERT 加权损失")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证宏F1")
    ax.set_title("BERT 微调验证宏F1变化")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.50, 0.60)

    # 子图3：四方法最终准确率柱状图
    ax = axes[2]
    method_labels = [
        "BERT\n普通损失",
        "BERT\n加权损失",
        "大模型\n零样本",
        "大模型\nSFT+LoRA",
    ]
    best_cls = max(data_cls, key=lambda x: x["val_acc"])
    best_w = max(data_w, key=lambda x: x["val_acc"])
    acc_values = [
        best_cls["val_acc"],
        best_w["val_acc"],
        data_zs["accuracy"],
        data_sft["accuracy"],
    ]
    bar_colors = ["#4C72B0", "#4C72B0", "#C44E52", "#2ca02c"]
    bar_hatches = ["", "///", "", ""]
    bars = ax.bar(method_labels, acc_values, color=bar_colors, hatch=bar_hatches,
                  alpha=0.85, width=0.55, edgecolor="white")
    for bar, val in zip(bars, acc_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("准确率")
    ax.set_title("四种方法最终准确率对比")
    ax.set_ylim(0, 0.72)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="50% 基线")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 标注各方法的验证集规模差异
    note_texts = ["验证集 10K", "验证集 10K", "测试集 200条", "测试集 200条"]
    for bar, note in zip(bars, note_texts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                0.01, note, ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    out_path = save_dir / "compare_methods.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比柱状图已保存 → {out_path}")


def _draw_loss_curve(data_cls, data_w, sft_log, save_dir: Path):
    """绘制训练损失下降曲线（三种有训练过程的方法）"""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_bert = [r["epoch"] for r in data_cls]
    epochs_sft = [r["epoch"] for r in sft_log]

    ax.plot(epochs_bert, [r["train_loss"] for r in data_cls],
            marker="o", label="BERT 普通损失（训练损失）")
    ax.plot(epochs_bert, [r["train_loss"] for r in data_w],
            marker="s", label="BERT 加权损失（训练损失）")
    ax.plot(epochs_sft, [r["train_loss"] for r in sft_log],
            marker="^", linestyle="--", label="大模型 SFT（训练损失）")

    ax.set_xlabel("训练轮次")
    ax.set_ylabel("训练损失")
    ax.set_title("各方法训练损失下降对比")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 3])

    plt.tight_layout()
    out_path = save_dir / "compare_train_loss.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"训练损失曲线已保存 → {out_path}")


def main():
    # 加载各方法的评测结果和训练日志
    bert_normal = _read_bert_log("cls")
    bert_weighted = _read_bert_log("cls_weighted")
    zero_shot_data = _read_llm_result("llm_zero_shot_results")
    sft_data = _read_llm_result("llm_sft_results")
    sft_training_log = _read_bert_log("sft")

    # 打印文本汇总
    _display_summary(bert_normal, bert_weighted, zero_shot_data, sft_data)
    # 绘制对比图表
    _draw_comparison_figure(bert_normal, bert_weighted, zero_shot_data, sft_data, FIGURE_DIR)
    _draw_loss_curve(bert_normal, bert_weighted, sft_training_log, FIGURE_DIR)

    print("\n任务完成！所有对比图表已生成在：")
    print(f"  {FIGURE_DIR}/compare_methods.png")
    print(f"  {FIGURE_DIR}/compare_train_loss.png")


if __name__ == "__main__":
    main()
