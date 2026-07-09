"""
快速运行示例

展示如何使用本项目进行文本匹配任务

运行步骤：
  1. 数据探索
  2. 传统方法评估
  3. 深度学习方法训练和评估
  4. 方法对比
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import TraditionalDataset, build_traditional_loaders
from model import build_edit_distance_model, build_tfidf_model, build_bm25_model
from evaluate import eval_traditional
from torch.utils.data import DataLoader

# ── 配置 ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "lcqmc"

# ── 示例1：数据探索 ────────────────────────────────────────────────────

def example_explore_data():
    """数据探索示例"""
    print("\n" + "="*60)
    print("示例1：数据探索")
    print("="*60)

    # 运行数据探索脚本
    import subprocess
    subprocess.run([
        "python",
        str(ROOT / "src" / "explore_data.py"),
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(ROOT / "outputs" / "figures"),
        "--skip_token",
        "--skip_overlap",
    ])


# ── 示例2：传统方法评估 ────────────────────────────────────────────────

def example_traditional_methods():
    """传统方法评估示例"""
    print("\n" + "="*60)
    print("示例2：传统方法评估")
    print("="*60)

    # 加载验证集
    val_path = DATA_DIR / "validation.jsonl"
    val_ds = TraditionalDataset(val_path)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # 加载训练集（用于TF-IDF/BM25训练）
    train_path = DATA_DIR / "train.jsonl"
    train_ds = TraditionalDataset(train_path)
    train_sentences = []
    for batch in DataLoader(train_ds, batch_size=100, shuffle=False):
        train_sentences.extend(batch["sentence1"])
        train_sentences.extend(batch["sentence2"])

    # ── 编辑距离 ──────────────────────────────────────────────────────
    print("\n评估编辑距离...")
    model = build_edit_distance_model()
    metrics = eval_traditional(model, val_loader)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.2f}")

    # ── TF-IDF ──────────────────────────────────────────────────────────
    print("\n评估TF-IDF...")
    model = build_tfidf_model(train_sentences)
    metrics = eval_traditional(model, val_loader)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.2f}")

    # ── BM25 ────────────────────────────────────────────────────────────
    print("\n评估BM25...")
    model = build_bm25_model(train_sentences)
    metrics = eval_traditional(model, val_loader)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.2f}")


# ── 示例3：深度学习方法训练 ────────────────────────────────────────────

def example_deep_learning():
    """深度学习方法训练示例"""
    print("\n" + "="*60)
    print("示例3：深度学习方法训练")
    print("="*60)

    # 注意：需要BERT预训练模型
    bert_path = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
    if not bert_path.exists():
        print("警告：BERT预训练模型不存在，跳过深度学习方法训练")
        print(f"请下载bert-base-chinese并放置在: {bert_path}")
        return

    import subprocess

    # ── BiEncoder训练 ──────────────────────────────────────────────────
    print("\n训练BiEncoder（CosineEmbeddingLoss）...")
    subprocess.run([
        "python",
        str(ROOT / "src" / "biencoder.py"),
        "--loss_type", "cosine",
        "--epochs", "1",
        "--num_hidden_layers", "4",
    ])

    print("\n训练BiEncoder（TripletLoss）...")
    subprocess.run([
        "python",
        str(ROOT / "src" / "biencoder.py"),
        "--loss_type", "triplet",
        "--epochs", "1",
        "--num_hidden_layers", "4",
    ])

    # ── CrossEncoder训练 ───────────────────────────────────────────────
    print("\n训练CrossEncoder...")
    subprocess.run([
        "python",
        str(ROOT / "src" / "crossencoder.py"),
        "--epochs", "1",
        "--num_hidden_layers", "4",
    ])


# ── 示例4：方法对比 ────────────────────────────────────────────────────

def example_comparison():
    """方法对比示例"""
    print("\n" + "="*60)
    print("示例4：方法对比")
    print("="*60)

    import subprocess

    # 只对比传统方法（速度快）
    subprocess.run([
        "python",
        str(ROOT / "src" / "compare_methods.py"),
        "--skip_deep",
    ])


# ── 主流程 ──────────────────────────────────────────────────────────────

def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("文本匹配项目 - 快速运行示例")
    print("="*60)

    # 示例1：数据探索
    example_explore_data()

    # 示例2：传统方法评估
    example_traditional_methods()

    # 示例3：深度学习方法训练（可选）
    # example_deep_learning()

    # 示例4：方法对比
    example_comparison()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n查看结果：")
    print(f"  可视化图表: {ROOT / 'outputs' / 'figures'}")
    print(f"  评估日志  : {ROOT / 'outputs' / 'logs'}")
    print(f"  模型checkpoint: {ROOT / 'outputs' / 'checkpoints'}")


if __name__ == "__main__":
    main()