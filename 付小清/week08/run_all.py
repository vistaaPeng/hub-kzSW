"""
一键运行 work8 全部实验（两个数据集 × 三种方法）

使用方式：
  cd work8
  python run_all.py                    # 全部
  python run_all.py --dataset bq_corpus  # 只跑 BQ（更快，建议先跑通）
  python run_all.py --skip_train         # 仅评估 + 对比（已有 checkpoint 时）
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORK8 = Path(__file__).parent
DATASETS = ("lcqmc", "bq_corpus")


def run(cmd: list[str]) -> None:
    print("\n>> " + " ".join(cmd))
    subprocess.run(cmd, cwd=WORK8, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="work8 批量实验")
    parser.add_argument("--dataset", choices=["lcqmc", "bq_corpus"], default=None,
                        help="只跑指定数据集（默认两个都跑）")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，只做评估与对比")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    datasets = [args.dataset] if args.dataset else list(DATASETS)

    extra_train = []
    if args.epochs is not None:
        extra_train += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra_train += ["--batch_size", str(args.batch_size)]

    for ds in datasets:
        print(f"\n{'#' * 60}\n# 数据集: {ds}\n{'#' * 60}")

        if not args.skip_train:
            run([py, "train_biencoder.py", "--dataset", ds, "--loss", "cosine"] + extra_train)
            run([py, "train_biencoder.py", "--dataset", ds, "--loss", "triplet"] + extra_train)
            run([py, "train_crossencoder.py", "--dataset", ds] + extra_train)

        run([py, "evaluate.py", "--dataset", ds, "--model_type", "biencoder",
             "--loss", "cosine", "--split", args.split])
        run([py, "evaluate.py", "--dataset", ds, "--model_type", "biencoder",
             "--loss", "triplet", "--split", args.split])
        run([py, "evaluate.py", "--dataset", ds, "--model_type", "crossencoder",
             "--split", args.split])

    run([py, "compare_results.py", "--split", args.split])
    print("\n全部完成。请将结果整理到 RESULTS.md。")


if __name__ == "__main__":
    main()
