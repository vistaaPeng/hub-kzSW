"""
汇总 LCQMC / BQ Corpus 上三种文本匹配方法的效果

使用方式：
  cd work8
  python compare_results.py
  python compare_results.py --split test
"""

import argparse
import json
from pathlib import Path

WORK8_ROOT = Path(__file__).parent
DATASETS = ("lcqmc", "bq_corpus")
METHODS = (
    ("BiEncoder + Cosine", "biencoder", "cosine", "eval_biencoder_cosine"),
    ("BiEncoder + Triplet", "biencoder", "triplet", "eval_biencoder_triplet"),
    ("CrossEncoder", "crossencoder", None, "eval_crossencoder"),
)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="work8 多数据集 × 多方法对比")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = []

    print("\n" + "=" * 78)
    print(f"文本匹配作业 — LCQMC / BQ Corpus 方法对比（{args.split} 集）")
    print("=" * 78)

    for dataset in DATASETS:
        print(f"\n【{dataset.upper()}】")
        print(f"{'方法':<24} {'Accuracy':>10} {'F1(w)':>10} {'F1(正例)':>10} {'备注':>12}")
        print("-" * 72)

        log_dir = WORK8_ROOT / "outputs" / dataset / "logs"
        for label, model_type, loss, log_prefix in METHODS:
            log_path = log_dir / f"{log_prefix}_{args.split}.json"
            res = load_json(log_path)
            if res:
                extra = f"thr={res['threshold']:.2f}" if "threshold" in res else "argmax"
                print(
                    f"{label:<24} {res['accuracy']:>10.4f} {res['f1']:>10.4f} "
                    f"{res['f1_pos']:>10.4f} {extra:>12}"
                )
                all_rows.append({"dataset": dataset, "method": label, **res})
            else:
                hint = "evaluate.py"
                if model_type == "biencoder":
                    hint += f" --loss {loss}"
                print(f"{label:<24} {'（未找到，请运行 ' + hint + '）':>48}")

    if all_rows:
        out_path = WORK8_ROOT / "outputs" / f"comparison_{args.split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"\n汇总 JSON → {out_path}")

        print("\n" + "─" * 78)
        print("简要结论（填写 RESULTS.md 时可参考）：")
        for dataset in DATASETS:
            ds_rows = [r for r in all_rows if r["dataset"] == dataset]
            if not ds_rows:
                continue
            best = max(ds_rows, key=lambda x: x["f1"])
            print(f"  {dataset}: 最高 F1(weighted) = {best['method']} ({best['f1']:.4f})")

    print("=" * 78)
    print("\n若尚未训练，请按 README.md 中的步骤依次运行 train_* 与 evaluate.py。")


if __name__ == "__main__":
    main()
