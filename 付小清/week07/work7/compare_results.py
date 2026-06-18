"""
汇总 peoples_daily 上 Linear vs CRF 的评估结果

使用方式：
  cd work7
  python compare_results.py
"""

import json
from pathlib import Path

LOG_DIR = Path(__file__).parent / "outputs" / "logs"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    split = "test"
    linear_res = load_json(LOG_DIR / f"eval_linear_{split}.json")
    crf_res = load_json(LOG_DIR / f"eval_crf_{split}.json")

    print("\n" + "=" * 70)
    print("人民日报 NER（peoples_daily）— Linear vs CRF 对比")
    print("=" * 70)

    header = f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    print(header)
    print("-" * 62)

    for name, res in [("BERT + Linear", linear_res), ("BERT + CRF", crf_res)]:
        if res:
            ill = res["illegal_stats"]["total_illegal"]
            print(
                f"{name:<20} "
                f"{res['precision']:>10.4f} "
                f"{res['recall']:>10.4f} "
                f"{res['f1']:>10.4f} "
                f"{ill:>10d}"
            )
        else:
            tag = "linear" if "Linear" in name else "crf"
            print(f"{name:<20} {'（未找到，请运行 evaluate.py' + (' --use_crf' if tag == 'crf' else '') + '）':>42}")

    if linear_res and crf_res:
        f1_diff = crf_res["f1"] - linear_res["f1"]
        print(f"\nCRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        print(f"Linear 非法序列：{linear_res['illegal_stats']['total_illegal']} 条")
        print(f"CRF 非法序列：  {crf_res['illegal_stats']['total_illegal']} 条")

    print("=" * 70)


if __name__ == "__main__":
    main()
