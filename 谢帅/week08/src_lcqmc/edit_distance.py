"""
编辑距离（Levenshtein Distance）文本匹配方法 —— 纯字面相似度基线

教学重点：
  1. 编辑距离是最经典的"字面相似度"度量，不依赖任何模型，纯动态规划
  2. 它与神经网络方法的本质差异：只看字面重叠，完全不懂语义
     → 同义换词（"花呗使用不了" vs "花呗被冻结"）会被判为极不相似，这正是 BERT 的价值所在
  3. 作为 lexical baseline，与 BiEncoder / CrossEncoder 同口径对比，凸显语义方法的意义

两个层次：
  - edit_distance(s1, s2)  标准动态规划，返回最小编辑次数（插入/删除/替换各算 1）
  - edit_similarity(s1, s2) 归一化到 [0, 1]，1 - dist / max(len)，可与余弦相似度同口径比较

使用方式：
  python edit_distance.py                      # 在 AFQMC validation 上评估，做阈值搜索
  python edit_distance.py --split validation --granularity char
  python edit_distance.py --demo               # 跑几个手写例子看编辑距离数值（无需数据/依赖）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT.parent / "data" / "lcqmc"
LOG_DIR  = ROOT / "outputs" / "logs"


# ── 核心算法：编辑距离（滚动数组，O(m·n) 时间 / O(min(m,n)) 空间）─────────

def edit_distance(s1, s2):
    """
    两个序列之间的 Levenshtein 编辑距离（动态规划）。

    编辑操作：插入一个元素 / 删除一个元素 / 替换一个元素，各代价为 1。

    参数：
      s1, s2 : str | list  任意可按位置索引、可比对的序列
                          （中文字符串天然按字符切分；也可传分词后的词表 list）

    返回：
      int  把 s1 变成 s2 所需的最少编辑次数

    递推关系（二维 dp[i][j] = s1[:i] 与 s2[:j] 的编辑距离）：
      s1[i-1] == s2[j-1] : dp[i][j] = dp[i-1][j-1]           # 字符相同，无需编辑
      否则               : dp[i][j] = 1 + min(               # 三选一：
                                 dp[i-1][j-1],               #   替换
                                 dp[i-1][j],                 #   删除 s1[i-1]
                                 dp[i][j-1])                 #   插入 s2[j-1]
    下面用一维滚动数组实现，dp[j] 在内层循环中代表当前行的 dp[i][j]。
    """
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m

    dp = list(range(n + 1))            # 第 0 行：s1 为空，需插入 s2 的全部 n 个字符

    for i in range(1, m + 1):
        prev = dp[0]                   # dp[i-1][0]
        dp[0] = i                      # dp[i][0]：s2 为空，需删除 s1 的全部 i 个字符
        for j in range(1, n + 1):
            temp = dp[j]               # 暂存 dp[i-1][j]，下一轮 j 的 prev
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev           # 字符相同，沿用 dp[i-1][j-1]
            else:
                dp[j] = 1 + min(
                    prev,              # 替换 dp[i-1][j-1]
                    dp[j],             # 删除 dp[i-1][j]
                    dp[j - 1],         # 插入 dp[i][j-1]
                )
            prev = temp

    return dp[n]


def edit_similarity(s1, s2, granularity="char"):
    """
    归一化编辑相似度，映射到 [0, 1]，可与余弦相似度同口径比较。

    公式：sim = 1 - dist / max(len(s1), len(s2))
      - 两句完全相同 → sim = 1.0
      - 编辑距离越大（字面差异越大）→ sim 越低

    参数：
      granularity : "char" 按字符（中文默认）/ "word" 按空格切分后的词
    """
    if granularity == "word":
        a, b = s1.split(), s2.split()
    else:
        a, b = s1, s2                  # Python 字符串本身就是字符序列
    if not a and not b:
        return 1.0
    return 1.0 - edit_distance(a, b) / max(len(a), len(b))


# ── 工具：加载 JSONL / 阈值搜索（与 dataset.py、evaluate.py 同口径）──────

def _load_jsonl(path):
    """读取 JSONL，每行一个 dict。字段：sentence1 / sentence2 / label。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_best_threshold(sims, labels):
    """枚举 [0,1] 的 101 个候选阈值，返回使 weighted-F1 最高的那个（镜像 evaluate.py）。"""
    best_f1, best_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh)


# ── 评估：在 AFQMC 上跑编辑距离基线 ───────────────────────────────────────

def evaluate_edit_distance(data_path, granularity="char"):
    """
    在 AFQMC 句对数据上用编辑距离打分，并搜索最优阈值。

    返回 dict（与 evaluate.eval_biencoder 同口径，便于横向对比）：
      similarities, labels, accuracy, f1, threshold, auc, n_pairs
    """
    rows = _load_jsonl(data_path)

    sims, labels = [], []
    for r in rows:
        sims.append(edit_similarity(r["sentence1"], r["sentence2"], granularity))
        labels.append(r["label"])

    sims, labels = np.array(sims), np.array(labels)
    threshold = _find_best_threshold(sims, labels)

    preds = (sims >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:                # 只有单类标签时 AUC 无定义
        auc = float("nan")

    return {
        "similarities": sims.tolist(),
        "labels": labels.tolist(),
        "accuracy": accuracy,
        "f1": f1,
        "threshold": threshold,
        "auc": auc,
        "n_pairs": len(rows),
    }


# ── 演示：手写例子，直观感受编辑距离数值 ──────────────────────────────────

def run_demo():
    cases = [
        ("花呗怎么提额", "花呗如何提额"),
        ("花呗使用不了", "花呗被冻结"),        # 同义但字面零重叠 → 编辑距离大
        ("双十一花呗提额在哪", "里可以提花呗额度"),
        ("kitten", "sitting"),                 # 经典英文例子：距离 3
        ("", "abc"),
    ]
    print(f"{'s1':<22}{'s2':<22}{'dist':>6}{'sim':>8}")
    print("-" * 58)
    for a, b in cases:
        d = edit_distance(a, b)
        s = edit_similarity(a, b)
        print(f"{a:<22}{b:<22}{d:>6}{s:>8.3f}")


# ── 命令行入口 ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="编辑距离文本匹配基线（AFQMC 评估）")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--granularity", default="char", choices=["char", "word"],
                   help="char=按字符（中文默认）/ word=按空格分词后的词")
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--demo", action="store_true", help="只跑手写例子，不做全量评估")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        run_demo()
        return

    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    print(f"编辑距离基线评估  粒度={args.granularity}  数据={data_path.name}")

    metrics = evaluate_edit_distance(data_path, granularity=args.granularity)

    print(f"\n{'='*50}")
    print(f"样本数 : {metrics['n_pairs']:,}")
    print(f"最优阈值: {metrics['threshold']:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1      : {metrics['f1']:.4f}")
    print(f"AUC     : {metrics['auc']:.4f}")

    preds = (np.array(metrics["similarities"]) >= metrics["threshold"]).astype(int)
    print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")

    print("结论速览：编辑距离只看字面，遇到同义换词即失效；")
    print("          对比 BiEncoder/CrossEncoder 的 F1 可直观看到语义方法的价值。")

    # 保存日志（与 biencoder_*_log.json 同口径）
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {k: v for k, v in metrics.items() if k not in ("similarities", "labels")}
    log["method"] = "edit_distance"
    log["granularity"] = args.granularity
    log["split"] = args.split
    log_path = LOG_DIR / "edit_distance_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n日志 → {log_path}")


if __name__ == "__main__":
    main()
