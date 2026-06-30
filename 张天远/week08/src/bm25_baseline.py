"""
BM25 文本匹配基线（传统方法对比）

教学重点：
  1. BM25 是纯关键词匹配——不涉及任何语义理解
  2. 与 BiEncoder/CrossEncoder 对比，直观看到"语义"带来的提升
  3. RAG 系统的初始召回阶段通常用 BM25 或混合检索

实现：参照经典的 BM25 公式，无外部依赖（不需 rank_bm25 包）
  BM25(q,d) = Σ IDF(qi) × f(qi,d)×(k1+1) / [f(qi,d) + k1×(1-b + b×|d|/avgdl)]

使用方式：
  python src/bm25_baseline.py
  python src/bm25_baseline.py --split validation
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"  # default, overridable via --data_dir


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text):
    """简单字符级分词——中文BM25通常按字切分。"""
    return list(text)


class BM25:
    """极简 BM25 实现，无外部依赖。"""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b
        self.corpus = [tokenize(doc) for doc in corpus]
        self.N  = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.N, 1)
        self.df = Counter()   # 文档频率
        self.tf = []          # 每文档的词频
        for doc in self.corpus:
            cnt = Counter(doc)
            self.tf.append(cnt)
            for term in cnt:
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        if n == 0:
            return 0
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query, doc_idx):
        q_terms = tokenize(query)
        doc_tf  = self.tf[doc_idx]
        doc_len = len(self.corpus[doc_idx])
        total = 0.0
        for t in q_terms:
            f = doc_tf.get(t, 0)
            if f == 0:
                continue
            idf_val = self.idf(t)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += idf_val * numerator / denominator
        return total

    def get_scores(self, query):
        return [self.score(query, i) for i in range(self.N)]


def evaluate():
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    data_path = data_dir / f"{args.split}.jsonl"
    rows = load_jsonl(data_path)
    s1_list = [r["sentence1"] for r in rows]
    s2_list = [r["sentence2"] for r in rows]
    labels  = [r["label"]      for r in rows]
    print(f"数据集: {args.split}  共 {len(rows):,} 条")

    # ── 构建 BM25 索引（以 sentence2 为语料库）─────────────────────────
    print(f"\n构建 BM25 索引（{len(s2_list):,} 文档）...")
    bm25 = BM25(s2_list, k1=1.5, b=0.75)

    # ── 检索评估（一对一匹配，非 Top-K 检索）─────────────────────────────
    # 对每对句子，用 s1 作为 query 检索所有 s2，看金标 s2 的排名
    print("评估中...")
    sims = []
    for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
        scores = bm25.get_scores(s1)             # query 对所有候选的分数
        sims.append(scores[i])                    # 取金标配对的那个分数
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(rows)}]")

    sims = np.array(sims)
    # 归一化到 [0,1] 区间便于阈值搜索
    s_min, s_max = sims.min(), sims.max()
    if s_max > s_min:
        sims_norm = (sims - s_min) / (s_max - s_min)
    else:
        sims_norm = sims

    labels_np = np.array(labels)

    # ── 阈值搜索 ─────────────────────────────────────────────────────────
    best_f1, best_thresh = -1.0, 0.5
    best_acc = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims_norm >= t).astype(int)
        f1 = f1_score(labels_np, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = accuracy_score(labels_np, preds)

    preds = (sims_norm >= best_thresh).astype(int)
    acc = accuracy_score(labels_np, preds)
    f1  = f1_score(labels_np, preds, average="weighted", zero_division=0)

    print(f"\n{'='*55}")
    print(f"BM25 基线评估结果（{args.split}，{len(rows):,} 条）")
    print(f"  k1={bm25.k1}  b={bm25.b}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1(weighted): {f1:.4f}")
    print(f"  最优阈值    : {best_thresh:.2f}（归一化后）")
    print(f"\n{classification_report(labels_np, preds, target_names=['不相似', '相似'])}")

    # ── 保存结果 ─────────────────────────────────────────────────────────
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = data_dir.name  # "afqmc" or "lcqmc"
    out = {"dataset": ds_tag, "k1": bm25.k1, "b": bm25.b,
           "accuracy": acc, "f1_weighted": f1, "threshold": float(best_thresh)}
    log_path = log_dir / f"bm25_{ds_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果 → {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="BM25 文本匹配基线")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--data_dir", default=None, type=str,
                   help="数据目录（默认: data/afqmc）")
    return p.parse_args()


if __name__ == "__main__":
    evaluate()
