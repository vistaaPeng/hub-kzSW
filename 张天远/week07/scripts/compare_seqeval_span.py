"""
量化 seqeval vs span F1 的评估口径差异。
在同一份 LLM 输出上跑两套评估，计算"评估口径税"。

用法：python scripts/compare_seqeval_span.py [eval_json_path]
     不传参数则使用默认文件（QLoRA 7B cluener2020）
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd()

# ── 加载 LLM eval 结果 ──
default_eval = ROOT / "outputs/logs/eval_Qwen2.5-7B-Instruct_sft_cluener2020_qlora_r8_cluener2020.json"
eval_file = Path(sys.argv[1]) if len(sys.argv) > 1 else default_eval

d = json.load(open(eval_file, encoding="utf-8"))
detail = d["detail"]

# 加载验证集原始数据（用于重建 BIO gold）
val_data = json.load(open(ROOT / "data/cluener/validation.json", encoding="utf-8"))

# ── 构建 seqeval 格式的 gold ──
# cluener 数据格式：{"text": "...", "label": {"type": {"surface": [[start, end], ...]}}}
# 先把 text → gold BIO 序列
def text_to_bio(text, labels):
    """Span 标注 → character-level BIO。"""
    bio = ["O"] * len(text)
    for etype, surfaces in labels.items():
        for surface, positions in surfaces.items():
            for start, end in positions:
                bio[start] = f"B-{etype}"
                for i in range(start + 1, end + 1):
                    bio[i] = f"I-{etype}"
    return bio

# ── span F1 评估（现有逻辑） ──
def span_f1_from_detail(detail):
    """与 evaluate_sft.py 完全一致的 span F1 计算：text.find() 定位 → span 匹配。"""
    tp = 0
    pred_total = 0
    gold_total = 0
    for rec in detail:
        text = rec["text"]
        # gold spans
        gold_spans = set()
        for e in rec["gold"]:
            idx = text.find(e["text"])
            if idx == -1: continue
            gold_spans.add((e["text"], e["type"], idx, idx + len(e["text"]) - 1))
        # pred spans
        pred_spans = set()
        for e in rec["pred"]:
            idx = text.find(e["text"])
            if idx == -1: continue
            pred_spans.add((e["text"], e["type"], idx, idx + len(e["text"]) - 1))
        tp += len(gold_spans & pred_spans)
        pred_total += len(pred_spans)
        gold_total += len(gold_spans)
    p = tp / pred_total if pred_total else 0
    r = tp / gold_total if gold_total else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


def pred_to_bio(text, entities):
    """LLM JSON 输出 → character-level BIO（用 text.find 定位）。"""
    bio = ["O"] * len(text)
    for ent in entities:
        surface = ent["text"]
        etype = ent["type"]
        idx = text.find(surface)
        if idx == -1:
            continue
        bio[idx] = f"B-{etype}"
        for i in range(idx + 1, idx + len(surface)):
            bio[i] = f"I-{etype}"
    return bio


# ── seqeval 评估 ──
# 需要 seqeval 库
try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
except ImportError:
    print("需要 pip install seqeval")
    import sys; sys.exit(1)

# 收集所有样本的 gold/pred BIO
gold_bios = []
pred_bios = []

for rec in detail:
    text = rec["text"]
    gold_entities = rec["gold"]
    pred_entities = rec["pred"]

    gold_bio = pred_to_bio(text, gold_entities)
    pred_bio = pred_to_bio(text, pred_entities)

    # seqeval 评价BIO 序列
    gold_bios.append(gold_bio)
    pred_bios.append(pred_bio)

# 计算 seqeval
sf1 = f1_score(gold_bios, pred_bios)
sp = precision_score(gold_bios, pred_bios)
sr = recall_score(gold_bios, pred_bios)

# 计算 span F1
span = span_f1_from_detail(detail)

print("=" * 60)
print("seqeval vs span F1 评估口径差异量化")
print("=" * 60)
print(f"\n数据来源: {eval_file.name}")
print(f"样本数: {len(detail)}")
print(f"\n{'指标':<15} {'seqeval':>10} {'span F1':>10} {'差值':>10}")
print("-" * 45)
print(f"{'Precision':<15} {sp:>10.4f} {span['precision']:>10.4f} {sp - span['precision']:>10.4f}")
print(f"{'Recall':<15} {sr:>10.4f} {span['recall']:>10.4f} {sr - span['recall']:>10.4f}")
print(f"{'F1':<15} {sf1:>10.4f} {span['f1']:>10.4f} {sf1 - span['f1']:>10.4f}")
print(f"\n★ 评估口径税（F1）: {sf1 - span['f1']:.4f}")

# ── 分析：为什么有差距 ──
# 找出 span F1 正确但 seqeval 错误的样本
discrepancy = 0
boundary_errors = 0
for i, rec in enumerate(detail):
    gold_set = set((e["text"], e["type"]) for e in rec["gold"])
    pred_set = set((e["text"], e["type"]) for e in rec["pred"])
    span_match = gold_set == pred_set

    gold_bio = pred_to_bio(rec["text"], rec["gold"])
    pred_bio = pred_to_bio(rec["text"], rec["pred"])
    seq_match = gold_bio == pred_bio

    if span_match and not seq_match:
        discrepancy += 1
        # 检查是否有边界偏移
        for ge in rec["gold"]:
            for pe in rec["pred"]:
                if ge["text"] != pe["text"] and ge["type"] == pe["type"]:
                    if ge["text"] in pe["text"] or pe["text"] in ge["text"]:
                        boundary_errors += 1
                        break

print(f"\nspan F1 全对但 seqeval 不对的样本: {discrepancy}")
print(f"其中疑似边界误差的: {boundary_errors}")
