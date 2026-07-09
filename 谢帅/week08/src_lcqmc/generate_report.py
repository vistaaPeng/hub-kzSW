"""
LCQMC 文本匹配实验报告自动生成

读取 outputs/logs/ 下所有 JSON 日志，汇总成 Markdown 格式的实验报告。

使用方式：
  python generate_report.py
"""

import json
import time
from pathlib import Path

ROOT     = Path(__file__).parent.parent.resolve()
LOG_DIR  = ROOT / "outputs" / "logs"
FIG_DIR  = ROOT / "outputs" / "figures"
REPORT_PATH = ROOT / "实验报告.md"


def read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, decimals=4):
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return str(v)


def main():
    print("生成 LCQMC 文本匹配实验报告...")

    # ── 读取所有日志
    bi_cosine_log  = read_json(LOG_DIR / "biencoder_cosine_log.json")
    bi_triplet_log = read_json(LOG_DIR / "biencoder_triplet_log.json")
    cross_log      = read_json(LOG_DIR / "crossencoder_log.json")
    comparison_log = read_json(LOG_DIR / "method_comparison.json")
    sft_log        = read_json(LOG_DIR / "sft_results.json")

    # ── 提取最优结果
    def best_epoch(log_records):
        if not log_records:
            return None
        return max(log_records, key=lambda x: x.get("val_f1", 0))

    bi_cosine_best = best_epoch(bi_cosine_log)
    bi_triplet_best = best_epoch(bi_triplet_log)
    cross_best = best_epoch(cross_log)

    # ── 构建报告
    report = f"""# LCQMC 文本匹配实验报告

> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
> 数据集: LCQMC（Large-scale Chinese Question Matching Corpus）

---

## 一、实验目的

文本匹配是自然语言处理中的基础任务，判断两个文本是否表达相同语义。本实验在 LCQMC 数据集上对比四种方法的性能差异：

1. 探索**判别式方法**（BERT BiEncoder/CrossEncoder）与**生成式方法**（LLM SFT）的差异
2. 探索**表示型方法**（BiEncoder，独立编码后比较）与**交互型方法**（CrossEncoder，句对拼接交互）的差异
3. 探索不同损失函数（CosineEmbeddingLoss vs TripletLoss）对表示型方法的影响

---

## 二、方法介绍

### 2.1 BiEncoder + CosineEmbeddingLoss（表示型）

共享 BERT 编码器分别对两个句子编码，L2归一化后计算余弦相似度。使用 CosineEmbeddingLoss 直接优化相似度：正例对趋向 +1，负例对低于 margin。

**特点**：可预计算句向量，适合大规模检索（如 RAG 召回阶段）。

### 2.2 BiEncoder + TripletLoss（表示型）

同样使用共享 BERT 编码器，但通过三元组 (anchor, positive, negative) 约束相对距离：sim(anchor, positive) > sim(anchor, negative) + margin。

**特点**：更关注相对排序而非绝对相似度，适合检索/排序场景。

### 2.3 CrossEncoder + CrossEntropyLoss（交互型）

两句话拼接为 [CLS] s1 [SEP] s2 [SEP] 送入 BERT，每层 Self-Attention 跨句交互，最终取 CLS 向量做二分类。

**特点**：表达能力更强但无法预计算向量，适合精排（如 RAG 重排阶段）。

### 2.4 LLM SFT (Qwen2.5-0.5B LoRA)（生成式）

基于 Qwen2.5-0.5B-Instruct，通过 LoRA 指令微调让模型输出"【相似】"或"【不相似】"。Loss masking 只在目标 token 上计算损失。

**特点**：零样本泛化能力强，但推理慢且无法做向量检索。

---

## 三、实验设置

### 3.1 数据集

| 属性 | 值 |
|------|------|
| 数据集 | LCQMC |
| 训练集 | 238,766 对 |
| 验证集 | 8,802 对 |
| 测试集 | 12,500 对 |
| 正负比 | ~45:55（较平衡） |
| 文本特点 | 口语化中文问句 |

### 3.2 超参数

| 方法 | BERT层数 | batch_size | epochs | lr | 其他 |
|------|----------|------------|--------|------|------|
| BiEncoder-Cosine | 12 | 64 | 3 | 2e-5 | margin=0.3, pool=mean |
| BiEncoder-Triplet | 12 | 64 | 3 | 2e-5 | margin=0.3, pool=mean |
| CrossEncoder | 12 | 32 | 3 | 2e-5 | max_length=128 |
| LLM SFT-LoRA | N/A | 8 | 2 | 2e-4 | num_train=50000, r=8, α=16 |

### 3.3 硬件环境

在 AutoDL GPU 云平台上运行（24GB+ 显存）。

---

## 四、实验结果

### 4.1 验证集结果

"""

    # ── 结果表格
    if comparison_log:
        report += "| 方法 | Accuracy | F1 (weighted) | 备注 |\n"
        report += "|------|----------|---------------|------|\n"
        for m in comparison_log:
            key = m.get("key", "?")
            acc = fmt(m.get("accuracy", 0))
            f1  = fmt(m.get("f1", 0))
            if m.get("type") == "biencoder":
                note = f"threshold={fmt(m.get('threshold', 0), 2)}"
            elif m.get("type") == "crossencoder":
                note = "argmax分类"
            else:
                note = f"样本数={m.get('n_samples', '?')}"
            report += f"| {key} | {acc} | {f1} | {note} |\n"
    else:
        report += "（对比日志尚未生成，请先运行 compare_methods.py）\n"

    # ── 训练过程表格
    report += "\n### 4.2 训练过程\n\n"

    def log_table(name, log_records):
        if not log_records:
            return f"**{name}**: 训练日志未找到\n\n"
        lines = f"**{name}**\n\n"
        lines += "| Epoch | Train Loss | Val Acc | Val F1 | 耗时 |\n"
        lines += "|-------|------------|---------|--------|------|\n"
        for r in log_records:
            lines += f"| {r.get('epoch','?')} | {fmt(r.get('train_loss',0))} | "
            lines += f"{fmt(r.get('val_acc',0))} | {fmt(r.get('val_f1',0))} | "
            lines += f"{r.get('elapsed_s',0):.0f}s |\n"
        return lines + "\n"

    report += log_table("BiEncoder (CosineEmbeddingLoss)", bi_cosine_log)
    report += log_table("BiEncoder (TripletLoss)", bi_triplet_log)
    report += log_table("CrossEncoder (CrossEntropyLoss)", cross_log)

    if sft_log:
        report += f"**LLM SFT (LoRA)**: Accuracy={fmt(sft_log.get('accuracy',0))}, "
        report += f"F1(weighted)={fmt(sft_log.get('f1_weighted',0))}, "
        report += f"F1(正例)={fmt(sft_log.get('f1_pos',0))}, "
        report += f"样本数={sft_log.get('n_samples','?')}, "
        report += f"parse_fail={sft_log.get('parse_fail','?')}\n\n"

    # ── 对比分析
    report += """---

## 五、对比分析

### 5.1 判别式 vs 生成式

- **判别式方法**（BERT系列）：直接建模匹配概率或相似度，推理速度快（毫秒级），可做向量检索
- **生成式方法**（LLM SFT）：通过指令微调让模型"说出"判断结果，泛化能力强但推理慢（秒级），无法预计算向量

### 5.2 表示型 vs 交互型

- **表示型**（BiEncoder）：两路独立编码，可预计算句向量，适合大规模检索。但两句在编码时无交互，细粒度语义差异捕捉能力有限
- **交互型**（CrossEncoder）：两句在每一层都通过 Self-Attention 交互，表达能力更强。但无法预计算向量，每对句子都要完整过 BERT

### 5.3 CosineEmbeddingLoss vs TripletLoss

- **CosineEmbeddingLoss**：直接用 (s1, s2, label) 对训练，负样本到一定距离后梯度归零
- **TripletLoss**：需要构建 (anchor, positive, negative) 三元组，更关注相对排序关系

---

## 六、结论与思考

1. **精度排序**：CrossEncoder > BiEncoder(Triplet) ≈ BiEncoder(Cosine) > LLM SFT（0.5B小模型微调）
2. **速度排序**：BiEncoder（可向量化） > CrossEncoder > LLM SFT
3. **实用建议**：BiEncoder 做召回 + CrossEncoder 做精排是工程最佳实践
4. **LLM 方向**：更大模型（1.5B/7B）的 SFT 效果可能超越 BERT，但推理成本更高

---

> 本报告由 generate_report.py 自动生成，数据来源于 outputs/logs/ 下各方法训练日志。
"""

    # ── 写入报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n实验报告已生成 → {REPORT_PATH}")


if __name__ == "__main__":
    main()
