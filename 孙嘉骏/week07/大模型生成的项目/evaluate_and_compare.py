"""
评估与对比脚本
汇总五种NER方法的结果并进行对比分析
"""
import json
import os
import sys

def load_results():
    """加载各模型的评估结果"""
    results = {}

    # 任务一: BERT
    bert_path = "BERT/output_bert/results.json"
    if os.path.exists(bert_path):
        with open(bert_path, "r", encoding="utf-8") as f:
            results["BERT"] = json.load(f)
        print(f"[OK] 加载BERT结果: F1={results['BERT']['test_f1']:.4f}")
    else:
        print(f"[MISS] BERT结果文件不存在: {bert_path}")

    # 任务二: BERT+CRF
    bert_crf_path = "BERT/output_bert_crf/results.json"
    if os.path.exists(bert_crf_path):
        with open(bert_crf_path, "r", encoding="utf-8") as f:
            results["BERT+CRF"] = json.load(f)
        print(f"[OK] 加载BERT+CRF结果: F1={results['BERT+CRF']['test_f1']:.4f}")
    else:
        print(f"[MISS] BERT+CRF结果文件不存在: {bert_crf_path}")

    # 任务三: Qwen Zero Shot
    zero_shot_path = "LLM/output_qwen_prompt/results_zero_shot.json"
    if os.path.exists(zero_shot_path):
        with open(zero_shot_path, "r", encoding="utf-8") as f:
            results["Qwen-ZeroShot"] = json.load(f)
        print(f"[OK] 加载Qwen Zero Shot结果: F1={results['Qwen-ZeroShot']['test_f1']:.4f}")
    else:
        print(f"[MISS] Qwen Zero Shot结果文件不存在: {zero_shot_path}")

    # 任务三: Qwen Few Shot
    few_shot_path = "LLM/output_qwen_prompt/results_few_shot.json"
    if os.path.exists(few_shot_path):
        with open(few_shot_path, "r", encoding="utf-8") as f:
            results["Qwen-FewShot"] = json.load(f)
        print(f"[OK] 加载Qwen Few Shot结果: F1={results['Qwen-FewShot']['test_f1']:.4f}")
    else:
        print(f"[MISS] Qwen Few Shot结果文件不存在: {few_shot_path}")

    # 任务四: Qwen LoRA
    lora_path = "LLM/output_qwen_lora/results.json"
    if os.path.exists(lora_path):
        with open(lora_path, "r", encoding="utf-8") as f:
            results["Qwen-LoRA"] = json.load(f)
        print(f"[OK] 加载Qwen LoRA结果: F1={results['Qwen-LoRA']['test_f1']:.4f}")
    else:
        print(f"[MISS] Qwen LoRA结果文件不存在: {lora_path}")

    # 任务五: Qwen 全量微调
    full_path = "LLM/output_qwen_full/results.json"
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            results["Qwen-FullFinetune"] = json.load(f)
        print(f"[OK] 加载Qwen Full Finetune结果: F1={results['Qwen-FullFinetune']['test_f1']:.4f}")
    else:
        print(f"[MISS] Qwen Full Finetune结果文件不存在: {full_path}")

    return results

def compare_results(results):
    """对比各模型结果"""
    print("\n" + "=" * 80)
    print("人民日报NER数据集 - 五种方法效果对比")
    print("=" * 80)

    if not results:
        print("没有可用的结果数据！请先运行各训练脚本。")
        return

    # 表格形式输出
    print(f"\n{'方法':<25} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 55)

    # 按F1排序
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("test_f1", 0), reverse=True)

    for name, result in sorted_results:
        f1 = result.get("test_f1", 0)
        precision = result.get("test_precision", 0)
        recall = result.get("test_recall", 0)
        print(f"{name:<25} {f1:>8.4f} {precision:>10.4f} {recall:>8.4f}")

    # 最佳模型
    if sorted_results:
        best_name, best_result = sorted_results[0]
        print(f"\n最佳方法: {best_name} (F1={best_result.get('test_f1', 0):.4f})")

    # 各实体类型的详细对比
    print("\n" + "=" * 80)
    print("各实体类型F1对比")
    print("=" * 80)

    entity_types = ["PER", "ORG", "LOC"]
    print(f"\n{'方法':<25}", end="")
    for et in entity_types:
        print(f" {et:>8}", end="")
    print()
    print("-" * 55)

    for name, result in sorted_results:
        report = result.get("report", {})
        print(f"{name:<25}", end="")
        for et in entity_types:
            if et in report:
                f1 = report[et].get("f1-score", 0)
                print(f" {f1:>8.4f}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    # 分析总结
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)

    if len(results) >= 2:
        # BERT vs BERT+CRF
        if "BERT" in results and "BERT+CRF" in results:
            bert_f1 = results["BERT"].get("test_f1", 0)
            bert_crf_f1 = results["BERT+CRF"].get("test_f1", 0)
            diff = bert_crf_f1 - bert_f1
            print(f"\n1. BERT+CRF vs BERT: {'CRF提升' if diff > 0 else 'CRF降低'}了F1 {abs(diff):.4f} ({abs(diff)/bert_f1*100:.2f}%)")

        # Zero Shot vs Few Shot
        if "Qwen-ZeroShot" in results and "Qwen-FewShot" in results:
            zero_f1 = results["Qwen-ZeroShot"].get("test_f1", 0)
            few_f1 = results["Qwen-FewShot"].get("test_f1", 0)
            diff = few_f1 - zero_f1
            print(f"2. Few Shot vs Zero Shot: Few Shot{'提升' if diff > 0 else '降低'}了F1 {abs(diff):.4f} ({abs(diff)/max(zero_f1, 0.001)*100:.2f}%)")

        # LoRA vs Full Finetune
        if "Qwen-LoRA" in results and "Qwen-FullFinetune" in results:
            lora_f1 = results["Qwen-LoRA"].get("test_f1", 0)
            full_f1 = results["Qwen-FullFinetune"].get("test_f1", 0)
            diff = full_f1 - lora_f1
            print(f"3. Full Finetune vs LoRA: 全量微调{'提升' if diff > 0 else '降低'}了F1 {abs(diff):.4f} ({abs(diff)/max(lora_f1, 0.001)*100:.2f}%)")

        # BERT系列 vs LLM系列
        bert_best = max(
            results.get("BERT", {}).get("test_f1", 0),
            results.get("BERT+CRF", {}).get("test_f1", 0)
        )
        llm_best = max(
            results.get("Qwen-ZeroShot", {}).get("test_f1", 0),
            results.get("Qwen-FewShot", {}).get("test_f1", 0),
            results.get("Qwen-LoRA", {}).get("test_f1", 0),
            results.get("Qwen-FullFinetune", {}).get("test_f1", 0)
        )
        print(f"\n4. BERT系列最佳F1: {bert_best:.4f}, LLM系列最佳F1: {llm_best:.4f}")
        if bert_best > llm_best:
            print(f"   BERT系列在NER任务上优于LLM系列，说明专用模型在结构化任务上仍有优势")
        else:
            print(f"   LLM系列在NER任务上优于BERT系列，说明大模型的生成能力在NER上也有竞争力")

    # 保存对比结果
    comparison = {
        "models": {name: {
            "f1": result.get("test_f1", 0),
            "precision": result.get("test_precision", 0),
            "recall": result.get("test_recall", 0)
        } for name, result in results.items()},
        "ranking": [name for name, _ in sorted_results]
    }

    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存到 comparison_results.json")

if __name__ == "__main__":
    results = load_results()
    compare_results(results)
