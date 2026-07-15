#!/usr/bin/env python
"""
RAG 诊断脚本 —— 验证报告未覆盖的 4 个假设
---
在 Windows 上运行: python tests/diagnose_rag.py
不修改任何现有代码，只写新的诊断数据到 tests/diagnose_results/
"""

import sys
import json
import time
import hashlib
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────
PROJECT = Path(r"E:\npl\workspaces\npl_tran\rag_scratch")
sys.path.insert(0, str(PROJECT))

from src.retrievers.vector_store import VectorStore
from src.retrievers.bm25_store import BM25Store
from src.retrievers.hybrid_retriever import HybridRetriever
from src.llm.generator import RAGGenerator
from src.llm.query_rewriter import QueryRewriter
from src.glossary import search_glossary, get_glossary
from evaluation.evaluator import RAGEvaluator, EvalQuestion, EvalResult

INDEX_DIR = PROJECT / "vectorstore"

def load_chunks():
    path = INDEX_DIR / "all_chunks.json"
    if not path.exists():
        path = INDEX_DIR / "children.json"
    from src.chunkers.narrative import Chunk
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(d["chunk_id"], d["text"], d["metadata"],
                  d.get("parent_chunk_id"), d.get("is_parent", False))
            for d in data]


def load_retriever():
    vs = VectorStore()
    vs.load(str(INDEX_DIR / "faiss.index"), str(INDEX_DIR / "children.json"))
    bm25 = BM25Store()
    bm25.load(str(INDEX_DIR / "bm25.pkl"))
    return HybridRetriever(vs, bm25)


def multi_search(retriever, queries, top_k=10):
    """多查询检索 + source_url 去重（复用 app.py / query.py 的逻辑）"""
    seen = {}
    for q in queries:
        try:
            results = retriever.search(q, top_k=top_k)
            for chunk, score in results:
                url = chunk.metadata.get("source_url", chunk.chunk_id)
                if url not in seen or score > seen[url][1]:
                    seen[url] = (chunk, score)
        except Exception:
            continue
    merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]


# ══════════════════════════════════════════════════════════
# 诊断 1：评估器 vs 生产管线的 Faithfulness 差异
# ══════════════════════════════════════════════════════════

def diagnose_evaluator_gap(questions, output_dir):
    """对比 3 种管线配置在同一组题上的 Faithfulness"""
    print("\n" + "="*60)
    print("诊断 1：评估器 vs 生产管线 Faithfulness 差异")
    print("="*60)

    retriever = load_retriever()
    all_chunks = load_chunks()
    generator = RAGGenerator()
    rewriter = QueryRewriter()

    # 取前 10 道手动题（避免 API 调用过多）
    test_qs = [q for q in questions if q.question_id.startswith("m")][:10]

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "questions": [q.question_id for q in test_qs],
            "n_questions": len(test_qs),
        },
        "configs": {}
    }

    # ── 配置 A：裸检索（无重写、无 Glossary、无兄弟拼接）──
    print("\n--- 配置 A: 裸检索（raw search, no rewrite, no glossary）---")
    config_a = []
    for q in test_qs:
        child_results = retriever.search(q.question, top_k=10)
        chunks = [c for c, _ in child_results]
        # 无兄弟拼接
        parent_chunks = [c for c, _ in reversed(
            [(c, s) for (c, s) in retriever.expand_to_parents(
                child_results, all_chunks, expand_siblings=False)]
        )] or chunks[:8]

        t0 = time.time()
        answer = generator.generate(q.question, parent_chunks[:8], max_chunks=8)
        latency = time.time() - t0

        # 用 evaluator 的 Faithfulness 评估器
        faith, detail = RAGEvaluator(retriever, generator).compute_faithfulness(
            q.question, answer, parent_chunks[:5]
        )
        config_a.append({
            "question_id": q.question_id,
            "answer": answer[:200],
            "faithfulness": round(faith, 4),
            "latency_s": round(latency, 2),
        })
        print(f"  {q.question_id}: Faith={faith:.3f}  ({latency:.1f}s)")

    results["configs"]["A_raw"] = {
        "label": "裸检索（无重写、无Glossary、无兄弟拼接）",
        "results": config_a,
        "avg_faithfulness": round(sum(r["faithfulness"] for r in config_a) / len(config_a), 4),
        "avg_latency_s": round(sum(r["latency_s"] for r in config_a) / len(config_a), 2),
    }

    # ── 配置 B：有重写、有兄弟拼接，无动态 Glossary ──
    print("\n--- 配置 B: 重写+兄弟拼接，无动态 Glossary ---")
    config_b = []
    for q in test_qs:
        queries = rewriter.rewrite(q.question, n_variants=3)
        merged = multi_search(retriever, queries, top_k=10)
        parent_results = retriever.expand_to_parents(merged, all_chunks)
        parent_chunks = [c for c, _ in parent_results[:8]]

        t0 = time.time()
        answer = generator.generate(q.question, parent_chunks, max_chunks=8)
        latency = time.time() - t0

        faith, detail = RAGEvaluator(retriever, generator).compute_faithfulness(
            q.question, answer, parent_chunks[:5]
        )
        config_b.append({
            "question_id": q.question_id,
            "answer": answer[:200],
            "faithfulness": round(faith, 4),
            "latency_s": round(latency, 2),
            "rewrites": queries,
        })
        print(f"  {q.question_id}: Faith={faith:.3f}  ({latency:.1f}s)")

    results["configs"]["B_rewrite"] = {
        "label": "重写+兄弟拼接，无动态 Glossary",
        "results": config_b,
        "avg_faithfulness": round(sum(r["faithfulness"] for r in config_b) / len(config_b), 4),
        "avg_latency_s": round(sum(r["latency_s"] for r in config_b) / len(config_b), 2),
    }

    # ── 配置 C：全量管线（重写 + 动态 Glossary + 兄弟拼接）──
    print("\n--- 配置 C: 全量管线（重写 + 动态 Glossary + 兄弟拼接）---")
    try:
        # 确认 glossary 索引存在
        gl_idx_path = INDEX_DIR / "glossary" / "glossary_faiss.index"
        if not gl_idx_path.exists():
            print("  ⚠️ Glossary 索引不存在，跳过配置 C")
            results["configs"]["C_full"] = {"error": "glossary index not found", "avg_faithfulness": None}
        else:
            config_c = []
            for q in test_qs:
                queries = rewriter.rewrite(q.question, n_variants=3)
                merged = multi_search(retriever, queries, top_k=10)
                parent_results = retriever.expand_to_parents(merged, all_chunks)
                parent_chunks = [c for c, _ in parent_results[:8]]

                # 动态 Glossary 检索
                glossary_terms = search_glossary(q.question, top_k=10)

                t0 = time.time()
                answer = generator.generate(
                    q.question, parent_chunks, max_chunks=8,
                    glossary_terms=glossary_terms
                )
                latency = time.time() - t0

                faith, detail = RAGEvaluator(retriever, generator).compute_faithfulness(
                    q.question, answer, parent_chunks[:5]
                )
                config_c.append({
                    "question_id": q.question_id,
                    "answer": answer[:200],
                    "faithfulness": round(faith, 4),
                    "latency_s": round(latency, 2),
                    "rewrites": queries,
                    "glossary_terms": [t["english"] for t in glossary_terms[:5]],
                })
                print(f"  {q.question_id}: Faith={faith:.3f}  ({latency:.1f}s)")

            results["configs"]["C_full"] = {
                "label": "全量管线（重写 + 动态 Glossary + 兄弟拼接）",
                "results": config_c,
                "avg_faithfulness": round(sum(r["faithfulness"] for r in config_c) / len(config_c), 4),
                "avg_latency_s": round(sum(r["latency_s"] for r in config_c) / len(config_c), 2),
            }
    except Exception as e:
        print(f"  ❌ 配置 C 失败: {e}")
        results["configs"]["C_full"] = {"error": str(e), "avg_faithfulness": None}

    # 保存
    out_path = output_dir / "diagnose_evaluator_gap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 结果保存: {out_path}")

    # 摘要
    print("\n📊 诊断 1 摘要:")
    for cfg_name, cfg_data in results["configs"].items():
        if cfg_data.get("avg_faithfulness") is not None:
            label = cfg_data["label"]
            print(f"  {label}:")
            print(f"    Avg Faithfulness: {cfg_data['avg_faithfulness']}")
            print(f"    Avg Latency:      {cfg_data['avg_latency_s']}s")
        elif cfg_data.get("error"):
            print(f"  {cfg_name}: ❌ {cfg_data['error']}")

    return results


# ══════════════════════════════════════════════════════════
# 诊断 2：MRR=1.00 的测试集偏易验证
# ══════════════════════════════════════════════════════════

def diagnose_mrr_bias(questions, output_dir):
    """分析 MRR=1.00 的根因：是检索太好还是测试集偏易"""
    print("\n" + "="*60)
    print("诊断 2：MRR=1.00 分析 —— 测试集偏易还是检索完美")
    print("="*60)

    retriever = load_retriever()
    rewriter = QueryRewriter()

    # 取所有手动题（m01-m40）
    test_qs = [q for q in questions if q.question_id.startswith("m")]

    results = []
    easy_count = 0
    hard_cases = []

    for q in test_qs:
        q_lower = q.question.lower()

        # 统计：问题中的关键词有多少直接出现在标题中？
        title_hits = 0
        for kw in q.keywords:
            if kw.lower() in q_lower:
                title_hits += 1
        keyword_in_question_ratio = title_hits / len(q.keywords) if q.keywords else 0

        # 裸检索（无重写），看 top-1 chunk 的 heading 是否包含问题关键词
        raw_results = retriever.search(q.question, top_k=5)
        top1_heading = ""
        top1_score = 0
        if raw_results:
            top1 = raw_results[0][0]
            top1_heading = top1.metadata.get("headings", "")
            top1_score = raw_results[0][1]

        # 有重写后检索
        queries = rewriter.rewrite(q.question, n_variants=3)
        merged = multi_search(retriever, queries, top_k=5)
        rewrite_top1_heading = ""
        rewrite_top1_score = 0
        if merged:
            rewrite_top1_heading = merged[0][0].metadata.get("headings", "")
            rewrite_top1_score = merged[0][1]

        # 判断：如果原始查询 top-1 heading 已含关键词 → 简单题
        is_easy = any(kw.lower() in top1_heading.lower() for kw in q.keywords)

        entry = {
            "question_id": q.question_id,
            "question": q.question[:60],
            "keyword_in_question_ratio": round(keyword_in_question_ratio, 2),
            "n_keywords": len(q.keywords),
            "raw_top1_heading": top1_heading[:60],
            "raw_top1_score": round(top1_score, 4),
            "rewrite_top1_heading": rewrite_top1_heading[:60],
            "rewrite_top1_score": round(rewrite_top1_score, 4),
            "is_easy": is_easy,
        }
        results.append(entry)
        if is_easy:
            easy_count += 1
        else:
            hard_cases.append(entry)
        flag = "🟢 easy" if is_easy else "🔴 hard"
        print(f"  {q.question_id}: {flag}  kw_in_q={keyword_in_question_ratio:.2f}  "
              f"top1_heading='{top1_heading[:40]}...'")

    out_path = output_dir / "diagnose_mrr_bias.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(test_qs),
        },
        "easy_count": easy_count,
        "hard_count": len(hard_cases),
        "easy_ratio": round(easy_count / len(test_qs), 3) if test_qs else 0,
        "details": results,
        "hard_cases": [h["question_id"] for h in hard_cases],
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 结果保存: {out_path}")
    print(f"\n📊 MRR 诊断:")
    print(f"  简单题（关键词藏于 heading）: {easy_count}/{len(test_qs)} = {summary['easy_ratio']:.1%}")
    print(f"  难题（需语义匹配）:           {len(hard_cases)}/{len(test_qs)}")
    if hard_cases:
        print(f"  难题列表: {', '.join(h['question_id'] for h in hard_cases)}")
    return summary


# ══════════════════════════════════════════════════════════
# 诊断 3：Faith<0.9 的 15 道题失败模式分类
# ══════════════════════════════════════════════════════════

def diagnose_failure_modes(questions, output_dir):
    """对 Faith<0.9 题目按失败模式分类"""
    print("\n" + "="*60)
    print("诊断 3：Faith<0.9 失败模式分类")
    print("="*60)

    retriever = load_retriever()
    all_chunks = load_chunks()
    generator = RAGGenerator()
    rewriter = QueryRewriter()

    # 用全量管线跑所有手动题
    test_qs = [q for q in questions if q.question_id.startswith("m")]

    results = []

    for q in test_qs:
        queries = rewriter.rewrite(q.question, n_variants=3)
        merged = multi_search(retriever, queries, top_k=10)
        parent_results = retriever.expand_to_parents(merged, all_chunks)
        parent_chunks = [c for c, _ in parent_results[:8]]

        answer = generator.generate(q.question, parent_chunks, max_chunks=8)
        faith, detail = RAGEvaluator(retriever, generator).compute_faithfulness(
            q.question, answer, parent_chunks[:5]
        )

        # 分类失败模式
        failure_mode = None
        mode_evidence = ""

        if faith >= 0.9:
            failure_mode = "pass"
        else:
            claims = detail.get("claims", [])
            unsupported = [c for c in claims if not c.get("supported")]
            n_unsupported = len(unsupported)

            # 检查是否包含"文档中未提供"等诚实表述
            has_honest_statement = any(
                "未提供" in c.get("statement", "") or
                "无法" in c.get("statement", "") or
                "未说明" in c.get("statement", "")
                for c in claims
            )
            # 检查是否包含代码示例
            has_code = any("```" in c.get("statement", "") for c in claims)
            # 检查引述样式
            has_citation = any("来源：" in c.get("statement", "") or
                             "文档" in c.get("statement", "") for c in claims)
            # 检查对比类问题
            q_lower = q.question.lower()
            is_comparison = any(kw in q_lower for kw in ["区别", "对比", "vs", "不同", "差异", "优劣"])

            if is_comparison and has_honest_statement:
                failure_mode = "对比类+诚实表述"
                mode_evidence = f"对比类问题, {n_unsupported}条unsupported, 含诚实表述"
            elif has_code and n_unsupported >= 2:
                failure_mode = "代码示例"
                mode_evidence = f"含代码示例, {n_unsupported}条unsupported"
            elif has_honest_statement and n_unsupported <= 2:
                failure_mode = "诚实表述"
                mode_evidence = f"诚实表述, {n_unsupported}条unsupported"
            elif has_citation:
                failure_mode = "引述样式"
                mode_evidence = f"引述声明, {n_unsupported}条unsupported"
            else:
                failure_mode = "其他"
                mode_evidence = f"{n_unsupported}条unsupported, claims: {[c.get('statement','')[:50] for c in unsupported]}"

        entry = {
            "question_id": q.question_id,
            "question": q.question[:80],
            "faithfulness": round(faith, 4),
            "failure_mode": failure_mode,
            "mode_evidence": mode_evidence,
            "n_unsupported_claims": len([c for c in detail.get("claims", []) if not c.get("supported")]),
            "faithfulness_detail": detail,
        }
        results.append(entry)
        flag = "✅" if faith >= 0.9 else "❌"
        print(f"  {q.question_id}: Faith={faith:.3f} {flag}  [{failure_mode}]")

    # 统计
    by_mode = {}
    for r in results:
        mode = r["failure_mode"]
        by_mode.setdefault(mode, {"count": 0, "cases": [], "avg_faith": []})
        by_mode[mode]["count"] += 1
        by_mode[mode]["cases"].append(r["question_id"])
        by_mode[mode]["avg_faith"].append(r["faithfulness"])

    out_data = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(test_qs),
        },
        "summary_by_mode": {
            mode: {
                "count": v["count"],
                "avg_faithfulness": round(sum(v["avg_faith"]) / len(v["avg_faith"]), 4),
                "cases": v["cases"],
            } for mode, v in sorted(by_mode.items(), key=lambda x: -x[1]["count"])
        },
        "pass_count": len([r for r in results if r["faithfulness"] >= 0.9]),
        "fail_count": len([r for r in results if r["faithfulness"] < 0.9]),
        "details": results,
    }

    out_path = output_dir / "diagnose_failure_modes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 结果保存: {out_path}")
    print(f"\n📊 失败模式分类:")
    for mode, v in sorted(by_mode.items(), key=lambda x: -x[1]["count"]):
        avg = round(sum(v["avg_faith"]) / len(v["avg_faith"]), 4)
        print(f"  {mode}: {v['count']} 题  avg Faith={avg}  cases={v['cases']}")
    return out_data


# ══════════════════════════════════════════════════════════
# 诊断 4：管线延迟分解（逐阶段耗时）
# ══════════════════════════════════════════════════════════

def diagnose_latency(questions, output_dir, n_samples=5):
    """测量管线各阶段的 P50/P95 延迟"""
    print("\n" + "="*60)
    print(f"诊断 4：管线延迟分解（{n_samples} 个样本）")
    print("="*60)

    retriever = load_retriever()
    all_chunks = load_chunks()
    generator = RAGGenerator()
    rewriter = QueryRewriter()

    test_qs = [q for q in questions if q.question_id.startswith("m")][:n_samples]

    stage_times = {
        "query_rewrite": [],
        "vector_search": [],
        "bm25_search": [],
        "parent_expansion": [],
        "glossary_search": [],
        "llm_generation": [],
    }
    profile_samples = []

    for q in test_qs:
        t0 = time.time()
        queries = rewriter.rewrite(q.question, n_variants=3)
        t1 = time.time()
        stage_times["query_rewrite"].append(t1 - t0)

        # 分别计时向量和 BM25 检索
        t2_vec_start = time.time()
        vec_results = retriever.vector_store.search(q.question, top_k=60)
        t2_vec_end = time.time()
        stage_times["vector_search"].append(t2_vec_end - t2_vec_start)

        t2_bm25_start = time.time()
        bm25_results = retriever.bm25_store.search(q.question, top_k=60)
        t2_bm25_end = time.time()
        stage_times["bm25_search"].append(t2_bm25_end - t2_bm25_start)

        # 多路合并
        seen = {}
        for rq in queries:
            for chunk, score in retriever.search(rq, top_k=10):
                url = chunk.metadata.get("source_url", chunk.chunk_id)
                if url not in seen or score > seen[url][1]:
                    seen[url] = (chunk, score)
        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:10]

        # 父块扩展
        t3 = time.time()
        parent_results = retriever.expand_to_parents(merged, all_chunks)
        t4 = time.time()
        stage_times["parent_expansion"].append(t4 - t3)
        parent_chunks = [c for c, _ in parent_results[:8]]

        # Glossary
        t5 = time.time()
        try:
            glossary_terms = search_glossary(q.question, top_k=10)
        except Exception:
            glossary_terms = []
        t6 = time.time()
        stage_times["glossary_search"].append(t6 - t5)

        # LLM
        t7 = time.time()
        answer = generator.generate(q.question, parent_chunks, max_chunks=8,
                                    glossary_terms=glossary_terms)
        t8 = time.time()
        stage_times["llm_generation"].append(t8 - t7)

        total = t8 - t0
        profile = {
            "question_id": q.question_id,
            "total_s": round(total, 3),
            "stages": {
                "query_rewrite_s": round(t1 - t0, 3),
                "vector_search_s": round(t2_vec_end - t2_vec_start, 4),
                "bm25_search_s": round(t2_bm25_end - t2_bm25_start, 4),
                "parent_expansion_s": round(t4 - t3, 4),
                "glossary_search_s": round(t6 - t5, 4),
                "llm_generation_s": round(t8 - t7, 3),
            }
        }
        profile_samples.append(profile)

        bottleneck = max(profile["stages"], key=profile["stages"].get)
        print(f"  {q.question_id}: total={total:.2f}s  "
              f"bottleneck={bottleneck}={profile['stages'][bottleneck]:.3f}s")

    # 计算 P50/P95
    from statistics import median
    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        else:
            return sorted_data[f]

    p50_p95 = {}
    for stage, times in stage_times.items():
        if times:
            p50_p95[stage] = {
                "p50_s": round(median(times), 4),
                "p95_s": round(percentile(times, 0.95), 4),
                "min_s": round(min(times), 4),
                "max_s": round(max(times), 4),
            }

    out_data = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": n_samples,
        },
        "p50_p95_by_stage": p50_p95,
        "samples": profile_samples,
    }

    out_path = output_dir / "diagnose_latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 结果保存: {out_path}")
    print(f"\n📊 延迟分解 P50/P95:")
    for stage, v in sorted(p50_p95.items(), key=lambda x: -x[1]["p95_s"]):
        print(f"  {stage:20s}  P50={v['p50_s']:.4f}s  P95={v['p95_s']:.4f}s  "
              f"min={v['min_s']:.4f}s  max={v['max_s']:.4f}s")
    return out_data


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def load_questions():
    """从 questions_manual.json 和 questions.json 加载"""
    qpath = PROJECT / "evaluation" / "questions_manual.json"
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8"))
        return [EvalQuestion(**d) for d in data]
    qpath2 = PROJECT / "evaluation" / "questions.json"
    if qpath2.exists():
        data = json.loads(qpath2.read_text(encoding="utf-8"))
        return [EvalQuestion(**d) for d in data]
    print("❌ 未找到问题 JSON 文件")
    return []


if __name__ == "__main__":
    output_dir = PROJECT / "tests" / "diagnose_results"
    questions = load_questions()
    if not questions:
        sys.exit(1)
    print(f"📋 加载 {len(questions)} 道题")

    # 全部运行（约 30-60 分钟，取决于 API 速度）
    # diagnose_evaluator_gap(questions, output_dir)  # 10+ API 调用
    # diagnose_mrr_bias(questions, output_dir)        # 0 API 调用
    # diagnose_failure_modes(questions, output_dir)   # 40+ API 调用
    # diagnose_latency(questions, output_dir)          # 5 API 调用

    # 先跑不费 API 的
    diagnose_mrr_bias(questions, output_dir)

    print("\n✅ 诊断脚本已就绪。可单独调用各诊断函数。")
    print("   耗时估计：")
    print("     diagnose_mrr_bias    — 0 API 调用（即时）")
    print("     diagnose_latency     — 5 API 调用（~1 分钟）")
    print("     diagnose_gap         — 30 API 调用（~5 分钟）")
    print("     diagnose_failure     — 40 API 调用（~7 分钟）")
