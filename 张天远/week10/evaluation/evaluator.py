"""
RAG 评估器 —— 四个核心指标 + 详细报告

指标：
- Context Precision: 检索结果中关键词命中的比例
- MRR (Mean Reciprocal Rank): 第一个相关 chunk 排名的倒数
- Faithfulness: LLM 逐句核查答案是否有检索依据（需 DeepSeek API）
- Answer Relevancy: LLM 逆向生成问题 + 语义相似度（需 DeepSeek API + Embedding）

报告：JSON 格式，含每题详情 + 汇总统计 + 可追溯元数据
"""

import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from src.chunkers.narrative import Chunk
from src.retrievers.hybrid_retriever import HybridRetriever
from src.llm.generator import RAGGenerator, _get_api_key
from src.glossary import get_glossary
from src.pipeline import RAGPipeline


@dataclass
class EvalQuestion:
    """评估用问题"""
    question_id: str
    question: str
    keywords: list[str] = field(default_factory=list)
    expected_source: str | None = None


@dataclass
class EvalResult:
    """单题评估结果"""
    question_id: str
    question: str = ""
    answer: str = ""
    num_retrieved: int = 0
    num_relevant: int = 0
    # 四个指标
    precision: float | None = None  # 兼容旧测试/旧报告字段
    context_precision: float = 0.0
    recall: float = -1.0
    mrr: float = 0.0
    faithfulness: float | None = None      # None = 未计算
    answer_relevancy: float | None = None  # None = 未计算
    # 召回证据
    retrieved_chunks: list[dict] = field(default_factory=list)
    first_relevant_rank: int | None = None
    # LLM 详细输出（可追溯）
    faithfulness_detail: dict | None = None
    relevancy_detail: dict | None = None

    def __post_init__(self):
        if self.precision is not None:
            self.context_precision = self.precision
        else:
            self.precision = self.context_precision


class RAGEvaluator:
    """完整 RAG 评估器"""

    def __init__(self, retriever: HybridRetriever | None = None,
                 generator: RAGGenerator | None = None,
                 top_k: int = 10,
                 pipeline: RAGPipeline | None = None):
        self.pipeline = pipeline
        self.retriever = retriever or (pipeline.retriever if pipeline else None)
        self.generator = generator or (pipeline.generator if pipeline else None)
        if self.retriever is None:
            raise ValueError("RAGEvaluator requires retriever or pipeline")
        self.top_k = top_k
        self.glossary = get_glossary()  # 中英术语映射

    def expand_keywords(self, keywords: list[str]) -> list[str]:
        """利用 Glossary 将中文关键词扩展为中英双语变体。"""
        expanded = list(keywords)
        for kw in keywords:
            lower = kw.lower()
            # 中文关键词 → 查找对应的英文术语
            for en, zh in self.glossary.items():
                if lower in zh.lower() or lower == zh.lower():
                    if en.lower() not in expanded:
                        expanded.append(en.lower())
            # 英文关键词 → 查找对应的中文术语
            for en, zh in self.glossary.items():
                if lower == en.lower() and zh.lower() not in expanded:
                    expanded.append(zh.lower())
        return expanded

    # ── 基础指标（无 LLM）─────────────────────────────────

    def compute_precision(self, question: EvalQuestion,
                          chunks: list[Chunk]) -> float:
        """Context Precision: 包含任一关键词的 chunk 比例。"""
        if not chunks:
            return 0.0
        keywords = [kw.lower() for kw in
                    self.expand_keywords(question.keywords)]
        if not keywords:
            return 0.0
        matched = sum(
            1 for c in chunks
            if any(kw in c.text.lower() for kw in keywords)
        )
        return matched / len(chunks)

    def compute_relevant_count(self, question: EvalQuestion,
                               chunks: list[Chunk]) -> int:
        """Count chunks containing any expanded keyword."""
        keywords = [kw.lower() for kw in self.expand_keywords(question.keywords)]
        if not keywords:
            return 0
        return sum(
            1 for c in chunks
            if any(kw in c.text.lower() for kw in keywords)
        )

    def compute_recall(self, question: EvalQuestion, chunks: list[Chunk],
                       ground_truth: list[Chunk] | None = None) -> float:
        """Recall placeholder kept for backward-compatible reports/tests."""
        return -1.0

    def compute_mrr(self, question: EvalQuestion,
                    chunks: list[Chunk]) -> tuple[float, int | None]:
        """MRR: 1 / 第一个相关 chunk 的排名。"""
        keywords = [kw.lower() for kw in
                    self.expand_keywords(question.keywords)]
        if not keywords or not chunks:
            return 0.0, None
        for rank, chunk in enumerate(chunks, 1):
            if any(kw in chunk.text.lower() for kw in keywords):
                return 1.0 / rank, rank
        return 0.0, None

    # ── LLM 指标 ──────────────────────────────────────────

    def _classify_question_type(self, question: str) -> str:
        """将问题分类为 'factual'（事实类）或 'inferential'（推导/对比类）。"""
        q = question.lower()
        inferential_keywords = [
            "区别", "对比", "比较", "不同", "异同", "优劣", "优缺点",
            "关系", "联系", "异同点", "哪个更好", "有什么不同",
            "综合", "分析", "如何选择", "什么情况", "适用场景",
            "为什么", "vs", "versus", "有何不同", "如何选择", "如何区分", "如何影响",
        ]
        if any(kw in q for kw in inferential_keywords):
            return "inferential"
        # 问题长度较短且含"是什么"等→事实类
        if len(question) < 40 and any(kw in q for kw in ["是什么", "什么是", "定义"]):
            return "factual"
        # 默认用 LLM 判断（回退：短问题→事实，长问题→推导）
        return "inferential" if len(question) > 60 else "factual"

    def compute_faithfulness(self, question: str, answer: str,
                              chunks: list[Chunk]) -> tuple[float, dict]:
        """
        Faithfulness: LLM 核查答案声明是否有检索依据。
        事实类问题 → 逐句核查单 chunk 支持
        推导/对比类问题 → 允许联合多 chunk 推导，验证证据链完整性
        """
        if not answer.strip():
            return 1.0, {"error": "empty answer"}
        if self.generator is None:
            return 0.5, {"error": "generator unavailable"}

        qtype = self._classify_question_type(question)
        context = "\n\n---\n\n".join(
            f"[文档{i+1}] {c.text}" for i, c in enumerate(chunks[:5])
        )

        if qtype == "inferential":
            prompt = f"""请评估以下回答的忠实度（Faithfulness）。这是一道**对比/综合/推导类**问题，答案可能整合了多个文档的信息形成新结论。

## 检索到的文档内容
{context}

## 用户问题
{question}

## 模型回答
{answer}

## 任务
1. 将回答拆分为独立的声明（claims）
2. 逐条判断每个声明是否能被检索文档支持：
   - **事实声明**（如"X 是 Y"）：检验是否在文档中找到直接依据
   - **推导结论**（如"X 优于 Y"）：检验推导前提是否都被文档覆盖（如 X 的成本数据 + Y 的性能数据 → "X 成本更低但性能弱于 Y"），**只要前提都在文档中且有逻辑关系，即视为 supported**
   - **诚实表述**："文档中未提供/无法说明"等 → 直接判定 supported
   - **代码示例**：只要示例中的概念在文档中有说明，即视为 supported
3. 输出 JSON：
{{
  "claims": [
    {{"statement": "声明内容", "supported": true/false, "source": "文档编号", "type": "factual/inference"}}
  ],
  "faithfulness_score": 0.0~1.0,
  "notes": "简要说明"
}}

只返回 JSON。"""
        else:
            prompt = f"""请评估以下回答的忠实度（Faithfulness）。

## 检索到的文档内容
{context}

## 用户问题
{question}

## 模型回答
{answer}

## 任务
1. 将回答拆分为独立的声明（claims）
2. 逐条判断每个声明是否能被检索到的文档中找到依据
3. **重要例外**：
   - "文档中未提供/未说明/未列出/无法回答"等诚实表述 → **直接判定为 supported**（这恰恰是最忠实的行为）
   - 代码示例（```块）→ 只要示例中的概念在文档中有说明，即视为 supported
   - "来源：文档 X"形式的引用 → 只要 X 在检索文档范围内，该引用声明即视为 supported
4. 输出 JSON：
{{
  "claims": [
    {{"statement": "声明内容", "supported": true/false, "source": "文档编号或无"}}
  ],
  "faithfulness_score": 0.0~1.0,
  "notes": "简要说明"
}}

只返回 JSON，不要其他内容。"""

        detail = {}
        score = 0.5  # 默认
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            text = response.choices[0].message.content or "{}"
            detail = self._parse_json(text)
            score = float(detail.get("faithfulness_score", 0.5))
        except Exception as e:
            detail = {"error": str(e)}
        return score, detail

    def compute_answer_relevancy(self, question: str, answer: str
                                 ) -> tuple[float, dict]:
        """
        Answer Relevancy: LLM 逆向生成问题 + 与原始问题的语义相似度。

        Returns:
            (relevancy_score, detail_dict)
        """
        if not answer.strip():
            return 0.0, {"error": "empty answer"}
        if self.generator is None:
            return 0.5, {"error": "generator unavailable"}

        # Step 1: LLM 生成逆向问题
        prompt = f"""根据以下回答，生成 3 个这个回答可能对应的问题。
只返回 JSON 数组，不要其他内容。

回答：{answer}

格式：["问题1", "问题2", "问题3"]"""

        detail = {"generated_questions": []}
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            text = response.choices[0].message.content or "[]"
            gen_questions = self._parse_json_array(text)
            detail["generated_questions"] = gen_questions
        except Exception as e:
            detail["error"] = str(e)
            return 0.5, detail

        if not detail["generated_questions"]:
            return 0.5, detail

        # Step 2: 计算语义相似度
        try:
            orig_vec = self.retriever.vector_store.encode([question])[0]
        except Exception:
            return 0.5, detail

        gen_vecs = self.retriever.vector_store.encode(
            detail["generated_questions"]
        )
        similarities = []
        for gv in gen_vecs:
            sim = float(orig_vec.dot(gv))  # 余弦相似度（已归一化）
            similarities.append(sim)
        score = sum(similarities) / len(similarities)
        detail["similarities"] = similarities
        return max(0.0, min(1.0, score)), detail

    # ── 综合评估 ──────────────────────────────────────────

    def evaluate_one(self, question: EvalQuestion,
                     compute_llm: bool = True) -> EvalResult:
        """评估单个问题。"""
        # 检索：优先走完整 pipeline，确保评估和真实查询链路一致。
        pipeline_result = None
        if self.pipeline is not None:
            pipeline_result = self.pipeline.retrieve(question.question, top_k=self.top_k)
            scored_chunks = pipeline_result.parent_results
        else:
            scored_chunks = self.retriever.search(question.question, top_k=self.top_k)

        chunks = [c for c, _ in scored_chunks]
        scores = [s for _, s in scored_chunks]

        result = EvalResult(
            question_id=question.question_id,
            question=question.question,
            num_retrieved=len(chunks),
        )

        # 基础指标
        result.context_precision = self.compute_precision(question, chunks)
        result.precision = result.context_precision
        result.num_relevant = self.compute_relevant_count(question, chunks)
        result.mrr, result.first_relevant_rank = self.compute_mrr(
            question, chunks
        )

        # 记录检索结果
        result.retrieved_chunks = [
            {
                "rank": i + 1,
                "score": round(scores[i], 5),
                "chunk_id": getattr(c, "chunk_id", f"chunk_{i}"),
                "source_url": getattr(c, "metadata", {}).get("source_url", ""),
                "headings": getattr(c, "metadata", {}).get("headings", ""),
                "text_preview": c.text[:200],
            }
            for i, c in enumerate(chunks)
        ]

        if not compute_llm:
            return result

        # 生成答案
        try:
            if self.pipeline is not None and pipeline_result is not None:
                pipeline_result = self.pipeline.generate_answer(pipeline_result, max_chunks=8)
                result.answer = pipeline_result.answer
                chunks = pipeline_result.parent_chunks[:8]
            elif self.generator is not None:
                result.answer = self.generator.generate(
                    question.question, chunks, max_chunks=8
                )
            else:
                return result
        except Exception as e:
            result.answer = f"[生成失败: {e}]"

        # LLM 指标
        if result.answer:
            result.faithfulness, result.faithfulness_detail = (
                self.compute_faithfulness(
                    question.question, result.answer, chunks
                )
            )
            result.answer_relevancy, result.relevancy_detail = (
                self.compute_answer_relevancy(
                    question.question, result.answer
                )
            )

        return result

    def evaluate_all(self, questions: list[EvalQuestion],
                     compute_llm: bool = True) -> list[EvalResult]:
        """批量评估。"""
        results = []
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q.question_id}: {q.question[:50]}...")
            try:
                r = self.evaluate_one(q, compute_llm=compute_llm)
                results.append(r)
            except Exception as e:
                print(f"    ❌ {e}")
        return results

    # ── 报告生成 ──────────────────────────────────────────

    def generate_report(self, questions: list[EvalQuestion],
                        results: list[EvalResult],
                        output_path: str,
                        compute_llm: bool = True) -> str:
        """生成详细评估报告（JSON）。"""
        # 汇总
        precisions = [r.context_precision for r in results]
        mrrs = [r.mrr for r in results]
        faiths = [r.faithfulness for r in results
                  if r.faithfulness is not None]
        relevs = [r.answer_relevancy for r in results
                  if r.answer_relevancy is not None]

        def stats(vals):
            if not vals:
                return {}
            return {
                "mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "count": len(vals),
            }

        # 索引指纹
        index_hash = self._index_fingerprint()

        report = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions": len(questions),
                "top_k": self.top_k,
                "compute_llm": compute_llm,
                "index_fingerprint": index_hash,
                "model": self.generator.model,
            },
            "summary": {
                "context_precision": stats(precisions),
                "mrr": stats(mrrs),
                "faithfulness": stats(faiths),
                "answer_relevancy": stats(relevs),
            },
            "details": [],
        }

        for q, r in zip(questions, results):
            report["details"].append({
                "question_id": r.question_id,
                "question": r.question,
                "keywords": q.keywords,
                "answer": r.answer,
                "metrics": {
                    "context_precision": round(r.context_precision, 4),
                    "mrr": round(r.mrr, 4),
                    "first_relevant_rank": r.first_relevant_rank,
                    "faithfulness": (
                        round(r.faithfulness, 4)
                        if r.faithfulness is not None else None
                    ),
                    "answer_relevancy": (
                        round(r.answer_relevancy, 4)
                        if r.answer_relevancy is not None else None
                    ),
                },
                "retrieved_chunks": r.retrieved_chunks,
                "faithfulness_detail": r.faithfulness_detail,
                "relevancy_detail": r.relevancy_detail,
            })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _index_fingerprint() -> dict:
        """计算索引指纹（可复现性）。"""
        fp = {}
        for fname in ["faiss.index", "children.json", "bm25.pkl"]:
            fpath = Path("vectorstore") / fname
            if fpath.exists():
                fp[fname] = hashlib.md5(
                    fpath.read_bytes()
                ).hexdigest()[:8]
        return fp

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _parse_json_array(text: str) -> list:
        try:
            arr = json.loads(text)
            return arr if isinstance(arr, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def summarize(results: list[EvalResult]) -> dict:
        """控制台摘要。"""
        prec = [r.context_precision for r in results]
        mrr = [r.mrr for r in results]
        faith = [r.faithfulness for r in results if r.faithfulness is not None]
        relev = [r.answer_relevancy for r in results if r.answer_relevancy is not None]
        return {
            "avg_precision": round(sum(prec) / len(prec), 4) if prec else 0,
            "avg_mrr": round(sum(mrr) / len(mrr), 4) if mrr else 0,
            "avg_faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
            "avg_relevancy": round(sum(relev) / len(relev), 4) if relev else None,
            "questions_evaluated": len(results),
        }

    def summary(self, results: list[EvalResult]) -> dict:
        """Backward-compatible summary shape used by older tests."""
        if not results:
            return {
                "total_questions": 0,
                "avg_precision": 0.0,
                "avg_num_retrieved": 0.0,
                "avg_num_relevant": 0.0,
            }
        return {
            "total_questions": len(results),
            "avg_precision": sum(r.context_precision for r in results) / len(results),
            "avg_num_retrieved": sum(r.num_retrieved for r in results) / len(results),
            "avg_num_relevant": sum(r.num_relevant for r in results) / len(results),
        }
