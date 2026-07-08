"""
混合检索器 —— 基于 RRF（Reciprocal Rank Fusion）融合向量检索和 BM25 检索结果。

RRF 公式: score(doc) = Σ(1/(k + best_rank_i(doc)))
- k 默认 60，用于平滑排名差异
- best_rank_i(doc) 是该文档在第 i 个检索器中任意 chunk 的最佳排名（1-indexed）
- 按 source_url 聚合，避免长文档因多 chunk 而占优
"""

from typing import Optional

from src.chunkers.narrative import Chunk

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 10  # 从 4 增大到 10，确保小众文档能进候选池
HEADING_MATCH_BOOST = 0.003
STRUCTURED_INTENT_BOOST = 0.002
CODE_INTENT_BOOST = 0.002
COMPARISON_INTENT_BOOST = 0.001
HEADING_QUERY_MARKERS = (
    "关键字",
    "保留字",
    "列表",
    "所有权",
    "生命周期",
    "trait",
    "函数",
    "方法",
    "结构体",
    "枚举",
    "模块",
    "泛型",
    "闭包",
    "迭代器",
    "模式匹配",
    "借用",
    "引用",
)


class HybridRetriever:
    """RRF 融合检索器，合并向量检索和 BM25 检索结果。"""

    def __init__(self, vector_store, bm25_store, k: int = DEFAULT_RRF_K, reranker=None):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.k = k
        self.reranker = reranker

    def search(
        self, query: str, top_k: int = 5,
        rerank: bool = False, rerank_top_k: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        """执行 RRF 融合检索，可选后接 Cross-Encoder 重排。"""
        candidate_k = max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, 20)

        vector_results = self._safe_search(self.vector_store, query, candidate_k)
        bm25_results = self._safe_search(self.bm25_store, query, candidate_k)

        if not vector_results and not bm25_results:
            return []

        # 收集每个 source_url 在两个检索器中的最佳排名和 best chunk
        doc_best_bm25: dict[str, int] = {}     # source_url -> best rank (1-indexed)
        doc_best_vector: dict[str, int] = {}
        doc_chunks: dict[str, Chunk] = {}       # source_url -> representative chunk

        for rank, (chunk, _) in enumerate(bm25_results, 1):
            url = chunk.metadata.get("source_url", chunk.chunk_id)
            if url not in doc_best_bm25 or rank < doc_best_bm25[url]:
                doc_best_bm25[url] = rank
                doc_chunks[url] = chunk

        for rank, (chunk, _) in enumerate(vector_results, 1):
            url = chunk.metadata.get("source_url", chunk.chunk_id)
            if url not in doc_best_vector or rank < doc_best_vector[url]:
                doc_best_vector[url] = rank
                if url not in doc_chunks:
                    doc_chunks[url] = chunk

        # 收集所有出现过的 source_url
        all_urls = set(doc_best_bm25) | set(doc_best_vector)

        # RRF 计算：不在某检索器结果中的 doc 赋予 default_rank
        default_rank = candidate_k + 1

        intent = self._detect_query_intent(query)
        doc_scores = {}
        for url in all_urls:
            bm25_r = doc_best_bm25.get(url, default_rank)
            vec_r = doc_best_vector.get(url, default_rank)
            base_score = 1.0 / (self.k + bm25_r) + 1.0 / (self.k + vec_r)
            doc_scores[url] = base_score + self._metadata_boost(
                query,
                doc_chunks[url],
                intent,
            )

        # 按 RRF 分数降序排序
        sorted_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:top_k]
        results = [(doc_chunks[url], doc_scores[url]) for url in sorted_docs]

        # 可选重排
        if rerank and self.reranker is not None:
            rerank_k_n = rerank_top_k if rerank_top_k is not None else top_k
            chunks = [c for c, _ in results]
            results = self.reranker.rerank(query, chunks, top_k=rerank_k_n)

        return results

    @staticmethod
    def _safe_search(store, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """安全调用检索器的 search 方法。"""
        try:
            result = store.search(query, top_k)
            return result if result is not None else []
        except Exception:
            return []

    @staticmethod
    def _detect_query_intent(query: str) -> str:
        """Coarse query intent used for lightweight metadata boosts."""
        q = query.lower()
        if any(k in q for k in ["有哪些", "列出", "列表", "关键字", "保留字"]):
            return "list"
        if any(k in q for k in ["代码", "示例", "如何", "怎么", "fn", "函数"]):
            return "code"
        if any(k in q for k in ["区别", "对比", "比较", "不同", "vs", "versus"]):
            return "comparison"
        return "general"

    @staticmethod
    def _metadata_boost(query: str, chunk: Chunk, intent: str) -> float:
        """Small score adjustment from headings and morphology."""
        boost = 0.0
        headings = chunk.metadata.get("headings", "")
        morphology = chunk.metadata.get("morphology", "")
        if HybridRetriever._heading_matches_query(query, headings):
            boost += HEADING_MATCH_BOOST

        if intent == "list" and morphology == "structured":
            boost += STRUCTURED_INTENT_BOOST
        elif intent == "code" and morphology == "code_unit":
            boost += CODE_INTENT_BOOST
        elif intent == "comparison" and morphology in {"narrative", "mixed"}:
            boost += COMPARISON_INTENT_BOOST
        return boost

    @staticmethod
    def _heading_matches_query(query: str, headings: str) -> bool:
        """Return True when query words or common Chinese markers hit headings."""
        query = query.lower().strip()
        headings = headings.lower().strip()
        if not query or not headings:
            return False

        if query in headings or headings in query:
            return True

        query_terms = [t for t in query.lower().split() if t]
        if query_terms and any(t in headings.lower() for t in query_terms):
            return True

        return any(marker in query and marker in headings for marker in HEADING_QUERY_MARKERS)

    @staticmethod
    def expand_to_parents(
        child_results: list[tuple[Chunk, float]],
        all_chunks: list[Chunk],
        deduplicate: bool = True,
        expand_siblings: bool = True,
        max_chars: int = 3000,
    ) -> list[tuple[Chunk, float]]:
        """
        将检索到的子块扩展为对应的父块，可选拼接前后兄弟父块。

        Args:
            child_results: 检索返回的 [(子块, score), ...]
            all_chunks: 完整 chunk 列表（含父块和子块）
            deduplicate: 是否去重（同一父块只出现一次，保留最高分）
            expand_siblings: 是否扩展到前后各一个兄弟父块（拼接 text）
            max_chars: 拼接后最大字符数

        Returns:
            [(父块, 最高分), ...] 保持检索顺序
        """
        # 构建父块索引
        parent_map = {}
        sibling_index: dict[str, list[Chunk]] = {}  # source_url → 已排序的兄弟父块
        for c in all_chunks:
            if c.is_parent:
                parent_map[c.chunk_id] = c
                url = c.metadata.get("source_url", "")
                sibling_index.setdefault(url, []).append(c)

        # 按 chunk_id 排序（自然顺序 = 文档顺序）
        for url in sibling_index:
            sibling_index[url].sort(key=lambda c: c.chunk_id)

        seen_pids = set()
        ordered_results = []  # 保持 search 顺序

        for child, score in child_results:
            pid = child.parent_chunk_id
            if pid is None:
                pid = child.chunk_id
                parent_map[pid] = child

            parent = parent_map.get(pid)
            if parent is None:
                continue

            if expand_siblings:
                url = parent.metadata.get("source_url", "")
                siblings = sibling_index.get(url, [parent])
                try:
                    idx = next(i for i, s in enumerate(siblings) if s.chunk_id == pid)
                except StopIteration:
                    idx = 0

                selected = [siblings[idx]]
                total = len(siblings[idx].text)
                left = idx - 1
                right = idx + 1
                prefer_left = True

                while left >= 0 or right < len(siblings):
                    if prefer_left and left >= 0:
                        candidate = siblings[left]
                        left -= 1
                    elif right < len(siblings):
                        candidate = siblings[right]
                        right += 1
                    elif left >= 0:
                        candidate = siblings[left]
                        left -= 1
                    else:
                        break

                    prefer_left = not prefer_left
                    if total + len(candidate.text) > max_chars:
                        continue
                    selected.append(candidate)
                    total += len(candidate.text)

                selected_ids = {s.chunk_id for s in selected}
                merged = [s for s in siblings if s.chunk_id in selected_ids]
                parts = [s.text for s in merged]

                merged_text = "\n\n---\n\n".join(parts)
                merged_chunk = Chunk(
                    chunk_id=parent.chunk_id,
                    text=merged_text,
                    metadata={
                        **parent.metadata,
                        "sibling_count": len(merged),
                        "sibling_chunk_ids": [s.chunk_id for s in merged],
                        "source_url": url,
                    },
                    is_parent=True,
                )
                parent = merged_chunk

            # 去重：同一 pid 只保留第一次出现（search 中排名最高的）
            if pid not in seen_pids:
                seen_pids.add(pid)
                ordered_results.append((parent, score))

        return ordered_results
