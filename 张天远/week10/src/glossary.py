"""
Rust 术语表中英对照 —— 从 Rust Wiki 下载并解析 glossary，
作为术语映射注入 LLM system prompt，确保生成答案的译名一致性。
"""

import requests
from pathlib import Path
from bs4 import BeautifulSoup

GLOSSARY_URL = "https://rustwiki.org/wiki/translate/english-chinese-glossary-of-rust/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_glossary() -> dict[str, str]:
    """
    下载并解析 Rust 术语中英对照表。

    Returns:
        {english_term: chinese_translation} 映射
    """
    resp = requests.get(GLOSSARY_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")

    if table is None:
        return {}

    glossary = {}
    for row in table.find_all("tr")[1:]:  # 跳过表头
        cols = row.find_all("td")
        if len(cols) >= 2:
            english = cols[0].get_text(strip=True)
            chinese = cols[1].get_text(strip=True)
            if english and chinese:
                glossary[english.strip()] = chinese.strip()

    return glossary


def format_glossary_for_prompt(glossary: dict[str, str], max_terms: int = 50) -> str:
    """
    格式化术语表为可注入 prompt 的文本。

    优先包含补充术语（_GLOSSARY_SUPPLEMENT），再按长度填充复合术语。
    跳过单字中文翻译（LLM 已知），节省 prompt 空间给真正需要注入的术语。

    Args:
        glossary: {english: chinese} 映射
        max_terms: 最多包含的术语数

    Returns:
        格式化的文本
    """
    def _is_compound(chinese: str) -> bool:
        """是否为复合术语（2 个及以上非英文/数字字符）"""
        cjk_count = sum(1 for c in chinese if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
        return cjk_count >= 2

    # 1. 补充术语优先注入
    supplement_keys = [k for k in _GLOSSARY_SUPPLEMENT if k in glossary]
    remaining_slots = max_terms - len(supplement_keys)

    # 2. 剩余从复合术语中按中文长度选
    other_keys = [k for k in glossary if k not in _GLOSSARY_SUPPLEMENT and _is_compound(glossary[k])]
    other_sorted = sorted(other_keys, key=lambda k: len(glossary[k]))[:max(0, remaining_slots)]

    all_terms = supplement_keys + other_sorted

    lines = [
        "## Rust 术语中英对照（请使用以下标准译名）",
        "| 英文 | 中文 |",
        "|------|------|",
    ]
    for term in all_terms:
        lines.append(f"| {term} | {glossary[term]} |")

    return "\n".join(lines)


# 缓存——全局加载一次
_glossary_cache: dict[str, str] | None = None
_glossary_retriever_cache: dict[str, tuple[object, object]] = {}

# 补充：下载的 Glossary 可能缺失的术语
_GLOSSARY_SUPPLEMENT = {
    "supertrait": "父 trait",
    "associated type": "关联类型",
    "sub-trait": "子 trait",
}


def get_glossary() -> dict[str, str]:
    """获取术语表（带缓存），首次调用会下载，并合并补充术语。"""
    global _glossary_cache
    if _glossary_cache is None:
        try:
            _glossary_cache = download_glossary()
            # 合并补充术语
            for k, v in _GLOSSARY_SUPPLEMENT.items():
                if k not in _glossary_cache:
                    _glossary_cache[k] = v
        except Exception:
            _glossary_cache = dict(_GLOSSARY_SUPPLEMENT)
    return _glossary_cache


def build_glossary_index(save_dir: str = "vectorstore/glossary",
                            model: "SentenceTransformer | None" = None) -> tuple[int, int]:
    """
    将 Glossary 构建为可检索索引（FAISS + BM25）。

    每条术语作为一个 chunk，text = "english chinese_translation"。
    查询时与文档索引并行检索，按需召回相关术语注入 LLM prompt。

    Args:
        save_dir: 索引保存目录

    Returns:
        (term_count, term_count)
    """
    import json
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from src.retrievers.bm25_store import BM25Store
    from src.chunkers.narrative import Chunk

    glossary = get_glossary()
    terms = list(glossary.items())

    # 构建 chunk 列表
    chunks = []
    for eng, chn in terms:
        text = f"{eng} {chn}"
        chunks.append(Chunk(
            chunk_id=f"glossary_{eng.replace(' ', '_')}",
            text=text,
            metadata={"english": eng, "chinese": chn, "source_name": "glossary"},
        ))

    # FAISS 索引（直接构建，复用已有 model）
    if model is None:
        model = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="cpu")
    embeddings = model.encode([c.text for c in chunks], normalize_embeddings=True)
    emb_array = np.array(embeddings, dtype=np.float32)
    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    # 保存 FAISS 索引
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(save_dir) / "glossary_faiss.index"))

    # 保存 chunk 数据（JSON）
    chunk_data = []
    for c in chunks:
        chunk_data.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata,
            "parent_chunk_id": c.parent_chunk_id,
            "is_parent": c.is_parent,
        })
    json.dump(chunk_data, open(str(Path(save_dir) / "glossary_children.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    # BM25 索引
    bm25 = BM25Store()
    bm25.build_index(chunks)
    bm25.save(str(Path(save_dir) / "glossary_bm25.pkl"))

    print(f"  ✅ Glossary 索引: {len(chunks)} 术语 (FAISS + BM25)")
    return len(chunks), len(chunks)


def search_glossary(query: str, top_k: int = 10,
                    index_dir: str = "vectorstore/glossary") -> list[dict]:
    """
    检索与查询相关的 Glossary 术语。

    并行检索 FAISS + BM25 → RRF 融合 → 返回 top_k 术语。

    Args:
        query: 用户查询
        top_k: 返回术语数
        index_dir: 索引目录

    Returns:
        [{"english": "supertrait", "chinese": "父 trait", "score": 0.95}, ...]
    """
    vs, bm25 = get_glossary_retrievers(index_dir)

    # 并行检索
    vector_results = vs.search(query, top_k=top_k * 2) or []
    bm25_results = bm25.search(query, top_k=top_k * 2) or []

    # RRF 融合
    K = 60
    scores = {}
    for rank, (chunk, _) in enumerate(vector_results, 1):
        key = chunk.metadata["english"]
        scores[key] = scores.get(key, 0) + 1.0 / (K + rank)
    for rank, (chunk, _) in enumerate(bm25_results, 1):
        key = chunk.metadata["english"]
        scores[key] = scores.get(key, 0) + 1.0 / (K + rank)

    # 取 top_k
    sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    glossary = get_glossary()
    return [
        {"english": eng, "chinese": glossary.get(eng, ""), "score": round(s, 4)}
        for eng, s in sorted_terms
    ]


def get_glossary_retrievers(index_dir: str = "vectorstore/glossary"):
    """Load glossary retrievers once per index directory."""
    key = str(Path(index_dir).resolve())
    if key not in _glossary_retriever_cache:
        from src.retrievers.vector_store import VectorStore
        from src.retrievers.bm25_store import BM25Store

        vs = VectorStore()
        vs.load(str(Path(index_dir) / "glossary_faiss.index"),
                str(Path(index_dir) / "glossary_children.json"))
        bm25 = BM25Store()
        bm25.load(str(Path(index_dir) / "glossary_bm25.pkl"))
        _glossary_retriever_cache[key] = (vs, bm25)
    return _glossary_retriever_cache[key]


def clear_glossary_retriever_cache() -> None:
    """Clear cached glossary retrievers, mainly for tests and index rebuilds."""
    _glossary_retriever_cache.clear()
