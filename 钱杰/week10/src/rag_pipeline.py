import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "dense_index.npy"
META_PATH = VECTORSTORE_DIR / "dense_meta.json"

MODEL_NAME = str(BASE_DIR / "model_cache" / "bert-base-chinese")

TOP_K_RETRIEVE = 10
TOP_K_RERANK = 4
SCORE_THRESHOLD = 0.05

SYSTEM_PROMPT = """你是一个专业的知识问答助手，专门回答关于人工智能、气候变化、Python编程和太空探索等主题的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：人工智能分为三个阶段[1]
4. 回答简洁，重点突出，避免无关废话"""


class VectorStore:
    def __init__(self):
        self.embeddings = np.load(str(INDEX_PATH))
        
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.info(f"加载模型: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        
        logger.info(f"密集向量索引加载完成，共 {len(self.meta_list)} 条，维度: {self.embeddings.shape[1]}")

    def encode_query(self, query: str) -> np.ndarray:
        import torch
        
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        query_vec = self.encode_query(query)
        
        scores = np.dot(self.embeddings, query_vec.T).flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k * 4]
        
        results = []
        for idx in top_indices:
            if scores[idx] < 1e-9:
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(scores[idx])
            
            if filter_meta:
                match = True
                for k, v in filter_meta.items():
                    if str(item.get(k, "")) != str(v):
                        match = False
                        break
                if not match:
                    continue
            
            results.append(item)
            if len(results) >= top_k:
                break
        
        return results


class SimpleBM25:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.doc_freq = {}
        self.total_docs = len(documents)
        self.avg_doc_len = sum(len(doc) for doc in documents) / self.total_docs
        
        for doc in documents:
            terms = set(self.tokenize(doc))
            for term in terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
    
    def tokenize(self, text: str) -> List[str]:
        chars = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff' or char.isalnum():
                chars.append(char)
        return chars
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[int]:
        query_terms = self.tokenize(query)
        scores = []
        
        for doc in self.documents:
            doc_terms = self.tokenize(doc)
            doc_len = len(doc_terms)
            score = 0.0
            
            for term in query_terms:
                tf = doc_terms.count(term)
                if tf == 0:
                    continue
                
                df = self.doc_freq.get(term, 0)
                idf = np.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                
                k1 = 1.5
                b = 0.75
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                
                score += idf * numerator / denominator
            
            scores.append(score)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 1e-9]


class BM25Store:
    def __init__(self):
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        
        documents = [item["content"] for item in self.meta_list]
        logger.info("构建 BM25 索引...")
        self.bm25 = SimpleBM25(documents)
        logger.info("BM25 索引完成")
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict[str, Any]]:
        results = []
        for idx, score in self.bm25.search(query, top_k):
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(score)
            results.append(item)
        return results


def reciprocal_rank_fusion(
    vec_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    
    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item
    
    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item
    
    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


def build_context(retrieved: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    parts = []
    citations = []
    
    for i, item in enumerate(retrieved, 1):
        doc_type = item.get("doc_type", "")
        title = item.get("title", "")
        page = item.get("page_num", "")
        section = item.get("section", "")
        
        label = f"[{i}] {title}"
        if section:
            label += f" · {section}"
        if page and page != -1:
            label += f" · 第{page}页"
        
        content = item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})
    
    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        try:
            import requests
            
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            user_msg = (
                f"【参考资料】\n{context}\n\n"
                f"【问题】\n{query}\n\n"
                "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
            )
            
            payload = {
                "model": "qwen-plus",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
            }
            
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return f"参考资料:\n{context}\n\n问题: {query}\n\n(API调用失败，请检查网络连接)"
    
    return f"参考资料:\n{context}\n\n问题: {query}\n\n(需要设置 DASHSCOPE_API_KEY 环境变量才能使用大模型生成答案)"


class RAGPipeline:
    def __init__(self, use_bm25: bool = True):
        self.vec_store = VectorStore()
        self.use_bm25 = use_bm25
        self.bm25_store = BM25Store() if use_bm25 else None
    
    def query(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        vec_results = self.vec_store.search(question, TOP_K_RETRIEVE, filter_meta)
        
        if verbose and vec_results:
            logger.info(f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}")
        
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(question, TOP_K_RETRIEVE)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results
        
        final = candidates[:TOP_K_RERANK]
        
        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")
        
        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [],
                "retrieved": [],
            }
        
        top_score = final[0].get("vec_score", 1.0)
        if top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据知识库未能找到与该问题相关的内容。",
                "citations": [],
                "retrieved": final,
            }
        
        context, citations = build_context(final)
        answer = call_llm(question, context)
        
        return {"answer": answer, "citations": citations, "retrieved": final}


def main():
    parser = argparse.ArgumentParser(description="知识问答系统")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--doc-type", type=str, default=None, help="文档类型：ai, climate, python, space")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25")
    args = parser.parse_args()
    
    pipeline = RAGPipeline(use_bm25=not args.no_bm25)
    
    filter_meta = {}
    if args.doc_type:
        filter_meta["doc_type"] = args.doc_type
    if not filter_meta:
        filter_meta = None
    
    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")
    
    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
    else:
        print("知识问答系统")
        print(f"向量库：{INDEX_PATH}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            result = pipeline.query(q, filter_meta=filter_meta, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()