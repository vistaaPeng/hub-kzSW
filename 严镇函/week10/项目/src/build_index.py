"""
向量索引构建脚本：将刑法分块结果 Embedding → FAISS 索引

Embedding 方案：阿里云 DashScope text-embedding-v3
  - 无需下载本地模型，直接 API 调用
  - 维度：1024
  - 费用极低

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

使用前请设置环境变量：
  set DASHSCOPE_API_KEY=sk-xxx

依赖：
  pip install faiss-cpu openai numpy
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 路径配置
BASE_DIR        = Path(__file__).parent.parent          # 项目根目录

CHUNKS_DIR      = BASE_DIR / "data" / "chunks"          # 分块结果目录
VECTORSTORE_DIR = "D:\PythonStudy\yzh_study\Embedding\week10\\vectorstore"
VECTORSTORE_DIR = Path(VECTORSTORE_DIR)# 向量库保存目录
print(VECTORSTORE_DIR)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding 配置
STRATEGY     = "by_article"                             # 分块策略，需与 chunk_documents.py 一致
CHUNKS_FILE  = CHUNKS_DIR / f"chunks_{STRATEGY}.json"   # 输入：分块结果
EMBED_MODEL  = "text-embedding-v3"                      # DashScope 向量模型
EMBED_DIM    = 1024                                     # 向量维度（可选 768/512 节省存储）
BATCH_SIZE   = 10                                       # API 每次最多处理 10 条
DASHSCOPE_URL = "https://ws-4hvtvp46nkb7dftd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# ── DashScope 客户端 ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """
    创建 DashScope API 客户端

    从环境变量读取 API Key：
      Windows: set DASHSCOPE_API_KEY=sk-xxx
      Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx
    """
    # 优先从环境变量读取
    api_key = "sk-ws-H.EMERMYL.YoR7.MEYCIQDk2OkODYwBYANDFmDpOuM38y4OdLKOk3v1ak7STlm7-AIhAMDt20IxGX_pjFAT-u3Y3qzSt1ox3FI2pqYOqGqkr662"

# 如果环境变量没设置，尝试从年报项目的代码里找（你之前写死的 key）
    if not api_key:
        # 提示用户设置
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows: set DASHSCOPE_API_KEY=sk-xxx\n"
            "  或在代码中直接填写 api_key（不推荐，仅供测试）"
        )

    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """
    批量计算文本的向量 Embedding

    参数 client: OpenAI 客户端
    参数 texts: 文本列表（如 ["第一条...", "第二条...", ...]）
    返回: shape=(N, 1024) 的 float32 数组，已 L2 归一化

    注意：DashScope text-embedding-v3 每次最多 10 条
    """
    all_embeddings = []       # 存放所有向量
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE  # 总批次数

    logger.info(f"开始计算 {len(texts)} 条文本的 Embedding...")

    # 分批处理
    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]           # 取一批文本
        batch_idx = i // BATCH_SIZE + 1                  # 当前批次编号

        # 每 10 批打印一次进度
        if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == total_batches:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        # 调用 API（带重试机制，最多试 3 次）
        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=EMBED_DIM,
                )
                # 提取向量
                vecs = [e.embedding for e in resp.data]
                all_embeddings.extend(vecs)
                break  # 成功就跳出重试循环
            except Exception as e:
                if attempt == 2:  # 最后一次重试也失败 → 报错
                    raise
                logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)  # 等 1s、2s、4s 递增等待

    # 转成 numpy 数组
    embeddings = np.array(all_embeddings, dtype="float32")

    # L2 归一化：让每个向量的长度为 1
    # 这样后续用"内积"（点积）就等价于"余弦相似度"
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)     # 防止除零（比如全零向量）
    embeddings = embeddings / norms

    return embeddings
# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], client: OpenAI):
    """
    构建 FAISS 向量索引

    流程：
      1. 提取所有 chunk 的文本
      2. 计算向量 Embedding
      3. 构建 FAISS 索引（IndexFlatIP = 暴力内积检索）
      4. 保存索引文件 + 元数据文件

    FAISS 说明：
      IndexFlatIP = 暴力检索，精确但不近似
      数据量 < 10 万时速度完全够用
      刑法只有 500+ 条，毫秒级响应
    """
    import faiss

    # 第1步：提取所有文本
    texts = [c["content"] for c in chunks]
    logger.info(f"共 {len(texts)} 条文本需要向量化")

    # 第2步：计算 Embedding
    embeddings = embed_texts(client, texts)
    logger.info(f"Embedding 完成，向量维度: {embeddings.shape[1]}")

    # 第3步：构建 FAISS 索引
    logger.info("构建 FAISS 索引（IndexFlatIP）...")
    index = faiss.IndexFlatIP(EMBED_DIM)  # 创建索引
    index.add(embeddings)                 # 把所有向量加入索引
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 第4步：保存索引文件（.bin 二进制文件，体积小速度快）
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    size_kb = index_path.stat().st_size // 1024
    logger.info(f"FAISS 索引已保存 → {index_path}  ({size_kb} KB)")

    # 第5步：保存元数据（JSON 文件，记录每个 chunk 的原文和来源）
    meta_list = []
    for c in chunks:
        meta_list.append({
            "chunk_id":    c["chunk_id"],
            "content":     c["content"],
            "article_num": c["metadata"].get("article_num", ""),
            "page_num":    c["metadata"].get("page_num", 0),
            "section_path": c["metadata"].get("section_path", ""),
            "block_type":  c["metadata"].get("block_type", ""),
            "strategy":    c["metadata"].get("strategy", ""),
        })

    meta_path = VECTORSTORE_DIR / "faiss_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    """主函数：加载分块 → 向量化 → 建索引"""

    # 第1步：检查分块结果是否存在
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    # 第2步：加载分块数据
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    meta   = data.get("meta", {})
    logger.info(f"加载分块结果: {meta.get('total_chunks', 0)} 个 chunks（策略={STRATEGY}）")

    # 第3步：创建 API 客户端
    client = get_client()

    # 第4步：构建 FAISS 索引
    build_faiss_index(chunks, client)

    # 第5步：完成
    logger.info("\n" + "=" * 50)
    logger.info("索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")
    logger.info(f"  共 {len(chunks)} 条法条")
    logger.info("=" * 50)
    logger.info("下一步：运行 RAG 问答流水线 → python src/rag_pipeline.py")


# ── 程序入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()