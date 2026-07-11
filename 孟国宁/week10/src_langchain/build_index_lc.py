"""
向量索引构建脚本（LangChain 版）— 文献问答版

与原版的对比：
  原生版：手动实现 embedding 批处理、FAISS 操作、元数据管理
  LangChain 版：Loader / Splitter / Embedding / Vectorstore 全链路框架接管

Embedding：本地 BAAI/bge-small-zh-v1.5
  需要先运行: python src_langchain/download_model.py

向量库：LangChain FAISS 封装，保存到 vectorstore/faiss_lc/

依赖：
  pip install langchain langchain-community langchain-huggingface faiss-cpu pymupdf sentence-transformers
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
RAW_DIR         = BASE_DIR / "data" / "raw_pdf"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-small-zh-v1.5"


# ── 1. 加载 PDF ─────────────────────────────────────────────────────────────────

def load_documents():
    from langchain_community.document_loaders import PyMuPDFLoader

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"data/raw_pdf/ 目录下没有 PDF 文件，请先运行 download_literature.py")

    all_docs = []
    for pdf_path in sorted(pdf_files):
        logger.info(f"加载: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs   = loader.load()

        # 从文件名解析元信息
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name

        all_docs.extend(docs)
        logger.info(f"  → {len(docs)} 页")

    logger.info(f"共加载 {len(all_docs)} 页（来自 {len(pdf_files)} 个文件）")
    return all_docs


# ── 2. 文本分块 ─────────────────────────────────────────────────────────────────

def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["。\n", "！\n", "？\n", "\n\n", "。", "！", "？", "；", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"分块完成：{len(docs)} 页 → {len(chunks)} 个 chunk")
    logger.info(f"平均 chunk 长度：{sum(len(c.page_content) for c in chunks)//len(chunks)} 字符")
    return chunks


# ── 3. Embedding 模型 ───────────────────────────────────────────────────────────

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    if not BGE_MODEL_PATH.exists():
        logger.warning(
            f"本地模型不存在: {BGE_MODEL_PATH}\n"
            "  将从 HuggingFace 下载（建议先运行 python src_langchain/download_model.py）"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Embedding 模型加载完成: {model_path}")
    return embeddings


# ── 4. 构建并保存 FAISS 向量库 ──────────────────────────────────────────────────

def build_vectorstore(chunks, embeddings):
    from langchain_community.vectorstores import FAISS

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"构建向量库（{len(chunks)} 个 chunk）...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"向量库已保存 → {VECTORSTORE_DIR}")
    return vectorstore


# ── 主流程 ──────────────────────────────────────────────────────────────────────

def main():
    docs   = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    build_vectorstore(chunks, embeddings)

    print(f"\nLangChain 向量库构建完成！")
    print(f"  路径: {VECTORSTORE_DIR}")
    print(f"  下一步: python src_langchain/rag_chain_lc.py")


if __name__ == "__main__":
    main()
