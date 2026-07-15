"""
RAG 问答链（LangChain LCEL 版）— 文献问答版

与原生版（src/rag_pipeline.py）的对比：
┌──────────────────┬──────────────────────┬──────────────────────┐
│ 环节             │ 原生版               │ LangChain 版         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 检索             │ FAISS + BM25 混合    │ FAISS 单路           │
│ 排序             │ RRF + CrossEncoder   │ 相似度得分直接排序   │
│ 链路组织         │ 手写流程控制         │ LCEL pipe (|) 操作符 │
│ 代码量           │ ~300 行              │ ~120 行              │
└──────────────────┴──────────────────────┴──────────────────────┘

使用方式：
  python rag_chain_lc.py                        # 交互式
  python rag_chain_lc.py --query "Transformer的注意力机制"
  python rag_chain_lc.py --query "..." --with-sources

依赖：
  pip install langchain langchain-openai langchain-community langchain-huggingface faiss-cpu
  需配置环境变量 DASHSCOPE_API_KEY
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-small-zh-v1.5"

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen-plus"

SYSTEM_PROMPT = """你是一个专业的学术文献分析助手，专门回答关于学术论文的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的数据或理论
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体内容时，在句末标注来源文件名，如：注意力机制的核心计算方式为...（来源：Attention_Is_All_You_Need.pdf）
4. 技术描述要精确，学术概念不得随意解释
5. 回答简洁，重点突出"""


# ── 组件初始化 ────────────────────────────────────────────────────────────────

def get_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=DASHSCOPE_URL,
        temperature=0.1,
    )


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings):
    from langchain_community.vectorstores import FAISS

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在: {VECTORSTORE_DIR}\n"
            "请先运行: python src_langchain/build_index_lc.py"
        )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── LCEL 链构建 ───────────────────────────────────────────────────────────────

def build_chain(vectorstore):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta     = doc.metadata
            filename = meta.get("source_file", "")
            page     = meta.get("page", "")
            label    = f"[{i}] {filename}"
            if page:
                label += f" 第{page+1}页"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，标注来源文件。"),
    ])

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def build_chain_with_sources(vectorstore):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser

    llm       = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta  = doc.metadata
            label = f"[{i}] {meta.get('source_file', '')} 第{meta.get('page', 0)+1}页"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ])

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt | llm | StrOutputParser()
    )

    chain_with_sources = RunnableParallel(
        {
            "context":  retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    return chain_with_sources


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="文献 RAG 问答（LangChain LCEL 版）")
    parser.add_argument("--query",        type=str, default=None)
    parser.add_argument("--with-sources", action="store_true", help="输出结果时附带来源文档片段")
    args = parser.parse_args()

    logger.info("加载 embedding 模型...")
    embeddings  = get_embeddings()
    logger.info("加载向量库...")
    vectorstore = get_vectorstore(embeddings)

    if args.with_sources:
        chain = build_chain_with_sources(vectorstore)
    else:
        chain, _ = build_chain(vectorstore)

    def run_query(question: str):
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        if args.with_sources:
            result  = chain.invoke(question)
            answer  = result["answer"]
            sources = result["context"]
            print(f"\n{answer}")
            print("\n── 来源文档片段 ──")
            for i, doc in enumerate(sources, 1):
                meta = doc.metadata
                print(f"[{i}] {meta.get('source_file', '')} 第{meta.get('page', 0)+1}页")
                print(f"    {doc.page_content[:120]}...")
        else:
            answer = chain.invoke(question)
            print(f"\n{answer}")

    if args.query:
        run_query(args.query)
    else:
        print(f"文献 RAG 问答系统（LangChain LCEL 版）")
        print(f"模型：{LLM_MODEL}  |  向量库：{VECTORSTORE_DIR}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "exit":
                break
            run_query(q)


if __name__ == "__main__":
    main()
