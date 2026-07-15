"""全局配置"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
META_PATH = VECTORSTORE_DIR / "meta.json"

# DashScope API（Embedding + LLM 共用）
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 10
LLM_MODEL = "qwen-plus"

# 分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索参数
TOP_K_RETRIEVE = 10   # 向量/BM25 各自初始召回数
TOP_K = 4             # 融合后送给 LLM 的数量
SCORE_THRESHOLD = 0.25  # 向量最高分低于此值时拒绝回答
RRF_K = 60            # RRF 融合参数

SYSTEM_PROMPT = """你是一个医学科普问答助手，基于知识库回答常见疾病和症状相关问题。

重要规则：
1. 只根据【参考资料】回答，不得编造资料外的医学信息
2. 明确声明：你的回答仅供参考，不能替代医生诊断和治疗
3. 遇到急重症信号（如剧烈胸痛、突发剧烈头痛、呼吸困难等），必须建议立即就医或拨打 120
4. 不提供具体用药剂量，涉及用药时建议"请遵医嘱"
5. 引用内容时标注来源编号，如：发热超过 39°C 应就医[1]
6. 回答简洁、条理清晰，使用通俗语言"""
