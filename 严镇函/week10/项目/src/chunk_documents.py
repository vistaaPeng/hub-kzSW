"""
文档分块脚本：将解析后的刑法 JSON 按法条结构分块

分块策略（针对法律条文）：
  策略A  按法条分块  —— 每条法条作为一个 chunk（默认）
  策略B  长法条切分  —— 超过800字的法条，按"款"（换行处）切分

输出格式：
  每个 chunk 是一个 dict：
    chunk_id      唯一标识（如 "第一条_001"）
    content       文本内容（供 embedding）
    metadata      元信息（供过滤/溯源）
      - article_num  条号
      - page_num     来源页码
      - section_path 章节路径
      - strategy     分块策略名
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
#路径配置
BASE_DIR    = Path(__file__).parent.parent
PARSED_DIR  = BASE_DIR / "data" / "parsed"
CHUNKS_DIR  = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# 分块策略配置
STRATEGY      = "by_article"   # 分块策略：by_article（按条）| by_paragraph（按段落）
MAX_CHUNK_LEN = 800            # 超过这个长度的法条，才切分
# ── 分块策略 A：按法条分块 ──────────────────────────────────────────────────

def chunk_by_article(blocks: list[dict]) -> list[dict]:
    """
    策略A：每条法条（article）作为 1 个 chunk

    这是最自然的策略，因为：
    - 法律文书中，"第X条"是最小引用单位
    - 律师/法官提问时都说"根据刑法第X条"
    - 每个 chunk 的长度适中（平均 200~500 字）

    参数 blocks: 从 JSON 读取的 blocks 列表
    返回 chunks: 分块后的 chunk 列表
    """
    chunks = []

    for block in blocks:
        btype = block["block_type"]

        # 跳过标题（编/章/节）和普通文字（目录等）
        if btype != "article":
            continue

        # 提取法条信息
        content     = block["content"]
        article_num = block.get("article_num", "")
        page_num    = block.get("page_num", 0)
        section_path = block.get("section_path", [])

        # 构建 chunk_id：如 "第一条_001"
        # 从 article_num 提取数字部分，比如"第一条"→"01"
        num_str = extract_article_number(article_num)
        chunk_id = f"{num_str}" if num_str else f"block_{len(chunks):05d}"

        # 构建 chunk
        chunk = {
            "chunk_id": chunk_id,
            "content":  content,
            "metadata": {
                "article_num":  article_num,
                "page_num":     page_num,
                "section_path": " > ".join(section_path) if section_path else "",
                "block_type":   "article",
                "strategy":     "by_article",
            }
        }
        chunks.append(chunk)

    return chunks
# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def extract_article_number(article_num: str) -> str:
    """
    从"第四百五十二条"提取数字，转为"452"
    用于生成排序友好的 chunk_id

    示例：
      "第一条"      → "001"
      "第一百条"    → "100"
      "第四百五十二条" → "452"
    """
    # 中文数字映射表
    chinese_num = {
        "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        "十": 10, "百": 100, "千": 1000,
    }

    # 去掉"第"和"条"，只保留中间的数字部分
    # 比如"第四百五十二条" → "四百五十二"
    num_part = article_num.replace("第", "").replace("条", "").strip()

    if not num_part:
        return ""

    # 中文数字转阿拉伯数字
    result = 0
    temp = 0
    for char in num_part:
        if char in chinese_num:
            value = chinese_num[char]
            if value >= 10:          # 十、百、千 → 乘到 temp 上
                if temp == 0:
                    temp = 1
                result += temp * value
                temp = 0
            else:                    # 一~九 → 累加到 temp
                temp += value
        else:
            # 遇到非中文数字字符（如空格），直接跳过
            continue

    result += temp  # 加上末尾的个位数

    # 格式化为3位数字，如"1"→"001"，"452"→"452"
    return f"{result:03d}"


# ── 分块策略 B：按段落切分长法条 ────────────────────────────────────────────

def chunk_by_paragraph(blocks: list[dict], max_len: int = MAX_CHUNK_LEN) -> list[dict]:
    """
    策略B：法条太长时，按段落（换行符）切分成多个子 chunk

    适用于：
      - "第二百三十四条 故意伤害罪"（好几款，很长）
      - 每款作为一个独立 chunk，携带相同的 article_num

    参数 blocks: blocks 列表
    参数 max_len: 超过此长度的法条才切分
    返回 chunks: 分块后的 chunk 列表
    """
    chunks = []
    sub_idx = 0  # 子 chunk 编号

    for block in blocks:
        btype = block["block_type"]
        if btype != "article":
            continue

        content     = block["content"]
        article_num = block.get("article_num", "")
        page_num    = block.get("page_num", 0)
        section_path = block.get("section_path", [])
        num_str     = extract_article_number(article_num)

        # 如果法条不长，整条作为一个 chunk
        if len(content) <= max_len:
            chunk_id = f"{num_str}" if num_str else f"block_{len(chunks):05d}"
            chunks.append({
                "chunk_id": chunk_id,
                "content":  content,
                "metadata": {
                    "article_num":  article_num,
                    "page_num":     page_num,
                    "section_path": " > ".join(section_path) if section_path else "",
                    "block_type":   "article",
                    "strategy":     "by_paragraph",
                }
            })
            continue

        # 长法条：按换行符切分（刑法条文里，换行通常表示"款"的分界）
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

        if len(paragraphs) <= 1:
            # 没有换行，但长度超过阈值 → 按字符切
            chunk_id = f"{num_str}" if num_str else f"block_{len(chunks):05d}"
            chunks.append({
                "chunk_id": chunk_id,
                "content":  content,
                "metadata": {
                    "article_num":  article_num,
                    "page_num":     page_num,
                    "section_path": " > ".join(section_path) if section_path else "",
                    "block_type":   "article",
                    "strategy":     "by_paragraph",
                }
            })
        else:
            # 每"款"作为一个子 chunk
            for i, para in enumerate(paragraphs):
                sub_idx += 1
                chunk_id = f"{num_str}_{i+1:02d}" if num_str else f"block_{sub_idx:05d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content":  para,
                    "metadata": {
                        "article_num":   article_num,
                        "page_num":      page_num,
                        "section_path":  " > ".join(section_path) if section_path else "",
                        "block_type":    "article",
                        "sub_paragraph": i + 1,  # 第几款
                        "strategy":      "by_paragraph",
                    }
                })

    return chunks


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    """主函数：加载解析结果 → 分块 → 保存"""

    # 查找解析后的 JSON 文件
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error(f"找不到解析结果，请先运行 parse_pdf.py")
        return

    parsed_path = parsed_files[0]
    logger.info(f"加载解析结果: {parsed_path.name}")

    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", [])
    meta   = data.get("meta", {})
    logger.info(f"共 {len(blocks)} 个 block（{meta.get('article_count', 0)} 条法条）")

    # 根据策略分块
    if STRATEGY == "by_article":
        chunks = chunk_by_article(blocks)
    elif STRATEGY == "by_paragraph":
        chunks = chunk_by_paragraph(blocks)
    else:
        logger.error(f"未知策略: {STRATEGY}")
        return

    logger.info(f"分块完成：{len(blocks)} blocks → {len(chunks)} chunks")

    # 统计信息
    total_len = sum(len(c["content"]) for c in chunks)
    avg_len   = total_len // len(chunks) if chunks else 0
    logger.info(f"平均 chunk 长度: {avg_len} 字")

    # 保存分块结果
    output_path = CHUNKS_DIR / f"chunks_{STRATEGY}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "source":         parsed_path.name,
                "strategy":       STRATEGY,
                "total_chunks":   len(chunks),
                "avg_chunk_len":  avg_len,
            },
            "chunks": chunks,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"分块结果已保存 → {output_path}")
    logger.info(f"文件大小: {output_path.stat().st_size // 1024} KB")

    # 打印前5个 chunk 预览
    print(f"\n── 分块结果预览（策略={STRATEGY}，前5条）──")
    for chunk in chunks[:5]:
        cid     = chunk["chunk_id"]
        content = chunk["content"][:60]
        art     = chunk["metadata"].get("article_num", "")
        print(f"  [{cid}] {art}  {content}...")


# ── 程序入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()