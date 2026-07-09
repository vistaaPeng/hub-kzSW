"""
刑法 PDF 解析脚本：将《中华人民共和国刑法》PDF 解析为结构化文本

教学重点：
  1. 法条结构识别：编→章→节→条→款→项的层级关系
  2. 条（Article）是核心检索单位，必须完整保留
  3. 每条法条带元信息（所属编/章/节），供 RAG 溯源
  4. 项的编号模式：（一）（二）（三）或 1. 2. 3.

依赖安装：
  pip install pdfplumber pymupdf
"""
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF
# ── 日志配置 ──────────────────────────────────────────────────────────────────
# 配置日志格式：显示"时间 [级别] 消息"，比如 2026-07-09 10:00:00 [INFO] 开始解析...
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# ── 路径配置 ──────────────────────────────────────────────────────────────────
# __file__ 是当前脚本的绝对路径（如 D:/.../项目/src/parse_pdf.py）
# .parent 是上一级目录（src/），再 .parent 就是项目根目录（项目/）
#D:\PythonStudy\yzh_study\严镇函\week10\项目
BASE_DIR    = Path(__file__).parent.parent
print(BASE_DIR)

# 放原始 PDF 的目录：项目/data/raw_pdf/
RAW_DIR     = BASE_DIR / "data" / "raw_pdf"

# 解析结果输出目录：项目/data/parsed/
PARSED_DIR  = BASE_DIR / "data" / "parsed"

# 自动创建 parsed 文件夹（如果不存在的话），parents=True 表示连父目录一起创建
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# 备用路径：如果 PDF 直接放在 data/ 根目录下也能找到
FALLBACK_PDF = BASE_DIR / "data" / "中华人民共和国刑法.pdf"

# ── 正则模式 ──────────────────────────────────────────────────────────────────
# 这些正则表达式用来识别"编→章→节→条"的层级结构

# 编：匹配"第一编""第二编""第三编"
# ^ 表示行开头，[一二三四] 匹配一~四（刑法只有三编），\s+ 匹配后面的空格
PART_PATTERN = re.compile(r"^第[一二三四]编\s+")

# 章：匹配"第一章""第二章"……"第一百章"
# [一二三四五六七八九十百千]+ 可以匹配任意中文数字组合
CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百千]+章\s+")

# 节：匹配"第一节""第二节"……
SECTION_PATTERN = re.compile(r"^第[一二三四五六七八九十百千]+节\s+")

# 条：匹配"第一条""第二条"……"第四百五十二条"
# 注意：刑法条文编号是中文数字，最多到"第四百五十二条"
ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零]+[条]")

# 项：匹配"（一）""（二）"……"（十）""（十一）"
# [（(] 匹配中文括号或英文括号，[）)] 同理
ITEM_PATTERN = re.compile(r"^[（(][一二三四五六七八九十百千]+[）)]")

# 数字项：匹配"1.""2.""3."（刑法条文里有时也出现）
DIGIT_ITEM_PATTERN = re.compile(r"^\d+\.")

# 噪声模式：匹配独立页码等不需要的内容
# 比如单独一行的"38"或者"— 38 —"
NOISE_PATTERNS = [
    re.compile(r"^\d+\s*$"),          # 独立的数字（页码）
    re.compile(r"^—\s*\d+\s*—$"),    # — 38 —
    re.compile(r"^[-—]\s*\d+\s*[-—]$"),  # - 38 -
]
# ═══════════════════════════════════════════════════════════════════════════════
# 第3块：数据结构 + 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 一条法条 or 章节标题

    保留 part/chapter/section/article_num 非常重要——
    RAG 答案引用时能告诉用户"来自《刑法》第二编第三章第一节"
    """
    block_type:   str          # 类型："title"（标题）| "article"（法条）| "text"（普通文字）
    content:      str          # 文本内容（法条全文）
    page_num:     int          # 来源页码，供溯源用
    part:         str = ""     # 所属"编"：如"第一编 总则"
    chapter:      str = ""     # 所属"章"：如"第一章"
    section:      str = ""     # 所属"节"：如"第一节 犯罪和刑事责任"
    article_num:  str = ""     # 条号：如"第一条"
    section_path: list[str] = field(default_factory=list)  # 完整路径，如["第一编","第一章","第一条"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def is_noise_line(line: str) -> bool:
    """
    判断一行文字是不是"噪声"（页码、空行等，不需要的内容）

    参数 line: 从 PDF 里提取的一行文字
    返回 True 表示"这是噪声，应该跳过"
    """
    line = line.strip()          # 去掉首尾空格
    if not line or len(line) < 2:   # 空行或只有1个字符 → 噪声
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)  # 匹配任一噪声模式 → 噪声


def is_title_line(line: str) -> bool:
    """
    判断一行是不是标题（编/章/节）

    参数 line: 一行文字
    返回 True 表示"这是编/章/节标题"
    """
    line = line.strip()
    # 只要匹配 编/章/节 三种模式之一，就认为是标题
    return bool(PART_PATTERN.match(line)
                or CHAPTER_PATTERN.match(line)
                or SECTION_PATTERN.match(line))


def extract_title_type(line: str) -> tuple[str, str]:
    """
    判断一行是"编"还是"章"还是"节"标题，并返回标题文本

    参数 line: 一行文字
    返回 (类型, 标题文本)
      类型: "part"（编） | "chapter"（章） | "section"（节） | "text"（普通文字）
    """
    if PART_PATTERN.match(line):
        return ("part", line.strip())      # 比如 ("part", "第一编 总则")
    elif CHAPTER_PATTERN.match(line):
        return ("chapter", line.strip())   # 比如 ("chapter", "第一章 刑法的任务、基本原则和适用范围")
    elif SECTION_PATTERN.match(line):
        return ("section", line.strip())   # 比如 ("section", "第一节 犯罪和刑事责任")
    return ("text", line.strip())          # 都不是 → 普通文字
# ═══════════════════════════════════════════════════════════════════════════════
# 第4块：主解析器类（CriminalLawParser）
# ═══════════════════════════════════════════════════════════════════════════════

class CriminalLawParser:
    """
    《刑法》PDF 解析器——核心类！

    策略：
      - 用 PyMuPDF (fitz) 提取文字
      - 逐行扫描，识别"编→章→节→条"的结构
      - 每条法条作为一个独立的 block，携带元信息（所属编/章/节）

    关键设计：
      - _buffer_lines：临时缓冲区，先攒着文字，遇到"下一条"才 flush 出去
      - 这样能保证一条法条的所有内容都归在一起，不会漏
    """

    def __init__(self, pdf_path: Path):
        """
        初始化解析器

        参数 pdf_path: PDF 文件的路径（Path 对象）
        """
        self.pdf_path = pdf_path
        self.blocks: list[ParsedBlock] = []  # 存放所有解析结果

        # ── 状态变量（跟踪当前读到哪了）──
        self._current_part = ""       # 当前所在的"编"，如"第一编 总则"
        self._current_chapter = ""    # 当前所在的"章"，如"第一章"
        self._current_section = ""    # 当前所在的"节"，如"第一节"
        self._current_article = ""    # 当前正在读的"条号"，如"第一条"
        self._buffer_lines = []       # 临时缓冲区，攒着法条内容

    def _build_section_path(self) -> list[str]:
        """
        构建完整的章节路径

        比如当前在"第一编 总则 > 第一章 > 第一条"
        就返回 ["第一编 总则", "第一章", "第一条"]

        这个路径在 RAG 回答时用来告诉用户"答案来自第X编第X章第X条"
        """
        path = []
        if self._current_part:
            path.append(self._current_part)
        if self._current_chapter:
            path.append(self._current_chapter)
        if self._current_section:
            path.append(self._current_section)
        if self._current_article:
            path.append(self._current_article)
        return path

    def _flush_buffer(self, page_num: int):
        """
        把缓冲区里的文字"正式存档"为一个 ParsedBlock

        什么时候调用？当遇到新的"条"或"章"标题时，说明当前法条结束了，
        要把草稿纸（_buffer_lines）上的内容正式写到笔记本（blocks）里

        参数 page_num: 当前页码，用于溯源
        """
        if not self._buffer_lines:    # 草稿纸是空的 → 不用处理
            return

        # 把攒的几行文字合并成一段
        content = "".join(self._buffer_lines).strip()
        if not content:                # 合并后是空的 → 跳过
            self._buffer_lines = []
            return

        # 如果当前有条号，说明这是法条内容 → 追加到上一条 article block
        # （防止一条法条跨多页，内容被分散）
        if self._current_article and self.blocks and self.blocks[-1].block_type == "article":
            # 找到上一条 article，把新内容追加进去
            self.blocks[-1].content += "\n" + content
        else:
            # 没有条号 → 作为普通 text block 存档
            self.blocks.append(ParsedBlock(
                block_type="text",
                content=content,
                page_num=page_num,
                part=self._current_part,
                chapter=self._current_chapter,
                section=self._current_section,
                article_num=self._current_article,
                section_path=self._build_section_path(),
            ))

        # 清空草稿纸，准备攒下一条
        self._buffer_lines = []
    def parse(self) -> list[ParsedBlock]:
        """
        主解析入口——读取 PDF 的所有页面，逐行解析

        返回: list[ParsedBlock] 所有解析出来的块（标题+法条）
        """
        logger.info(f"📖 开始解析: {self.pdf_path.name}")

        # 检查文件是否存在
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"找不到 PDF 文件: {self.pdf_path}")

        # 用 PyMuPDF 打开 PDF 文件
        doc = fitz.open(str(self.pdf_path))
        logger.info(f"📄 PDF 共 {len(doc)} 页")

        # 逐页遍历
        for page_num in range(len(doc)):
            page = doc[page_num]

            # 获取当前页的文字（纯文本格式）
            page_text = page.get_text("text")

            # 如果页面文字太少（比如少于20个字），可能是扫描件或空白页 → 跳过
            if len(page_text.strip()) < 20:
                logger.warning(f"  ⚠️ 第{page_num+1}页文字极少，跳过")
                continue

            # 获取带布局信息的文字（dict 格式，包含段落信息）
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            # 逐 block 处理（fitz 的 block ≈ 段落）
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # type=0 是文字块，type=1 是图片块
                    continue

                # 把 block 内的所有行合并成一段完整文字
                block_text = ""
                for line in block.get("lines", []):
                    # 一行可能由多个 span（片段）组成，比如"第一"+"条"被拆成两段
                    # 这里把它们拼回来
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    )
                    block_text += line_text + "\n"  # 每行加换行符

                block_text = block_text.strip()
                if not block_text:    # 空的 block → 跳过
                    continue

                # 逐行判断（一个 block 可能包含多行）
                lines = block_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line or is_noise_line(line):   # 空行或噪声 → 跳过
                        continue

                    # ═══════════════════════════════════════════════
                    # 下面就是"决策树"：判断这一行是什么类型
                    # ═══════════════════════════════════════════════

                    # ① 是"编"标题吗？（如"第一编 总则"）
                    if PART_PATTERN.match(line):
                        self._flush_buffer(page_num + 1)          # 先存档当前内容
                        self._current_part = line                 # 更新"当前编"
                        self._current_chapter = ""                # 进入新编，重置章/节/条
                        self._current_section = ""
                        self._current_article = ""
                        self.blocks.append(ParsedBlock(           # 把编标题存为一个 block
                            block_type="title",
                            content=line,
                            page_num=page_num + 1,
                            part=self._current_part,
                            section_path=self._build_section_path(),
                        ))
                        continue

                    # ② 是"章"标题吗？（如"第一章 刑法的任务、基本原则和适用范围"）
                    if CHAPTER_PATTERN.match(line):
                        self._flush_buffer(page_num + 1)          # 先存档
                        self._current_chapter = line              # 更新"当前章"
                        self._current_section = ""                # 进入新章，重置节/条
                        self._current_article = ""
                        self.blocks.append(ParsedBlock(           # 存为 title block
                            block_type="title",
                            content=line,
                            page_num=page_num + 1,
                            part=self._current_part,
                            chapter=self._current_chapter,
                            section_path=self._build_section_path(),
                        ))
                        continue

                    # ③ 是"节"标题吗？（如"第一节 犯罪和刑事责任"）
                    if SECTION_PATTERN.match(line):
                        self._flush_buffer(page_num + 1)          # 先存档
                        self._current_section = line              # 更新"当前节"
                        self._current_article = ""                # 进入新节，重置条
                        self.blocks.append(ParsedBlock(           # 存为 title block
                            block_type="title",
                            content=line,
                            page_num=page_num + 1,
                            part=self._current_part,
                            chapter=self._current_chapter,
                            section=self._current_section,
                            section_path=self._build_section_path(),
                        ))
                        continue

                    # ④ 是"条"标题吗？（如"第一条"）—— 这是最核心的！
                    if ARTICLE_PATTERN.match(line):
                        self._flush_buffer(page_num + 1)          # 先存档→上一条的内容正式保存
                        # 提取条号，比如"第一条"→ article_num = "第一条"
                        match = ARTICLE_PATTERN.match(line)
                        self._current_article = match.group(0).strip()
                        self.blocks.append(ParsedBlock(           # 存为 article block
                            block_type="article",
                            content=line,
                            page_num=page_num + 1,
                            part=self._current_part,
                            chapter=self._current_chapter,
                            section=self._current_section,
                            article_num=self._current_article,
                            section_path=self._build_section_path(),
                        ))
                        continue

                    # ⑤ 普通文字 → 先攒到缓冲区，等遇到下一条时再 flush
                    self._buffer_lines.append(line)

            # 翻页了！把当前页缓冲区的内容 flush 掉
            # 防止一条法条跨页时内容丢失
            self._flush_buffer(page_num + 1)

        # 关闭 PDF 文件
        doc.close()
        logger.info(f"✅ 解析完成！共 {len(self.blocks)} 个 block")
        return self.blocks

# ═══════════════════════════════════════════════════════════════════════════════
# 第5块：查找 PDF + 主入口 main()
# ═══════════════════════════════════════════════════════════════════════════════

def find_pdf() -> Path:
    """
    自动查找刑法 PDF 文件

    查找顺序：
      1. 优先找 data/raw_pdf/ 目录下的 .pdf 文件
      2. 如果没找到，找 data/ 根目录下的 中华人民共和国刑法.pdf
      3. 都找不到 → 报错
    """
    # 在 raw_pdf/ 目录下搜所有 .pdf 文件
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if pdf_files:
        logger.info(f"找到 PDF: {pdf_files[0].name}")
        return pdf_files[0]

    # 备用路径
    if FALLBACK_PDF.exists():
        logger.info(f"找到 PDF: {FALLBACK_PDF.name}")
        return FALLBACK_PDF

    # 都没找到 → 抛出错误，告诉用户该怎么做
    raise FileNotFoundError(
        f"未找到 PDF 文件！\n"
        f"  请将刑法 PDF 放在 {RAW_DIR} 或 {FALLBACK_PDF}"
    )


def main():
    """
    主函数：解析 PDF → 保存为 JSON

    完整流程：
      1. 查找 PDF 文件
      2. 创建解析器
      3. 执行解析
      4. 统计结果
      5. 保存为 JSON 文件
      6. 打印预览
    """
    # 第1步：找到 PDF
    pdf_path = find_pdf()

    # 第2步：创建解析器
    parser = CriminalLawParser(pdf_path)

    # 第3步：执行解析
    blocks = parser.parse()

    # 第4步：统计
    article_count = sum(1 for b in blocks if b.block_type == "article")  # 数一数有多少条法条
    title_count = sum(1 for b in blocks if b.block_type == "title")      # 数一数有多少标题
    logger.info(f"  标题: {title_count} 个")
    logger.info(f"  法条: {article_count} 条")

    # 第5步：输出 JSON
    output = {
        "meta": {
            "source": pdf_path.name,           # 原始文件名
            "total_blocks": len(blocks),       # 总块数
            "article_count": article_count,    # 法条数
            "title_count": title_count,         # 标题数
        },
        "blocks": [asdict(b) for b in blocks],  # 把 ParsedBlock 转成字典（JSON 可序列化）
    }

    # 保存到 data/parsed/ 目录
    output_path = PARSED_DIR / f"{pdf_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"解析结果已保存 → {output_path}")
    logger.info(f"文件大小: {output_path.stat().st_size // 1024} KB")

    # 第6步：打印前5条作为预览
    print("\n── 解析结果预览（前5条）──")
    for b in blocks[:5]:
        label = f"[{b.block_type}]"
        if b.article_num:
            label += f" {b.article_num}"
        print(f"  {label}  {b.content[:80]}...")   # 只显示前80个字符
        if b.section_path:
            print(f"      路径: {' > '.join(b.section_path)}")


# ── 程序入口 ──────────────────────────────────────────────────────────────────
# 只有直接运行这个脚本时才会执行 main()
# 如果是被其他脚本 import，不会自动运行
if __name__ == "__main__":
    main()
