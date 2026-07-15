"""
PDF 解析脚本：将学术文献 PDF 转换为结构化文本

与年报解析的核心差异：
  1. 学术论文的章节结构不同（摘要/引言/方法/实验/结论）
  2. 标题层级更规范（通常为 1. / 1.1 / 1.1.1）
  3. 包含公式、算法伪代码等特殊元素
  4. 参考文献列表需要保留引用索引

依赖安装：
  pip install pdfplumber pymupdf pytesseract pillow
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber          # 擅长表格提取
import fitz                # PyMuPDF，擅长文字+图片提取

# OCR 依赖可选
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_pdf"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 文献里的一段连续内容（文字段落 / 表格 / 标题）
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int
    section_path: list[str]      # ["1. 引言", "1.1 研究背景"]
    is_ocr:       bool = False
    raw_table:    Optional[list] = field(default=None, repr=False)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 学术论文常见章节标题模式
CHAPTER_PATTERNS = [
    re.compile(r"^\d+\.\d+\.\d+\s"),      # 1.1.1
    re.compile(r"^\d+\.\d+\s"),            # 1.1
    re.compile(r"^\d+\.\s"),               # 1.
    re.compile(r"^第[一二三四五六七八九十]+[章节]"),  # 第一章
    re.compile(r"^[一二三四五六七八九十]、"),          # 一、
]

# 论文中常见的章节关键词
SECTION_KEYWORDS = [
    "摘要", "abstract",
    "引言", "introduction",
    "相关工作", "related work",
    "方法", "method", "approach",
    "实验", "experiment", "experimental",
    "结果", "results",
    "讨论", "discussion",
    "结论", "conclusion",
    "参考文献", "references",
    "附录", "appendix",
]

NOISE_PATTERNS = [
    re.compile(r"^.{1,60}(会议|conference|workshop|journal|学报)\s*$", re.I),
    re.compile(r"^\d+\s*$"),              # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),        # — 38 —
    re.compile(r"^(arXiv|DOI|http).*", re.I),  # arXiv/DOI/URL 行
]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    """
    判断一行是否是标题。
    学术论文的标题通常：字体较大、加粗、或匹配编号模式。
    """
    if fontsize and fontsize >= 13:       # 学术论文标题通常 13-16pt
        return True
    if is_bold and len(line.strip()) < 80:
        return True
    line_stripped = line.strip()
    if any(kw.lower() in line_stripped.lower() for kw in SECTION_KEYWORDS):
        if len(line_stripped) < 50:       # 避免匹配正文中包含关键词的句子
            return True
    return any(p.match(line_stripped) for p in CHAPTER_PATTERNS)


def table_to_markdown(table: list[list]) -> str:
    """把 pdfplumber 提取的表格转成 markdown 格式。"""
    if not table:
        return ""
    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)
    if not rows:
        return ""
    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


def detect_if_scanned(page: fitz.Page, text: str) -> bool:
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images(full=True)
    return len(image_list) > 0


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用，内容跳过]"
    try:
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        clip = page.rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        logger.warning(f"  OCR 失败，跳过此页: {e}")
        return "[扫描页，OCR 失败，内容跳过]"


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

class LiteratureParser:
    """
    学术文献 PDF 解析器。

    与年报解析器的区别：
      - 章节标题模式适配学术论文
      - 过滤 arXiv/DOI 等元信息噪声
      - 保留参考文献引用标记
    """

    def __init__(self, pdf_path: Path, meta: dict = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []

    def _update_section(self, title: str):
        """维护章节栈。"""
        if re.match(r"^\d+\.\d+\.\d+\s", title):
            self._section_stack = self._section_stack[:2] + [title]
        elif re.match(r"^\d+\.\d+\s", title):
            self._section_stack = self._section_stack[:1] + [title]
        elif re.match(r"^\d+\.\s", title):
            self._section_stack = [title]
        elif re.match(r"^第[一二三四五六七八九十]+章", title):
            self._section_stack = [title]
        elif re.match(r"^[一二三四五六七八九十]、", title):
            self._section_stack = self._section_stack[:2] + [title]
        else:
            self._section_stack = self._section_stack[:3] + [title]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page   = fitz_doc[page_num]
            plumb_page  = plumber_doc.pages[page_num]

            # ── 1. 检测扫描页 ──
            raw_text = fitz_page.get_text("text")
            is_scanned = detect_if_scanned(fitz_page, raw_text)

            if is_scanned:
                logger.debug(f"  第{page_num+1}页：检测到扫描件，启动 OCR")
                ocr_text = ocr_page(fitz_page)
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content=ocr_text,
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                    is_ocr=True,
                ))
                continue

            # ── 2. 提取表格（用 pdfplumber）──
            table_bboxes = []
            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md:
                        self.blocks.append(ParsedBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                            raw_table=table,
                        ))
            for table_obj in plumb_page.find_tables():
                table_bboxes.append(table_obj.bbox)

            # ── 3. 提取文字（逐行处理）──
            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            current_para_lines = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()

                    if not line_text or is_noise_line(line_text):
                        continue

                    spans    = line.get("spans", [])
                    fontsize = spans[0].get("size", 0) if spans else 0
                    is_bold  = any("Bold" in span.get("font", "") for span in spans)

                    if is_title_line(line_text, fontsize, is_bold):
                        if current_para_lines:
                            self.blocks.append(ParsedBlock(
                                block_type="text",
                                content="\n".join(current_para_lines),
                                page_num=page_num + 1,
                                section_path=list(self._section_stack),
                            ))
                            current_para_lines = []

                        self._update_section(line_text)
                        self.blocks.append(ParsedBlock(
                            block_type="title",
                            content=line_text,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                    else:
                        current_para_lines.append(line_text)

            if current_para_lines:
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content="\n".join(current_para_lines),
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                ))

        plumber_doc.close()
        fitz_doc.close()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def save(self):
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"
        output = {
            "meta":   self.meta,
            "source": str(self.pdf_path),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    manifest_path = RAW_DIR.parent / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = [
            {"filename": p.name}
            for p in RAW_DIR.glob("*.pdf")
        ]

    if not manifest:
        logger.error("没有找到任何 PDF，请先运行 download_literature.py")
        return

    for item in manifest:
        pdf_path = RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        parser = LiteratureParser(pdf_path, meta=item)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
