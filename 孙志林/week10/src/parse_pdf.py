"""
PDF解析模块：提取PDF中的文字、表格和章节结构

组合策略：
  - pdfplumber：表格提取（对财务报表行列识别更准确）
  - PyMuPDF(fitz)：文字提取+字体元数据（用于识别标题层级）
  - pytesseract：扫描页OCR（可选）

解析输出：
  - title块：字体>14pt或加粗且行长<50字
  - table块：转为Markdown格式
  - text块：正常文本段落

每个块保留元数据：page_num, section_path, is_ocr, block_type
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdf"
PARSED_DIR = DATA_DIR / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_pdf_with_fitz(pdf_path: str):
    """使用PyMuPDF提取文字和字体信息"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
    except ImportError:
        logger.error("PyMuPDF未安装，请执行: pip install PyMuPDF")
        return []

    blocks = []
    section_stack = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_blocks = page.get_text("dict")["blocks"]

        for block in text_blocks:
            if block["type"] != 0:
                continue

            lines = block.get("lines", [])
            if not lines:
                continue

            content_parts = []
            font_sizes = []
            is_bold = False

            for line in lines:
                for span in line.get("spans", []):
                    content_parts.append(span["text"])
                    font_sizes.append(span.get("size", 12))
                    if "Bold" in span.get("font", "") or span.get("flags", 0) & 1:
                        is_bold = True

            content = "".join(content_parts).strip()
            if not content:
                continue

            avg_font_size = sum(font_sizes) / len(font_sizes)

            if is_bold or avg_font_size > 14:
                block_type = "title"
                text_len = len(content)
                if text_len <= 50:
                    if section_stack and len(content) < len(section_stack[-1]["content"]):
                        while section_stack and len(content) <= len(section_stack[-1]["content"]):
                            section_stack.pop()
                    section_stack.append({"content": content, "page": page_num})
            else:
                block_type = "text"

            blocks.append({
                "content": content,
                "block_type": block_type,
                "page_num": page_num + 1,
                "section_path": [s["content"] for s in section_stack],
                "is_ocr": False,
                "font_size": avg_font_size,
                "is_bold": is_bold,
            })

    doc.close()
    return blocks


def extract_tables_with_pdfplumber(pdf_path: str):
    """使用pdfplumber提取表格并转为Markdown"""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber未安装，跳过表格提取")
        return []

    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for table in page_tables:
                if not table or len(table) < 2:
                    continue

                md_lines = []
                header = table[0]
                md_lines.append("| " + " | ".join(str(c).strip() if c else "" for c in header) + " |")
                md_lines.append("|" + "---|" * len(header))

                for row in table[1:]:
                    md_lines.append("| " + " | ".join(str(c).strip() if c else "" for c in row) + " |")

                tables.append({
                    "content": "\n".join(md_lines),
                    "block_type": "table",
                    "page_num": page_num + 1,
                    "section_path": [],
                    "is_ocr": False,
                })

    return tables


def parse_pdf(pdf_path: str, stock_code: str, year: str):
    """完整解析PDF：文字+表格"""
    logger.info(f"Parsing PDF: {pdf_path}")

    text_blocks = parse_pdf_with_fitz(pdf_path)
    table_blocks = extract_tables_with_pdfplumber(pdf_path)

    all_blocks = sorted(text_blocks + table_blocks, key=lambda x: (x["page_num"], x.get("font_size", 0), x["content"]))

    parsed = {
        "stock_code": stock_code,
        "year": year,
        "pdf_path": pdf_path,
        "num_pages": max(b["page_num"] for b in all_blocks) if all_blocks else 0,
        "num_blocks": len(all_blocks),
        "blocks": all_blocks,
    }

    return parsed


def process_all_pdfs():
    """处理所有下载的PDF"""
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}\nPlease run download_reports.py first")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for item in manifest:
        pdf_path = item["pdf_path"]
        stock_code = item["stock_code"]
        year = item["year"]

        parsed = parse_pdf(pdf_path, stock_code, year)

        output_path = PARSED_DIR / f"{stock_code}_{year}_parsed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved parsed document: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDF documents")
    parser.add_argument("--pdf", type=str, help="Path to single PDF file")
    parser.add_argument("--stock", type=str, help="Stock code")
    parser.add_argument("--year", type=str, help="Year")
    args = parser.parse_args()

    if args.pdf and args.stock and args.year:
        parsed = parse_pdf(args.pdf, args.stock, args.year)
        output_path = PARSED_DIR / f"{args.stock}_{args.year}_parsed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to: {output_path}")
    else:
        process_all_pdfs()