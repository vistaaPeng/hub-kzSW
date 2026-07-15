import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import pdfplumber
import fitz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PDF_DIR    = Path(__file__).parent.parent / "data" / "PDF"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed_tech"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TechBlock:
    block_type:   str
    content:      str
    page_num:     int
    section_path: list[str] = field(default_factory=list)
    raw_table:    Optional[list] = field(default=None, repr=False)


def table_to_markdown(table: list[list]) -> str:
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


class TechManualParser:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.blocks: list[TechBlock] = []

    def parse(self) -> list[TechBlock]:
        logger.info(f"解析技术手册: {self.pdf_path.name}")

        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page  = fitz_doc[page_num]
            plumb_page = plumber_doc.pages[page_num]

            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md:
                        self.blocks.append(TechBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            raw_table=table,
                        ))

            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            current_lines = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if line_text:
                        current_lines.append(line_text)

            if current_lines:
                self.blocks.append(TechBlock(
                    block_type="text",
                    content="\n".join(current_lines),
                    page_num=page_num + 1,
                ))

        plumber_doc.close()
        fitz_doc.close()
        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def save(self):
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "source": str(self.pdf_path),
            "filename": self.pdf_path.name,
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"在 {PDF_DIR} 中未找到 PDF 文件")
        return

    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
    for pdf_path in pdf_files:
        parser = TechManualParser(pdf_path)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
