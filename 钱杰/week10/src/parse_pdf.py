import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdf"
PARSED_DIR = BASE_DIR / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PDFS = {"ai_basics.pdf", "climate_change.pdf", "python_programming.pdf", "space_exploration.pdf"}


def parse_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    blocks = []
    
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            lines = text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("第") and "章" in line:
                    blocks.append({
                        "block_type": "title",
                        "content": line,
                        "page_num": page_num + 1,
                        "section_path": [],
                        "is_ocr": False,
                    })
                elif line.isupper():
                    blocks.append({
                        "block_type": "title",
                        "content": line,
                        "page_num": page_num + 1,
                        "section_path": [],
                        "is_ocr": False,
                    })
                else:
                    blocks.append({
                        "block_type": "text",
                        "content": line,
                        "page_num": page_num + 1,
                        "section_path": [],
                        "is_ocr": False,
                    })
    except Exception as e:
        logger.error(f"解析失败: {e}")
    
    return blocks


def parse_single_pdf(pdf_path: Path) -> Dict[str, Any]:
    logger.info(f"解析: {pdf_path.name}")
    
    filename = pdf_path.stem
    
    meta = {
        "filename": pdf_path.name,
        "doc_type": filename.split('_')[0],
        "title": filename.replace('_', ' '),
    }
    
    blocks = parse_pdf(pdf_path)
    
    result = {
        "meta": meta,
        "blocks": blocks,
    }
    
    output_path = PARSED_DIR / f"{filename}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"完成: {output_path.name}, {len(blocks)} 个 blocks")
    return result


def main():
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
    
    target_files = [p for p in pdf_files if p.name in TARGET_PDFS]
    
    if not target_files:
        logger.warning("未找到目标PDF文件")
        return
    
    logger.info(f"找到 {len(target_files)} 个目标PDF文件")
    
    for pdf_path in target_files:
        parse_single_pdf(pdf_path)
    
    logger.info("所有PDF解析完成！")


if __name__ == "__main__":
    main()