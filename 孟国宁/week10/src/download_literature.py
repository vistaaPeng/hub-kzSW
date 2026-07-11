"""
文献导入脚本：将本地文献 PDF 导入项目数据目录

功能：
  1. 将用户放置的文献 PDF 从指定目录复制到 data/raw_pdf/
  2. 自动解析文件名，生成 manifest.json 元数据索引
  3. 支持多种命名格式：
     - 标准格式：作者_年份_标题.pdf
     - 简洁格式：标题.pdf

使用方式：
  1. 将有 PDF 复制到某个目录（如 D:/papers/）
  2. 运行：python download_literature.py --source D:/papers/
  3. 或者直接在 data/raw_pdf/ 下放 PDF，运行：python download_literature.py

命名建议（可选但推荐）：
  Attention Is All You Need_2017.pdf
  Vaswani_2017_Attention_Is_All_You_Need.pdf
  bert_2019.pdf
"""

import os
import json
import re
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw_pdf"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件扩展名
SUPPORTED_EXTS = {".pdf", ".PDF"}


def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符。"""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def infer_metadata(filename: str) -> dict:
    """
    从文件名推断文献元数据。
    支持多种格式，尽可能提取作者、年份、标题。
    """
    stem = Path(filename).stem
    meta = {
        "filename": filename,
        "title": stem,
        "authors": "",
        "year": "",
        "source": "local_import",
    }

    # 尝试匹配 "作者_年份_标题" 格式（下划线分隔）
    parts = stem.split("_")
    if len(parts) >= 2:
        # 检查是否有年份（4位数字）
        year_candidates = [p for p in parts if re.match(r"^(19|20)\d{2}$", p)]
        if year_candidates:
            meta["year"] = year_candidates[0]
            # 年份前的部分视为作者
            year_idx = parts.index(year_candidates[0])
            if year_idx > 0:
                meta["authors"] = parts[0]
            # 年份后的部分视为标题
            if year_idx < len(parts) - 1:
                meta["title"] = "_".join(parts[year_idx + 1:])

    # 尝试匹配 "标题_年份" 格式
    year_match = re.search(r"_(19|20)\d{2}", stem)
    if year_match and not meta["year"]:
        meta["year"] = year_match.group(0)[1:]

    return meta


def import_pdfs(source_dir: str = None) -> list[dict]:
    """
    导入文献 PDF。
    如果指定 source_dir，从该目录复制；否则扫描 data/raw_pdf/ 下的文件。
    """
    manifest = []

    if source_dir:
        src_path = Path(source_dir)
        if not src_path.is_dir():
            logger.error(f"源目录不存在: {src_path}")
            return manifest

        pdf_files = []
        for ext in SUPPORTED_EXTS:
            pdf_files.extend(src_path.glob(f"*{ext}"))

        if not pdf_files:
            logger.warning(f"源目录中没有找到 PDF 文件: {src_path}")
            return manifest

        for pdf_path in pdf_files:
            target_path = RAW_DIR / sanitize_filename(pdf_path.name)
            if target_path.exists():
                logger.info(f"已存在，跳过: {target_path.name}")
            else:
                shutil.copy2(pdf_path, target_path)
                logger.info(f"已导入: {target_path.name}")

            meta = infer_metadata(target_path.name)
            meta["source_path"] = str(pdf_path)
            manifest.append(meta)
    else:
        # 扫描 raw_pdf 目录已有文件
        for ext in SUPPORTED_EXTS:
            for pdf_path in RAW_DIR.glob(f"*{ext}"):
                meta = infer_metadata(pdf_path.name)
                meta["source_path"] = str(pdf_path)
                manifest.append(meta)

        if not manifest:
            logger.warning(
                f"data/raw_pdf/ 下没有 PDF 文件。\n"
                f"请手动将文献 PDF 放入 {RAW_DIR}，然后重新运行此脚本。"
            )

    # 保存 manifest
    manifest_path = RAW_DIR.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"\n共导入 {len(manifest)} 篇文献")
    for item in manifest:
        logger.info(f"  {item.get('authors', '?')} ({item.get('year', '?')}) {item.get('title', '?')}")

    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description="文献 PDF 导入工具")
    parser.add_argument("--source", type=str, default=None,
                        help="源目录路径，从该目录复制 PDF 到 data/raw_pdf/")
    args = parser.parse_args()

    import_pdfs(args.source)


if __name__ == "__main__":
    main()
