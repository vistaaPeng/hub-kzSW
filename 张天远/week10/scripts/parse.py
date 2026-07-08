#!/usr/bin/env python
"""
HTML → 结构化 JSON —— 纯解析，不涉及分块
---
用法: python scripts/parse.py [--sources book,reference,...]
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import parse_html_to_json


def parse_all(data_dir: str = "data", sources: list[str] | None = None):
    data = Path(data_dir)
    raw_dir = data / "raw"
    parsed_dir = data / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if sources:
        source_dirs = [d for d in source_dirs if d.name in sources]

    total_pages = 0
    print(f"📂 解析 {len(source_dirs)} 个来源...")

    for src_dir in sorted(source_dirs):
        source_name = src_dir.name
        html_files = list(src_dir.glob("*.html"))
        if not html_files:
            continue

        out_dir = parsed_dir / source_name
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for html_path in html_files:
            try:
                html = html_path.read_text(encoding="utf-8")
                source_url = f"https://rustwiki.org/zh-CN/{source_name}/{html_path.name}"
                doc = parse_html_to_json(html, source_url, source_name)
                doc.to_json(str(out_dir / f"{html_path.stem}.json"))
                count += 1
            except Exception as e:
                print(f"  ⚠ {html_path.name}: {e}")

        total_pages += count
        print(f"  ✅ {source_name}: {count}/{len(html_files)} 页")

    print(f"\n📊 总计: {total_pages} JSON → {parsed_dir}")
    print("下一步: python scripts/build_index.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTML → JSON 解析")
    parser.add_argument("--sources", type=str, help="逗号分隔的来源，默认全部")
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None
    parse_all(sources=sources)
