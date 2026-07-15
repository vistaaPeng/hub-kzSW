#!/usr/bin/env python
"""
数据下载脚本 —— 6 个 Rust 中文 mdBook 文档源 + Glossary 术语表
---
用法: python scripts/download.py [--sources book,reference,...]
所有源默认全部下载。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader import download_book
from src.glossary import download_glossary

SOURCES = {
    "book":             "https://rustwiki.org/zh-CN/book/",
    "reference":        "https://rustwiki.org/zh-CN/reference/",
    "rust-by-example":  "https://rustwiki.org/zh-CN/rust-by-example/",
    "rust-cookbook":    "https://rustwiki.org/zh-CN/rust-cookbook/",
    "edition-guide":    "https://rustwiki.org/zh-CN/edition-guide/",
    "rustdoc":          "https://rustwiki.org/zh-CN/rustdoc/",
}


def download_all(sources: list[str] | None = None, data_dir: str = "data"):
    """下载所有数据源 → data/raw/<source_name>/"""
    raw_dir = Path(data_dir) / "raw"
    selected = {k: v for k, v in SOURCES.items() if sources is None or k in sources}

    total = 0
    for name, url in selected.items():
        out_dir = raw_dir / name
        files = download_book(url, str(out_dir))
        total += len(files)
        print(f"  ✅ {name}: {len(files)} 页 → {out_dir}")

    print(f"\n📊 总计: {total} HTML → {raw_dir}")
    print("下一步: python scripts/parse.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="下载 Rust 中文文档")
    parser.add_argument("--sources", type=str,
                        help="逗号分隔的来源，默认全部。可选: " + ",".join(SOURCES))
    args = parser.parse_args()
    download_all(args.sources.split(",") if args.sources else None)
