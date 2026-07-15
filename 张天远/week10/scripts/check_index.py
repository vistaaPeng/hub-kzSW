#!/usr/bin/env python
"""
Validate persisted RAG index files.

Checks are deliberately structural and cheap: they catch broken parent links,
missing fields, duplicate IDs, and mismatched index document counts before a
query silently degrades.
"""

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_CHUNK_FIELDS = {"chunk_id", "text", "metadata", "parent_chunk_id", "is_parent"}


@dataclass
class IndexCheckReport:
    index_dir: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _read_json_array(path: Path, report: IndexCheckReport) -> list[dict[str, Any]]:
    if not path.exists():
        report.add_error(f"missing file: {path.name}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add_error(f"invalid JSON in {path.name}: {exc}")
        return []
    if not isinstance(data, list):
        report.add_error(f"{path.name} must contain a JSON array")
        return []
    return data


def _validate_chunk_records(name: str, records: list[dict[str, Any]],
                            report: IndexCheckReport) -> None:
    ids: set[str] = set()
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            report.add_error(f"{name}[{i}] is not an object")
            continue
        missing = REQUIRED_CHUNK_FIELDS - set(record)
        if missing:
            report.add_error(f"{name}[{i}] missing fields: {sorted(missing)}")
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            report.add_error(f"{name}[{i}] has empty chunk_id")
        elif chunk_id in ids:
            report.add_error(f"{name} duplicate chunk_id: {chunk_id}")
        else:
            ids.add(chunk_id)
        if not isinstance(record.get("text", ""), str) or not record.get("text", "").strip():
            report.add_warning(f"{name}[{i}] has empty text")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            report.add_error(f"{name}[{i}] metadata is not an object")
        elif not metadata.get("source_url"):
            report.add_error(f"{name}[{i}] missing metadata.source_url")


def _validate_parent_links(children: list[dict[str, Any]],
                           all_chunks: list[dict[str, Any]],
                           report: IndexCheckReport) -> None:
    parent_ids = {c["chunk_id"] for c in all_chunks if c.get("is_parent")}
    all_ids = {c["chunk_id"] for c in all_chunks}
    child_ids = {c["chunk_id"] for c in children}

    for child in children:
        if child.get("is_parent"):
            report.add_error(f"children.json contains parent chunk: {child.get('chunk_id')}")
        pid = child.get("parent_chunk_id")
        if not pid:
            report.add_error(f"child missing parent_chunk_id: {child.get('chunk_id')}")
        elif pid not in parent_ids:
            report.add_error(
                f"child parent not found: {child.get('chunk_id')} -> {pid}"
            )
        if child.get("chunk_id") not in all_ids:
            report.add_error(f"child absent from all_chunks.json: {child.get('chunk_id')}")

    for record in all_chunks:
        if not record.get("is_parent") and record.get("chunk_id") not in child_ids:
            report.add_error(f"non-parent chunk absent from children.json: {record.get('chunk_id')}")


def _validate_faiss_count(index_dir: Path, child_count: int,
                          report: IndexCheckReport) -> None:
    path = index_dir / "faiss.index"
    if not path.exists():
        report.add_error("missing file: faiss.index")
        return
    try:
        import faiss
        index = faiss.read_index(str(path))
    except Exception as exc:
        report.add_warning(f"could not read faiss.index: {exc}")
        return
    report.stats["faiss_ntotal"] = int(index.ntotal)
    if int(index.ntotal) != child_count:
        report.add_error(f"faiss.ntotal {index.ntotal} != children count {child_count}")


def _validate_bm25_count(index_dir: Path, child_count: int,
                         report: IndexCheckReport) -> None:
    path = index_dir / "bm25.pkl"
    if not path.exists():
        report.add_error("missing file: bm25.pkl")
        return
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        report.add_warning(f"could not read bm25.pkl: {exc}")
        return
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    report.stats["bm25_doc_count"] = len(chunks)
    if len(chunks) != child_count:
        report.add_error(f"bm25 chunks {len(chunks)} != children count {child_count}")


def check_index(index_dir: str | Path = "vectorstore") -> IndexCheckReport:
    """Validate an index directory and return a structured report."""
    index_dir = Path(index_dir)
    report = IndexCheckReport(index_dir=str(index_dir))

    children = _read_json_array(index_dir / "children.json", report)
    all_chunks = _read_json_array(index_dir / "all_chunks.json", report)

    _validate_chunk_records("children.json", children, report)
    _validate_chunk_records("all_chunks.json", all_chunks, report)
    if children and all_chunks:
        _validate_parent_links(children, all_chunks, report)

    parents = [c for c in all_chunks if c.get("is_parent")]
    report.stats.update({
        "children": len(children),
        "parents": len(parents),
        "all_chunks": len(all_chunks),
    })

    if children:
        _validate_faiss_count(index_dir, len(children), report)
        _validate_bm25_count(index_dir, len(children), report)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate persisted RAG indexes")
    parser.add_argument("--index-dir", default="vectorstore")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = check_index(args.index_dir)
    payload = {
        "ok": report.ok,
        "index_dir": report.index_dir,
        "stats": report.stats,
        "errors": report.errors,
        "warnings": report.warnings,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = "OK" if report.ok else "FAILED"
        print(f"Index check: {status}")
        print(f"Stats: {report.stats}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
