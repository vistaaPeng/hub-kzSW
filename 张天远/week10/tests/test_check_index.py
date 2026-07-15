"""
Tests for the structural index checker.
"""

import json
import pickle

from scripts.check_index import check_index


def write_index(tmp_path, children, all_chunks, bm25_chunks=None, faiss_bytes=b"fake"):
    (tmp_path / "children.json").write_text(
        json.dumps(children, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "all_chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    with open(tmp_path / "bm25.pkl", "wb") as f:
        pickle.dump({"chunks": bm25_chunks if bm25_chunks is not None else children}, f)
    (tmp_path / "faiss.index").write_bytes(faiss_bytes)


def parent(chunk_id="p1", source_url="url"):
    return {
        "chunk_id": chunk_id,
        "text": "parent text",
        "metadata": {"source_url": source_url},
        "parent_chunk_id": None,
        "is_parent": True,
    }


def child(chunk_id="c1", parent_id="p1", source_url="url"):
    return {
        "chunk_id": chunk_id,
        "text": "child text",
        "metadata": {"source_url": source_url},
        "parent_chunk_id": parent_id,
        "is_parent": False,
    }


def test_check_index_accepts_valid_structure_with_unreadable_fake_faiss(tmp_path):
    p = parent()
    c = child()
    write_index(tmp_path, [c], [p, c])

    report = check_index(tmp_path)

    assert report.ok
    assert report.stats["children"] == 1
    assert report.stats["parents"] == 1
    assert report.stats["bm25_doc_count"] == 1
    assert any("could not read faiss.index" in w for w in report.warnings)


def test_check_index_rejects_duplicate_chunk_ids(tmp_path):
    p = parent()
    c = child()
    write_index(tmp_path, [c, c], [p, c])

    report = check_index(tmp_path)

    assert not report.ok
    assert any("duplicate chunk_id" in e for e in report.errors)


def test_check_index_rejects_missing_parent_link(tmp_path):
    p = parent()
    c = child(parent_id="missing-parent")
    write_index(tmp_path, [c], [p, c])

    report = check_index(tmp_path)

    assert not report.ok
    assert any("child parent not found" in e for e in report.errors)


def test_check_index_rejects_missing_source_url(tmp_path):
    p = parent(source_url="")
    c = child(source_url="")
    write_index(tmp_path, [c], [p, c])

    report = check_index(tmp_path)

    assert not report.ok
    assert any("missing metadata.source_url" in e for e in report.errors)


def test_check_index_rejects_bm25_count_mismatch(tmp_path):
    p = parent()
    c = child()
    write_index(tmp_path, [c], [p, c], bm25_chunks=[])

    report = check_index(tmp_path)

    assert not report.ok
    assert any("bm25 chunks 0 != children count 1" in e for e in report.errors)
