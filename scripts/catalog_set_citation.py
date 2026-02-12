#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Any, Dict


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _set_citation(conn: sqlite3.Connection, doc_id: str, citation: str) -> None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT metadata_json FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"doc_id not found: {doc_id}")

    metadata = _parse_metadata(row[0])
    metadata["citation"] = citation

    cur.execute(
        "UPDATE documents SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
        (json.dumps(metadata, ensure_ascii=False), doc_id),
    )
    conn.commit()
    print(f"OK updated doc_id={doc_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set document citation into documents.metadata_json['citation'].")
    ap.add_argument("--db", default="catalog.sqlite3", help="Path to catalog SQLite DB")
    ap.add_argument("--doc-id", required=True, help="Document ID to update")
    ap.add_argument("--citation", required=True, help="Citation string (PubMed/Vancouver style)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        doc_id = args.doc_id.strip()
        citation = args.citation.strip()
        if not doc_id or not citation:
            raise SystemExit("--doc-id and --citation must be non-empty")
        _set_citation(conn, doc_id, citation)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
