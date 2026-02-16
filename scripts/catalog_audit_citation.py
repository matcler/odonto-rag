#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Any, Dict, List, Tuple


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def _load_docs(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT doc_id, gcs_raw_path, metadata_json FROM documents ORDER BY doc_id"
    ).fetchall()
    out: List[Tuple[str, str, str]] = []
    for doc_id, gcs_raw_path, metadata_raw in rows:
        metadata = _parse_metadata(metadata_raw)
        citation = str(metadata.get("citation") or "").strip()
        out.append((str(doc_id or ""), str(gcs_raw_path or ""), citation))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit catalog documents and report missing metadata_json['citation']."
    )
    ap.add_argument("--db", default="catalog.sqlite3", help="Path to catalog SQLite DB")
    ap.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with code 1 if one or more documents are missing citation.",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        docs = _load_docs(conn)
    finally:
        conn.close()

    total = len(docs)
    missing = [(doc_id, raw_uri) for doc_id, raw_uri, citation in docs if not citation]

    print(f"total_docs={total}")
    print(f"missing_citation={len(missing)}")

    if missing:
        print("missing_list:")
        for doc_id, raw_uri in missing:
            print(f"- doc_id={doc_id} raw_uri={raw_uri}")

    if args.fail_on_missing and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
