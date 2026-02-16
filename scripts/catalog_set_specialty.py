#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from typing import Any, Dict


_SPECIALTY_RE = re.compile(r"^[a-z0-9_]+$")


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


def _normalize_specialty(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        raise SystemExit("--specialty must be non-empty")
    if not _SPECIALTY_RE.match(normalized):
        raise SystemExit(
            "--specialty must contain only lowercase letters, numbers, underscores"
        )
    return normalized


def _set_specialty(conn: sqlite3.Connection, doc_id: str, specialty: str) -> None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT metadata_json FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"doc_id not found: {doc_id}")

    metadata = _parse_metadata(row[0])
    metadata["specialty"] = specialty

    cur.execute(
        "UPDATE documents SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
        (json.dumps(metadata, ensure_ascii=False), doc_id),
    )
    conn.commit()
    print(f"OK updated doc_id={doc_id} specialty={specialty}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Set documents.metadata_json['specialty'] for an existing doc_id."
    )
    ap.add_argument("--db", default="catalog.sqlite3", help="Path to catalog SQLite DB")
    ap.add_argument("--doc-id", required=True, help="Document ID to update")
    ap.add_argument(
        "--specialty",
        required=True,
        help="Specialty value (e.g. implantology, endodontics)",
    )
    args = ap.parse_args()

    doc_id = args.doc_id.strip()
    specialty = _normalize_specialty(args.specialty)
    if not doc_id:
        raise SystemExit("--doc-id must be non-empty")

    conn = sqlite3.connect(args.db)
    try:
        _set_specialty(conn, doc_id, specialty)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
