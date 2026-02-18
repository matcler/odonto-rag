#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set


def _load_plan(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Plan file not found: {path}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit("Plan root must be a JSON object")
    return obj


def _doc_specialties(conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not doc_ids:
        return out
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in doc_ids)
    q = (
        "SELECT doc_id, metadata_json FROM documents "
        f"WHERE doc_id IN ({placeholders})"
    )
    for row in cur.execute(q, doc_ids):
        doc_id = str(row[0] or "")
        metadata_raw = row[1]
        meta: Dict[str, Any] = {}
        if isinstance(metadata_raw, dict):
            meta = metadata_raw
        elif isinstance(metadata_raw, str) and metadata_raw.strip():
            try:
                parsed = json.loads(metadata_raw)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                meta = {}
        specialty = str(meta.get("specialty") or "").strip().lower()
        out[doc_id] = specialty
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate that slide plan bullets are source-grounded and specialty-coherent."
    )
    ap.add_argument("--plan-json", required=True, help="Path to /rag/slides/plan response JSON")
    ap.add_argument("--specialty", required=True, help="Expected specialty value")
    ap.add_argument("--db", default="catalog.sqlite3", help="Path to catalog SQLite DB")
    args = ap.parse_args()

    plan = _load_plan(args.plan_json)
    slides = plan.get("slides")
    if not isinstance(slides, list) or not slides:
        raise SystemExit("FAIL: plan has no slides")

    errors: List[str] = []
    source_doc_ids: Set[str] = set()

    for i, slide in enumerate(slides, start=1):
        if not isinstance(slide, dict):
            errors.append(f"slide {i}: not an object")
            continue
        bullets = [str(x).strip() for x in (slide.get("bullets") or []) if str(x).strip()]
        sources = [x for x in (slide.get("sources") or []) if isinstance(x, dict)]
        bullet_src = slide.get("bullet_source_item_ids") or []

        if not bullets:
            errors.append(f"slide {i}: no bullets")
        if not sources:
            errors.append(f"slide {i}: no sources")
        if not isinstance(bullet_src, list):
            errors.append(f"slide {i}: bullet_source_item_ids missing/not-list")
            continue
        if len(bullet_src) != len(bullets):
            errors.append(
                f"slide {i}: bullet_source_item_ids length {len(bullet_src)} != bullets {len(bullets)}"
            )
        else:
            for bi, item_ids in enumerate(bullet_src, start=1):
                if not isinstance(item_ids, list) or not [x for x in item_ids if str(x).strip()]:
                    errors.append(f"slide {i} bullet {bi}: no source item_ids")

        for src in sources:
            doc_id = str(src.get("doc_id") or "").strip()
            if doc_id:
                source_doc_ids.add(doc_id)

    expected = args.specialty.strip().lower()
    if source_doc_ids:
        conn = sqlite3.connect(args.db)
        try:
            specialties = _doc_specialties(conn, sorted(source_doc_ids))
        finally:
            conn.close()
        mismatches = []
        for doc_id in sorted(source_doc_ids):
            found = specialties.get(doc_id, "")
            if found != expected:
                mismatches.append(f"{doc_id}:{found or '<missing>'}")
        if mismatches:
            errors.append("specialty mismatches: " + ", ".join(mismatches))

    if errors:
        print("FAIL")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("PASS")
    print(f"slides={len(slides)}")
    print("doc_ids=" + ",".join(sorted(source_doc_ids)))


if __name__ == "__main__":
    main()
