#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import uuid
from typing import Optional

from odonto_rag.catalog.db import make_engine, make_session_factory
from odonto_rag.catalog.models import Document, DocumentVersion
from odonto_rag.storage_layout import ParsedLayout, gcs_uri


def _derive_specialty_from_raw_uri(raw_uri: str) -> Optional[str]:
    raw = (raw_uri or "").strip()
    if not raw:
        return None
    match = re.search(r"(?:^|/)raw/([^/]+)/", raw)
    if not match:
        return None
    specialty = match.group(1).strip()
    return specialty or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a document + version in SQLite with canonical GCS layout paths.")
    ap.add_argument("--doc-type", required=True, choices=["pdf", "pptx", "docx", "video"])
    ap.add_argument("--raw-uri", required=True, help="GCS URI of the original source (gs://bucket/path.ext)")
    ap.add_argument("--doc-id", default="", help="Optional stable doc_id. If empty, one is generated.")
    ap.add_argument("--version-id", default="v1", help="Version string, e.g. v1, v2, 2026-01-15")
    ap.add_argument(
        "--citation",
        default="",
        help="Document-level citation string (required unless already present for existing doc_id).",
    )
    ap.add_argument(
        "--specialty",
        default="",
        help="Optional specialty (e.g. endodontics). If empty, derived from raw URI pattern raw/<specialty>/...",
    )
    ap.add_argument("--parsed-bucket", default=os.environ.get("PARSED_BUCKET", ""), help="Parsed bucket name")
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_PATH", "catalog.sqlite3"))
    args = ap.parse_args()

    if not args.parsed_bucket:
        raise SystemExit("PARSED_BUCKET is empty. Pass --parsed-bucket or set PARSED_BUCKET env var.")

    doc_id = args.doc_id.strip() or str(uuid.uuid4())
    version_id = args.version_id.strip()
    specialty = args.specialty.strip() or (_derive_specialty_from_raw_uri(args.raw_uri) or "")
    citation = args.citation.strip()

    layout = ParsedLayout(doc_id=doc_id, version_id=version_id)

    parsed_prefix_uri = gcs_uri(args.parsed_bucket, layout.prefix())
    items_uri = gcs_uri(args.parsed_bucket, layout.items_jsonl())
    assets_uri = gcs_uri(args.parsed_bucket, layout.assets_jsonl())

    engine = make_engine(args.sqlite)
    SessionLocal = make_session_factory(engine)

    with SessionLocal() as sess:
        # Upsert Document
        doc = sess.query(Document).filter(Document.doc_id == doc_id).one_or_none()
        if doc is None:
            if not citation:
                raise SystemExit(
                    "Missing citation for new document. Pass --citation with a bibliographic string."
                )
            doc = Document(
                doc_id=doc_id,
                doc_type=args.doc_type,
                gcs_raw_path=args.raw_uri,
                gcs_parsed_prefix=parsed_prefix_uri,
                active_version=version_id,
                status="registered",
            )
            metadata: dict = {}
            if specialty:
                metadata["specialty"] = specialty
            if citation:
                metadata["citation"] = citation
            if metadata:
                doc.metadata_json = metadata
            sess.add(doc)
        else:
            doc.doc_type = args.doc_type
            doc.gcs_raw_path = args.raw_uri
            doc.gcs_parsed_prefix = parsed_prefix_uri
            doc.active_version = version_id
            doc.status = "registered"
            existing_meta = doc.metadata_json if isinstance(doc.metadata_json, dict) else {}
            merged_meta = dict(existing_meta)
            if specialty:
                merged_meta["specialty"] = specialty
            if citation:
                merged_meta["citation"] = citation
            if not str(merged_meta.get("citation") or "").strip():
                raise SystemExit(
                    "Missing citation for existing document. "
                    "Provide --citation or set documents.metadata_json['citation'] first."
                )
            if merged_meta != existing_meta:
                doc.metadata_json = merged_meta

        # Upsert DocumentVersion
        dv = (
            sess.query(DocumentVersion)
            .filter(DocumentVersion.doc_id == doc_id, DocumentVersion.version == version_id)
            .one_or_none()
        )
        if dv is None:
            dv = DocumentVersion(
                doc_id=doc_id,
                version=version_id,
                gcs_chunks_path=None,
                n_chunks=0,
                gcs_items_path=items_uri,
                gcs_assets_path=assets_uri,
                n_items=0,
                n_assets=0,
                ingest_status="registered",
            )
            sess.add(dv)
        else:
            dv.gcs_items_path = items_uri
            dv.gcs_assets_path = assets_uri
            dv.ingest_status = "registered"

        sess.commit()

    print("OK registered")
    print(f"  doc_id        = {doc_id}")
    print(f"  doc_type      = {args.doc_type}")
    print(f"  version_id    = {version_id}")
    print(f"  raw_uri       = {args.raw_uri}")
    print(f"  parsed_prefix = {parsed_prefix_uri}")
    print(f"  items_uri     = {items_uri}")
    print(f"  assets_uri    = {assets_uri}")


if __name__ == "__main__":
    main()
