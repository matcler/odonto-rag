#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime

from odonto_rag.catalog.db import make_engine, make_session_factory
from odonto_rag.catalog.models import Document, DocumentVersion

def _ensure_gcsfs():
    try:
        import gcsfs  # noqa
        return True
    except Exception:
        return False

def _write_gcs_text(uri: str, text: str) -> None:
    import gcsfs
    fs = gcsfs.GCSFileSystem(project=os.environ.get("GCP_PROJECT", None))
    # gcsfs expects "bucket/path" not "gs://bucket/path"
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    path = uri[len("gs://"):]
    with fs.open(path, "w") as f:
        f.write(text)

def main() -> None:
    ap = argparse.ArgumentParser(description="Write sample items.jsonl/assets.jsonl to canonical GCS paths and update DB counts.")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--version-id", required=True)
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_PATH", "catalog.sqlite3"))
    ap.add_argument("--with-asset", action="store_true", help="Also write one sample asset record")
    args = ap.parse_args()

    if not _ensure_gcsfs():
        raise SystemExit("Missing dependency: gcsfs. Install it with: python -m pip install gcsfs")

    engine = make_engine(args.sqlite)
    SessionLocal = make_session_factory(engine)

    with SessionLocal() as sess:
        doc = sess.query(Document).filter(Document.doc_id == args.doc_id).one_or_none()
        if doc is None:
            raise SystemExit(f"doc_id not found in documents: {args.doc_id}")

        dv = (
            sess.query(DocumentVersion)
            .filter(DocumentVersion.doc_id == args.doc_id, DocumentVersion.version == args.version_id)
            .one_or_none()
        )
        if dv is None:
            raise SystemExit(f"doc_id/version not found in document_versions: {args.doc_id}/{args.version_id}")

        items_uri = dv.gcs_items_path
        assets_uri = dv.gcs_assets_path
        if not items_uri or not assets_uri:
            raise SystemExit("Missing gcs_items_path or gcs_assets_path in document_versions row.")

        now = datetime.utcnow().isoformat() + "Z"
        item_id = str(uuid.uuid4())
        item = {
            "item_id": item_id,
            "doc_id": args.doc_id,
            "version_id": args.version_id,
            "doc_type": doc.doc_type,
            "item_type": "chunk",
            "text": f"[SAMPLE] created_at={now} doc={args.doc_id} version={args.version_id}",
            "title": None,
            "section": None,
            "locator": {"page_start": 1, "page_end": 1, "slide_index": None, "t_start": None, "t_end": None, "bbox": None},
            "source": {"uri": doc.gcs_raw_path, "name": None, "publisher": None, "year": None},
            "tags": ["sample"],
            "meta": {"created_by": "dev_write_sample_extract"}
        }

        items_text = json.dumps(item, ensure_ascii=False) + "\n"

        assets_text = ""
        n_assets = 0
        if args.with_asset:
            asset_id = str(uuid.uuid4())
            asset = {
                "asset_id": asset_id,
                "doc_id": args.doc_id,
                "version_id": args.version_id,
                "doc_type": doc.doc_type,
                "asset_type": "figure",
                "caption": f"[SAMPLE FIGURE] created_at={now}",
                "locator": {"page_start": 1, "page_end": 1, "slide_index": None, "t_start": None, "t_end": None, "bbox": None},
                "files": {"image_uri": None, "table_uri": None},
                "table": None,
                "tags": ["sample"],
                "meta": {"created_by": "dev_write_sample_extract"}
            }
            assets_text = json.dumps(asset, ensure_ascii=False) + "\n"
            n_assets = 1
        else:
            assets_text = ""

        # Write to GCS
        _write_gcs_text(items_uri, items_text)
        _write_gcs_text(assets_uri, assets_text)

        # Update DB counts + status
        dv.n_items = 1
        dv.n_assets = n_assets
        dv.ingest_status = "written"
        sess.commit()

        print("OK wrote sample extract")
        print(f"  items_uri  = {items_uri}")
        print(f"  assets_uri = {assets_uri}")
        print(f"  n_items    = {dv.n_items}")
        print(f"  n_assets   = {dv.n_assets}")

if __name__ == "__main__":
    main()
