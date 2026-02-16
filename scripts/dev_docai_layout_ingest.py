#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Iterable

from google.protobuf.json_format import MessageToDict

from odonto_rag.catalog.db import make_engine, make_session_factory
from odonto_rag.catalog.models import Document, DocumentVersion
from odonto_rag.storage_layout import ParsedLayout, gcs_uri


def _ensure_dependencies() -> None:
    try:
        subprocess.run(["gsutil", "version"], check=True, capture_output=True)
        from google.cloud import documentai  # noqa
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies. Ensure gsutil and google-cloud-documentai are installed."
        ) from exc


def _get_access_token() -> str:
    try:
        out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    except FileNotFoundError as exc:
        raise SystemExit("gcloud not found. Install Google Cloud SDK and run 'gcloud auth login'.") from exc
    if not out:
        raise SystemExit("Empty access token. Run 'gcloud auth login'.")
    return out


def _parse_json_or_text(raw: bytes) -> dict[str, Any] | str:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return text


def _error_message_from_body(body: dict[str, Any] | str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or "").strip()
            if msg:
                return msg
        msg = str(body.get("message") or "").strip()
        if msg:
            return msg
        return json.dumps(body, ensure_ascii=False)
    return str(body or "").strip()


def _truncate_msg(msg: str, max_len: int = 300) -> str:
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 1] + "…"


def _non_200_body_excerpt(body: dict[str, Any] | str) -> str:
    # Prefer JSON error.message if present; otherwise return body text/JSON dump.
    return _truncate_msg(_error_message_from_body(body) or "<empty response body>")


def _body_excerpt(body: dict[str, Any] | str) -> str:
    if isinstance(body, dict):
        return _truncate_msg(json.dumps(body, ensure_ascii=False))
    return _truncate_msg(str(body or ""))


def _docai_base_url(location: str) -> str:
    loc = (location or "").strip()
    if loc == "eu":
        return "https://eu-documentai.googleapis.com"
    if loc == "us":
        return "https://us-documentai.googleapis.com"
    return f"https://{loc}-documentai.googleapis.com"


def http_get_json(url: str) -> tuple[int, dict[str, Any] | str]:
    token = _get_access_token()
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _parse_json_or_text(resp.read())
            msg = _body_excerpt(body)
            print(f"[docai] GET {url} -> status={int(resp.status)} body={msg}")
            return int(resp.status), body
    except urllib.error.HTTPError as exc:
        body = _parse_json_or_text(exc.read() or b"")
        msg = _non_200_body_excerpt(body)
        print(f"[docai] GET {url} -> status={int(exc.code)} body={msg}")
        return int(exc.code), body
    except urllib.error.URLError as exc:
        msg = _truncate_msg(str(exc.reason))
        print(f"[docai] GET {url} -> status=0 message={msg}")
        return 0, msg


def _raise_permission_error(project: str, location: str, body: dict[str, Any] | str) -> None:
    msg = _truncate_msg(_error_message_from_body(body))
    raise SystemExit(
        "Permission/API error while accessing Document AI in "
        f"project={project}, location={location} (403). Details: {msg}"
    )


def resolve_processor_location(project: str, processor_id: str, preferred_location: str) -> str:
    preferred = (preferred_location or "").strip()
    check_locations: list[str] = []
    if preferred:
        check_locations.append(preferred)
    for loc in ["us", "eu"]:
        if loc not in check_locations:
            check_locations.append(loc)

    for location in check_locations:
        url = (
            f"{_docai_base_url(location)}/v1/projects/"
            f"{project}/locations/{location}/processors/{processor_id}"
        )
        status, body = http_get_json(url)
        if status == 200:
            return location
        if status == 403:
            _raise_permission_error(project, location, body)
        if status in (400, 404):
            continue

    for location in ["us", "eu"]:
        list_url = (
            f"{_docai_base_url(location)}/v1/projects/"
            f"{project}/locations/{location}/processors"
        )
        status, body = http_get_json(list_url)
        if status == 403:
            _raise_permission_error(project, location, body)
        if status == 400:
            continue
        if status != 200 or not isinstance(body, dict):
            continue
        processors = body.get("processors", [])
        if not isinstance(processors, list):
            continue
        needle = f"/processors/{processor_id}"
        for p in processors:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "")
            if needle in name:
                return location

    raise SystemExit(
        f"Processor ID {processor_id} not found in project {project}. "
        "Checked locations: preferred, us, eu. "
        "Use Console to confirm processor ID/project."
    )


def _docai_api_endpoint(location: str) -> str:
    loc = (location or "").strip()
    if loc == "eu":
        return "eu-documentai.googleapis.com"
    if loc == "us":
        return "us-documentai.googleapis.com"
    return f"{loc}-documentai.googleapis.com"


def _gsutil_cat(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    proc = subprocess.run(
        ["gsutil", "cat", uri],
        check=True,
        capture_output=True,
    )
    return proc.stdout


def _gsutil_write_text(uri: str, text: str) -> None:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    proc = subprocess.Popen(
        ["gsutil", "cp", "-", uri],
        stdin=subprocess.PIPE,
    )
    proc.communicate(text.encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError(f"gsutil cp failed for {uri}")


def _gsutil_write_bytes(uri: str, payload: bytes) -> None:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    proc = subprocess.Popen(
        ["gsutil", "cp", "-", uri],
        stdin=subprocess.PIPE,
    )
    proc.communicate(payload)
    if proc.returncode != 0:
        raise RuntimeError(f"gsutil cp failed for {uri}")


def _text_from_anchor(anchor, full_text: str) -> str:
    if anchor is None:
        return ""
    parts: list[str] = []
    for segment in anchor.text_segments:
        start = int(segment.start_index or 0)
        if segment.end_index is None:
            end = len(full_text)
        else:
            end = int(segment.end_index or 0)
        if end <= start:
            continue
        parts.append(full_text[start:end])
    return "".join(parts).strip()


def _bounding_poly_to_list(poly) -> list[dict[str, float]] | None:
    if poly is None:
        return None
    vertices: Iterable = poly.normalized_vertices or poly.vertices
    if not vertices:
        return None
    return [{"x": float(v.x), "y": float(v.y)} for v in vertices]


def _page_text(page, full_text: str) -> str:
    text = _text_from_anchor(page.layout.text_anchor, full_text)
    if text:
        return text
    if page.paragraphs:
        parts = [_text_from_anchor(p.layout.text_anchor, full_text) for p in page.paragraphs]
        return "\n".join(p for p in parts if p)
    return ""


def _table_to_rows(table, full_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.header_rows + table.body_rows:
        row_cells = []
        for cell in row.cells:
            row_cells.append(_text_from_anchor(cell.layout.text_anchor, full_text))
        rows.append(row_cells)
    return rows


def _layout_table_to_rows(table_block: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table_block.get("header_rows", []) + table_block.get("body_rows", []):
        row_cells = []
        for cell in row.get("cells", []):
            parts = []
            for block in cell.get("blocks", []):
                text = block.get("text_block", {}).get("text", "")
                if text:
                    parts.append(text)
            row_cells.append(" ".join(parts).strip())
        rows.append(row_cells)
    return rows


def _layout_page_span(block: dict) -> dict[str, int] | None:
    span = block.get("page_span")
    if span and "page_start" in span and "page_end" in span:
        return {"page_start": span["page_start"], "page_end": span["page_end"]}
    table = block.get("table_block", {})
    for row in table.get("header_rows", []) + table.get("body_rows", []):
        for cell in row.get("cells", []):
            for child in cell.get("blocks", []):
                child_span = child.get("page_span")
                if child_span and "page_start" in child_span and "page_end" in child_span:
                    return {"page_start": child_span["page_start"], "page_end": child_span["page_end"]}
    return None


def _iter_layout_blocks(blocks: list[dict]) -> Iterable[dict]:
    for block in blocks:
        yield block
        child_blocks = block.get("text_block", {}).get("blocks", [])
        if child_blocks:
            yield from _iter_layout_blocks(child_blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Document AI Layout Parser and write items/assets to GCS.")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--version-id", required=True)
    ap.add_argument("--processor-id", required=True)
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_PATH", "catalog.sqlite3"))
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT", ""))
    ap.add_argument("--location", default=os.environ.get("DOCAI_LOCATION", ""))
    ap.add_argument("--raw-uri", default="")
    ap.add_argument("--mime-type", default="application/pdf")
    args = ap.parse_args()

    if not args.project or not args.location:
        raise SystemExit("Missing GCP_PROJECT or DOCAI_LOCATION")

    _ensure_dependencies()

    engine = make_engine(args.sqlite)
    SessionLocal = make_session_factory(engine)

    with SessionLocal() as sess:
        doc = sess.query(Document).filter(Document.doc_id == args.doc_id).one()
        dv = (
            sess.query(DocumentVersion)
            .filter(
                DocumentVersion.doc_id == args.doc_id,
                DocumentVersion.version == args.version_id,
            )
            .one()
        )

        items_uri = dv.gcs_items_path
        assets_uri = dv.gcs_assets_path
        raw_uri = args.raw_uri or doc.gcs_raw_path

        layout = ParsedLayout(args.doc_id, args.version_id)
        bucket = items_uri.split("/")[2]
        raw_output_uri = gcs_uri(bucket, layout.raw_extractor_output())

        pdf_bytes = _gsutil_cat(raw_uri)

        from google.cloud import documentai

        detected_location = resolve_processor_location(args.project, args.processor_id, args.location)
        if args.location != detected_location:
            print(
                f"[docai] WARNING: processor not found in requested --location={args.location}; "
                f"overriding with detected location={detected_location}"
            )
        else:
            print(f"[docai] using requested location={args.location}")

        api_endpoint = _docai_api_endpoint(detected_location)
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )
        processor = client.processor_path(args.project, detected_location, args.processor_id)
        print(f"[docai] api_endpoint={api_endpoint} location={detected_location} processor={processor}")

        raw_doc = documentai.RawDocument(content=pdf_bytes, mime_type=args.mime_type)
        result = client.process_document(
            request=documentai.ProcessRequest(
                name=processor,
                raw_document=raw_doc,
            )
        )

        document = result.document
        print("pdf_bytes_len:", len(pdf_bytes))
        print("pages_count:", len(document.pages))
        print("text_len:", len(document.text or ""))
        raw_dict = MessageToDict(document._pb, preserving_proto_field_name=True)
        _gsutil_write_text(raw_output_uri, json.dumps(raw_dict, ensure_ascii=False, indent=2))

        items, assets = [], []
        now = datetime.utcnow().isoformat() + "Z"
        full_text = document.text or ""

        if document.pages:
            for i, page in enumerate(document.pages, start=1):
                text = _page_text(page, full_text)
                if text:
                    items.append(json.dumps({
                        "item_id": str(uuid.uuid4()),
                        "doc_id": args.doc_id,
                        "version_id": args.version_id,
                        "item_type": "page",
                        "text": text,
                        "locator": {"page_start": i, "page_end": i},
                        "meta": {"ingested_at": now},
                    }, ensure_ascii=False))

                for table in page.tables:
                    aid = str(uuid.uuid4())
                    table_uri = gcs_uri(bucket, layout.asset_table(aid))
                    _gsutil_write_text(
                        table_uri,
                        json.dumps({
                            "rows": _table_to_rows(table, full_text),
                            "page": i,
                            "bbox": _bounding_poly_to_list(table.layout.bounding_poly),
                        }, ensure_ascii=False),
                    )
                    assets.append(json.dumps({
                        "asset_id": aid,
                        "asset_type": "table",
                        "files": {"table_uri": table_uri},
                        "locator": {"page_start": i, "page_end": i},
                        "meta": {"ingested_at": now},
                    }, ensure_ascii=False))
        else:
            layout_blocks = raw_dict.get("document_layout", {}).get("blocks", [])
            for block in _iter_layout_blocks(layout_blocks):
                text_block = block.get("text_block", {})
                text = text_block.get("text", "")
                if text:
                    locator = _layout_page_span(block) or {}
                    items.append(json.dumps({
                        "item_id": str(uuid.uuid4()),
                        "doc_id": args.doc_id,
                        "version_id": args.version_id,
                        "item_type": text_block.get("type_", "block"),
                        "text": text,
                        "locator": locator,
                        "meta": {"ingested_at": now},
                    }, ensure_ascii=False))

                table_block = block.get("table_block")
                if table_block:
                    aid = str(uuid.uuid4())
                    table_uri = gcs_uri(bucket, layout.asset_table(aid))
                    table_rows = _layout_table_to_rows(table_block)
                    _gsutil_write_text(
                        table_uri,
                        json.dumps({"rows": table_rows}, ensure_ascii=False),
                    )
                    locator = _layout_page_span(block) or {}
                    assets.append(json.dumps({
                        "asset_id": aid,
                        "asset_type": "table",
                        "files": {"table_uri": table_uri},
                        "locator": locator,
                        "meta": {"ingested_at": now},
                    }, ensure_ascii=False))

        _gsutil_write_text(items_uri, "\n".join(items) + "\n")
        _gsutil_write_text(assets_uri, "\n".join(assets) + "\n")

        dv.n_items = len(items)
        dv.n_assets = len(assets)
        dv.ingest_status = "ingested"
        sess.commit()

        print("OK ingested via Document AI (gsutil)")
        print("items:", len(items), "assets:", len(assets))


if __name__ == "__main__":
    main()
