#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Iterable

from google.protobuf.json_format import MessageToDict
import numpy as np
from PIL import Image, ImageFilter

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


def _asset_local_render_path(doc_id: str, version_id: str, asset_id: str) -> str:
    return f"out/assets/{doc_id}/{version_id}/{asset_id}.png"


def _doc_specialty(doc: Document) -> str | None:
    metadata = doc.metadata_json if isinstance(doc.metadata_json, dict) else {}
    raw = str(metadata.get("specialty") or "").strip()
    return raw or None


def _normalize_bbox_points(points: list[dict[str, float]] | None) -> list[dict[str, float]] | None:
    if not points:
        return None
    out: list[dict[str, float]] = []
    for p in points:
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        out.append({"x": round(x, 6), "y": round(y, 6)})
    return out or None


def _asset_id_deterministic(
    *,
    doc_id: str,
    version_id: str,
    asset_type: str,
    locator: dict[str, int] | None,
    bbox: list[dict[str, float]] | None,
    caption: str,
    seed_hint: str = "",
) -> str:
    payload = {
        "doc_id": doc_id,
        "version_id": version_id,
        "asset_type": asset_type,
        "locator": locator or {},
        "bbox": bbox or [],
        "caption": (caption or "").strip(),
        "seed_hint": seed_hint,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


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


def _layout_bbox(block: dict) -> list[dict[str, float]] | None:
    poly = block.get("bounding_poly")
    if isinstance(poly, dict):
        pts = poly.get("normalized_vertices") or poly.get("vertices") or []
        if isinstance(pts, list) and pts:
            out: list[dict[str, float]] = []
            for p in pts:
                if not isinstance(p, dict):
                    continue
                out.append({"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))})
            return _normalize_bbox_points(out)
    table = block.get("table_block", {})
    for row in table.get("header_rows", []) + table.get("body_rows", []):
        for cell in row.get("cells", []):
            for child in cell.get("blocks", []):
                child_bbox = _layout_bbox(child)
                if child_bbox:
                    return child_bbox
    return None


def _iter_layout_blocks(blocks: list[dict]) -> Iterable[dict]:
    for block in blocks:
        yield block
        child_blocks = block.get("text_block", {}).get("blocks", [])
        if child_blocks:
            yield from _iter_layout_blocks(child_blocks)


_FIGURE_TYPE_TOKENS = {
    "figure",
    "image",
    "chart",
    "diagram",
    "photo",
    "picture",
    "illustration",
    "graphic",
}


def _is_figure_like_type(raw_type: Any) -> bool:
    normalized = str(raw_type or "").strip().lower()
    if not normalized:
        return False
    return any(tok in normalized for tok in _FIGURE_TYPE_TOKENS)


def _layout_bbox_from_layout_dict(layout_obj: dict, page_obj: dict) -> list[dict[str, float]] | None:
    if not isinstance(layout_obj, dict):
        return None
    poly = layout_obj.get("bounding_poly")
    if not isinstance(poly, dict):
        return None

    normalized_vertices = poly.get("normalized_vertices")
    if isinstance(normalized_vertices, list) and normalized_vertices:
        points: list[dict[str, float]] = []
        for p in normalized_vertices:
            if not isinstance(p, dict):
                continue
            points.append({"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))})
        return _normalize_bbox_points(points)

    vertices = poly.get("vertices")
    if not isinstance(vertices, list) or not vertices:
        return None
    page_dim = page_obj.get("dimension") if isinstance(page_obj.get("dimension"), dict) else {}
    width = float(page_dim.get("width", 0.0) or 0.0)
    height = float(page_dim.get("height", 0.0) or 0.0)
    if width <= 0.0 or height <= 0.0:
        return None
    points = []
    for p in vertices:
        if not isinstance(p, dict):
            continue
        x = float(p.get("x", 0.0)) / width
        y = float(p.get("y", 0.0)) / height
        points.append({"x": x, "y": y})
    return _normalize_bbox_points(points)


def _bbox_rect_from_poly(poly: list[dict[str, float]] | None) -> tuple[float, float, float, float] | None:
    if not poly:
        return None
    xs = [float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)]
    ys = [float(p.get("y", 0.0)) for p in poly if isinstance(p, dict)]
    if not xs or not ys:
        return None
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(1.0, max(xs))
    y1 = min(1.0, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _bbox_iou(a: list[dict[str, float]] | None, b: list[dict[str, float]] | None) -> float:
    ra = _bbox_rect_from_poly(a)
    rb = _bbox_rect_from_poly(b)
    if not ra or not rb:
        return 0.0
    ax0, ay0, ax1, ay1 = ra
    bx0, by0, bx1, by1 = rb
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    den = area_a + area_b - inter
    if den <= 0:
        return 0.0
    return inter / den


def _rect_to_poly_norm(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, float]]:
    return _normalize_bbox_points(
        [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ]
    ) or []


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    comps: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = 1
        min_x = max_x = x0
        min_y = max_y = y0
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if not mask[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = 1
                    stack.append((ny, nx))
        comps.append((min_x, min_y, max_x + 1, max_y + 1, count))
    return comps


def _detect_pdf_figure_bboxes(
    *,
    pdf_bytes: bytes,
    table_bboxes_by_page: dict[int, list[list[dict[str, float]]]],
) -> dict[int, list[list[dict[str, float]]]]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception:
        return {}

    out: dict[int, list[list[dict[str, float]]]] = {}
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = 96.0 / 72.0
    for pidx in range(len(pdf)):
        page_no = pidx + 1
        pil = pdf[pidx].render(scale=scale).to_pil().convert("L")
        if max(pil.size) > 1200:
            ratio = 1200.0 / float(max(pil.size))
            pil = pil.resize((max(32, int(pil.width * ratio)), max(32, int(pil.height * ratio))), Image.Resampling.BILINEAR)
        # Close small white holes and connect strokes to isolate macro visual regions.
        proc = pil.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
        arr = np.asarray(proc, dtype=np.uint8)
        mask = arr < 238
        comps = _connected_components(mask)
        page_area = float(proc.width * proc.height)
        table_bboxes = table_bboxes_by_page.get(page_no, [])
        cand: list[tuple[float, list[dict[str, float]]]] = []
        for x0, y0, x1, y1, pixels in comps:
            w = x1 - x0
            h = y1 - y0
            if w <= 0 or h <= 0:
                continue
            box_area = float(w * h)
            if box_area / page_area < 0.015:
                continue
            if (w / proc.width) < 0.12 or (h / proc.height) < 0.08:
                continue
            if (w / proc.width) > 0.98 or (h / proc.height) > 0.98:
                continue
            density = float(pixels) / box_area
            if density < 0.08:
                continue
            poly = _rect_to_poly_norm(
                x0 / proc.width,
                y0 / proc.height,
                x1 / proc.width,
                y1 / proc.height,
            )
            if not poly:
                continue
            if any(_bbox_iou(poly, tb) >= 0.15 for tb in table_bboxes):
                continue
            score = box_area * density
            cand.append((score, poly))
        cand = sorted(cand, key=lambda x: -x[0])[:4]
        if cand:
            out[page_no] = [poly for _, poly in cand]
    return out


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
    ap.add_argument(
        "--disable-figure-fallback",
        action="store_true",
        help="Disable PDF-based fallback figure detection when DocAI has no visual figure/image blocks.",
    )
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
        seen_figure_keys: set[tuple[int, str]] = set()
        table_bboxes_by_page: dict[int, list[list[dict[str, float]]]] = {}
        now = datetime.utcnow().isoformat() + "Z"
        full_text = document.text or ""
        specialty = _doc_specialty(doc)

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
                    bbox = _normalize_bbox_points(_bounding_poly_to_list(table.layout.bounding_poly))
                    locator = {"page_start": i, "page_end": i}
                    table_rows = _table_to_rows(table, full_text)
                    caption = table_rows[0][0].strip() if table_rows and table_rows[0] else ""
                    aid = _asset_id_deterministic(
                        doc_id=args.doc_id,
                        version_id=args.version_id,
                        asset_type="table",
                        locator=locator,
                        bbox=bbox,
                        caption=caption,
                        seed_hint=f"page_table:{i}",
                    )
                    table_uri = gcs_uri(bucket, layout.asset_table(aid))
                    _gsutil_write_text(
                        table_uri,
                        json.dumps({
                            "rows": table_rows,
                            "page": i,
                            "bbox": bbox,
                        }, ensure_ascii=False),
                    )
                    assets.append(json.dumps({
                        "asset_id": aid,
                        "doc_id": args.doc_id,
                        "version_id": args.version_id,
                        "asset_type": "table",
                        "specialty": specialty,
                        "page": i,
                        "bbox": bbox,
                        "caption": caption or None,
                        "files": {"table_uri": table_uri},
                        "locator": locator,
                        "render_path": _asset_local_render_path(args.doc_id, args.version_id, aid),
                        "meta": {"ingested_at": now, "source": "docai_page_table"},
                    }, ensure_ascii=False))
                    if bbox:
                        table_bboxes_by_page.setdefault(i, []).append(bbox)

                page_raw = raw_dict.get("pages", [])[i - 1] if isinstance(raw_dict.get("pages"), list) and len(raw_dict.get("pages", [])) >= i else {}
                visual_elements = page_raw.get("visual_elements", []) if isinstance(page_raw, dict) else []
                if isinstance(visual_elements, list):
                    for ve in visual_elements:
                        if not isinstance(ve, dict):
                            continue
                        ve_type = str(ve.get("type") or "").strip().lower()
                        if not _is_figure_like_type(ve_type):
                            continue
                        bbox = _layout_bbox_from_layout_dict(ve.get("layout") if isinstance(ve.get("layout"), dict) else {}, page_raw if isinstance(page_raw, dict) else {})
                        if not bbox:
                            continue
                        locator = {"page_start": i, "page_end": i}
                        key = (i, json.dumps(bbox, ensure_ascii=False, sort_keys=True))
                        if key in seen_figure_keys:
                            continue
                        seen_figure_keys.add(key)
                        aid = _asset_id_deterministic(
                            doc_id=args.doc_id,
                            version_id=args.version_id,
                            asset_type="figure",
                            locator=locator,
                            bbox=bbox,
                            caption="",
                        )
                        assets.append(json.dumps({
                            "asset_id": aid,
                            "doc_id": args.doc_id,
                            "version_id": args.version_id,
                            "asset_type": "figure",
                            "specialty": specialty,
                            "page": i,
                            "bbox": bbox,
                            "caption": None,
                            "files": {},
                            "locator": locator,
                            "render_path": _asset_local_render_path(args.doc_id, args.version_id, aid),
                            "meta": {"ingested_at": now, "source": "docai_page_visual_element", "raw_type": ve_type},
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
                    locator = _layout_page_span(block) or {}
                    bbox = _layout_bbox(block)
                    table_rows = _layout_table_to_rows(table_block)
                    caption = table_rows[0][0].strip() if table_rows and table_rows[0] else ""
                    aid = _asset_id_deterministic(
                        doc_id=args.doc_id,
                        version_id=args.version_id,
                        asset_type="table",
                        locator=locator,
                        bbox=bbox,
                        caption=caption,
                        seed_hint=f"layout_table:{block.get('block_id','')}",
                    )
                    table_uri = gcs_uri(bucket, layout.asset_table(aid))
                    _gsutil_write_text(
                        table_uri,
                        json.dumps({"rows": table_rows}, ensure_ascii=False),
                    )
                    assets.append(json.dumps({
                        "asset_id": aid,
                        "doc_id": args.doc_id,
                        "version_id": args.version_id,
                        "asset_type": "table",
                        "specialty": specialty,
                        "page": locator.get("page_start"),
                        "bbox": bbox,
                        "caption": caption or None,
                        "files": {"table_uri": table_uri},
                        "locator": locator,
                        "render_path": _asset_local_render_path(args.doc_id, args.version_id, aid),
                        "meta": {"ingested_at": now, "source": "docai_layout_table"},
                    }, ensure_ascii=False))
                    try:
                        page_no = int(locator.get("page_start", 0))
                    except Exception:
                        page_no = 0
                    if page_no > 0 and bbox:
                        table_bboxes_by_page.setdefault(page_no, []).append(bbox)

                block_type = str(text_block.get("type_", "")).strip().lower()
                if _is_figure_like_type(block_type):
                    locator = _layout_page_span(block) or {}
                    bbox = _layout_bbox(block)
                    try:
                        page_no = int(locator.get("page_start", 0))
                    except Exception:
                        page_no = 0
                    if page_no > 0 and bbox:
                        key = (page_no, json.dumps(bbox, ensure_ascii=False, sort_keys=True))
                        if key in seen_figure_keys:
                            continue
                        seen_figure_keys.add(key)
                    caption = (text or "").strip()
                    aid = _asset_id_deterministic(
                        doc_id=args.doc_id,
                        version_id=args.version_id,
                        asset_type="figure",
                        locator=locator,
                        bbox=bbox,
                        caption=caption,
                        seed_hint=f"layout_figure:{block.get('block_id','')}",
                    )
                    assets.append(json.dumps({
                        "asset_id": aid,
                        "doc_id": args.doc_id,
                        "version_id": args.version_id,
                        "asset_type": "figure",
                        "specialty": specialty,
                        "page": locator.get("page_start"),
                        "bbox": bbox,
                        "caption": caption or None,
                        "files": {},
                        "locator": locator,
                        "render_path": _asset_local_render_path(args.doc_id, args.version_id, aid),
                        "meta": {"ingested_at": now, "source": "docai_layout_figure"},
                    }, ensure_ascii=False))

        if not args.disable_figure_fallback:
            figure_count = 0
            for raw in assets:
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if str(obj.get("asset_type") or "").strip().lower() in {"figure", "image", "chart"}:
                    figure_count += 1
            if figure_count == 0:
                fallback = _detect_pdf_figure_bboxes(
                    pdf_bytes=pdf_bytes,
                    table_bboxes_by_page=table_bboxes_by_page,
                )
                for page_no in sorted(fallback.keys()):
                    for idx, bbox in enumerate(fallback[page_no], start=1):
                        key = (page_no, json.dumps(bbox, ensure_ascii=False, sort_keys=True))
                        if key in seen_figure_keys:
                            continue
                        seen_figure_keys.add(key)
                        locator = {"page_start": page_no, "page_end": page_no}
                        aid = _asset_id_deterministic(
                            doc_id=args.doc_id,
                            version_id=args.version_id,
                            asset_type="figure",
                            locator=locator,
                            bbox=bbox,
                            caption="",
                            seed_hint=f"pdf_fallback_figure:{page_no}:{idx}",
                        )
                        assets.append(
                            json.dumps(
                                {
                                    "asset_id": aid,
                                    "doc_id": args.doc_id,
                                    "version_id": args.version_id,
                                    "asset_type": "figure",
                                    "specialty": specialty,
                                    "page": page_no,
                                    "bbox": bbox,
                                    "caption": None,
                                    "files": {},
                                    "locator": locator,
                                    "render_path": _asset_local_render_path(args.doc_id, args.version_id, aid),
                                    "meta": {"ingested_at": now, "source": "pdf_fallback_connected_components"},
                                },
                                ensure_ascii=False,
                            )
                        )

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
