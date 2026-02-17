#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List

from odonto_rag.catalog.db import make_engine, make_session_factory
from odonto_rag.catalog.models import Document, DocumentVersion
from PIL import Image, ImageDraw, ImageFont


def _gs_download_bytes(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    try:
        from google.cloud import storage

        _, _, rest = uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid gs:// URI: {uri}")
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        return blob.download_as_bytes()
    except Exception:
        return subprocess.check_output(["gsutil", "cat", uri])


def _load_assets_jsonl(uri: str) -> List[Dict[str, Any]]:
    raw = _gs_download_bytes(uri).decode("utf-8", errors="replace").splitlines()
    out: List[Dict[str, Any]] = []
    for line in raw:
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _bbox_to_px(
    bbox: List[Dict[str, Any]], width: int, height: int, pad_ratio: float
) -> tuple[int, int, int, int] | None:
    if not bbox:
        return None
    xs = [float(p.get("x", 0.0)) for p in bbox if isinstance(p, dict)]
    ys = [float(p.get("y", 0.0)) for p in bbox if isinstance(p, dict)]
    if not xs or not ys:
        return None
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(1.0, max(xs))
    y1 = min(1.0, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    pad_x = pad_ratio * (x1 - x0)
    pad_y = pad_ratio * (y1 - y0)
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(1.0, x1 + pad_x)
    y1 = min(1.0, y1 + pad_y)
    left = int(round(x0 * width))
    top = int(round(y0 * height))
    right = int(round(x1 * width))
    bottom = int(round(y1 * height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _asset_output_path(out_dir: Path, doc_id: str, version_id: str, asset_id: str) -> Path:
    return out_dir / doc_id / version_id / f"{asset_id}.png"


def _asset_table_local_path(out_dir: Path, doc_id: str, version_id: str, asset_id: str) -> Path:
    return out_dir / doc_id / version_id / f"{asset_id}.table.json"


def _render_table_rows_png(rows: List[List[str]], out_path: Path) -> tuple[int, int]:
    if not rows:
        rows = [["(empty table)"]]
    n_cols = max(1, max(len(r) for r in rows))
    normalized_rows: List[List[str]] = []
    for row in rows[:40]:
        vals = [str(c or "").strip() for c in row]
        vals += [""] * (n_cols - len(vals))
        normalized_rows.append(vals)

    font = ImageFont.load_default()
    pad = 8
    row_h = 28
    max_cell_w = 280
    col_widths = [120] * n_cols
    for c in range(n_cols):
        w = 120
        for r in normalized_rows:
            text = (r[c] or "")[:80]
            tw = int(font.getlength(text)) + (2 * pad)
            if tw > w:
                w = tw
        col_widths[c] = min(max_cell_w, max(120, w))

    img_w = sum(col_widths) + 2
    img_h = (len(normalized_rows) * row_h) + 2
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    y = 1
    for r_idx, row in enumerate(normalized_rows):
        x = 1
        for c_idx, cell in enumerate(row):
            w = col_widths[c_idx]
            bg = (240, 246, 255) if r_idx == 0 else (255, 255, 255)
            draw.rectangle([x, y, x + w, y + row_h], fill=bg, outline=(180, 180, 180))
            text = (cell or "")[:120]
            draw.text((x + pad, y + 7), text, fill=(20, 20, 20), font=font)
            x += w
        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return img_w, img_h


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render deterministic PNG crops for layout assets (table/figure) from a source PDF."
    )
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--version-id", required=True)
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_PATH", "catalog.sqlite3"))
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--pad-ratio", type=float, default=0.02)
    ap.add_argument("--out-dir", default="out/assets")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if target PNG already exists.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an asset cannot be rendered (missing bbox/page/renderer).",
    )
    args = ap.parse_args()

    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing dependency pypdfium2. Install with: python3 -m pip install pypdfium2"
        ) from exc

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

    if not doc.gcs_raw_path:
        raise SystemExit("documents.gcs_raw_path is empty")
    if not dv.gcs_assets_path:
        raise SystemExit("document_versions.gcs_assets_path is empty")

    pdf_bytes = _gs_download_bytes(doc.gcs_raw_path)
    assets = _load_assets_jsonl(dv.gcs_assets_path)

    pdf = pdfium.PdfDocument(pdf_bytes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    cached = 0
    skipped = 0
    enriched_lines: List[str] = []

    scale = max(1.0, float(args.dpi) / 72.0)
    for asset in assets:
        asset_id = str(asset.get("asset_id") or "").strip()
        locator = asset.get("locator") if isinstance(asset.get("locator"), dict) else {}
        page_raw = locator.get("page_start", asset.get("page"))
        bbox = asset.get("bbox") if isinstance(asset.get("bbox"), list) else []
        out_path = _asset_output_path(out_dir, args.doc_id, args.version_id, asset_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not asset_id or not page_raw:
            skipped += 1
            if args.strict:
                raise SystemExit(f"Asset missing asset_id/page: {asset}")
            asset["render_path"] = str(out_path)
            asset["meta"] = {
                **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
                "render_status": "skipped_missing_page_or_id",
            }
            enriched_lines.append(json.dumps(asset, ensure_ascii=False))
            continue

        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            cached += 1
            asset["render_path"] = str(out_path)
            asset["meta"] = {
                **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
                "render_status": "cached",
                "render_dpi": args.dpi,
            }
            enriched_lines.append(json.dumps(asset, ensure_ascii=False))
            continue

        page_no = int(page_raw)
        if page_no <= 0 or page_no > len(pdf):
            skipped += 1
            if args.strict:
                raise SystemExit(f"Asset page out of range: asset_id={asset_id} page={page_no}")
            asset["render_path"] = str(out_path)
            asset["meta"] = {
                **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
                "render_status": "skipped_page_out_of_range",
            }
            enriched_lines.append(json.dumps(asset, ensure_ascii=False))
            continue

        if not bbox:
            if (
                str(asset.get("asset_type") or "").strip().lower() == "table"
                and isinstance(asset.get("files"), dict)
                and str(asset.get("files", {}).get("table_uri") or "").startswith("gs://")
            ):
                table_uri = str(asset.get("files", {}).get("table_uri") or "")
                try:
                    table_obj = json.loads(_gs_download_bytes(table_uri).decode("utf-8", errors="replace"))
                except Exception:
                    table_obj = {}
                rows = table_obj.get("rows") if isinstance(table_obj, dict) and isinstance(table_obj.get("rows"), list) else []
                table_local_path = _asset_table_local_path(out_dir, args.doc_id, args.version_id, asset_id)
                table_local_path.write_text(
                    json.dumps({"rows": rows}, ensure_ascii=False),
                    encoding="utf-8",
                )
                files_obj = asset.get("files") if isinstance(asset.get("files"), dict) else {}
                files_obj = dict(files_obj)
                files_obj["table_local_path"] = str(table_local_path)
                asset["files"] = files_obj
                w, h = _render_table_rows_png(rows, out_path)
                rendered += 1
                asset["render_path"] = str(out_path)
                asset["meta"] = {
                    **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
                    "render_status": "rendered_from_table_rows",
                    "render_dpi": args.dpi,
                    "render_size": {"width": w, "height": h},
                }
                enriched_lines.append(json.dumps(asset, ensure_ascii=False))
                continue

            skipped += 1
            if args.strict:
                raise SystemExit(f"Asset missing bbox: asset_id={asset_id}")
            asset["render_path"] = str(out_path)
            asset["meta"] = {
                **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
                "render_status": "skipped_missing_bbox",
            }
            enriched_lines.append(json.dumps(asset, ensure_ascii=False))
            continue

        page = pdf[page_no - 1]
        pil = page.render(scale=scale).to_pil()
        crop_px = _bbox_to_px(bbox, pil.width, pil.height, args.pad_ratio)
        if not crop_px:
            skipped += 1
            if args.strict:
                raise SystemExit(f"Asset invalid bbox: asset_id={asset_id}")
            asset["render_path"] = str(out_path)
            asset["meta"] = {
                **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
                "render_status": "skipped_invalid_bbox",
            }
            enriched_lines.append(json.dumps(asset, ensure_ascii=False))
            continue

        cropped = pil.crop(crop_px)
        cropped.save(out_path, format="PNG")
        rendered += 1

        asset["render_path"] = str(out_path)
        asset["meta"] = {
            **(asset.get("meta") if isinstance(asset.get("meta"), dict) else {}),
            "render_status": "rendered",
            "render_dpi": args.dpi,
            "crop_px": {
                "left": crop_px[0],
                "top": crop_px[1],
                "right": crop_px[2],
                "bottom": crop_px[3],
            },
        }
        enriched_lines.append(json.dumps(asset, ensure_ascii=False))

    enriched_path = out_dir / args.doc_id / args.version_id / "assets.enriched.jsonl"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_path.write_text("\n".join(enriched_lines) + "\n", encoding="utf-8")

    print(f"doc_id={args.doc_id} version_id={args.version_id}")
    print(f"assets_total={len(assets)}")
    print(f"rendered={rendered}")
    print(f"cached={cached}")
    print(f"skipped={skipped}")
    print(f"enriched_jsonl={enriched_path}")


if __name__ == "__main__":
    main()
