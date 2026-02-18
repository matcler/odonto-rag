#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import zipfile

_NUMERIC_TOKENS = ("%", "mean", "median", "ratio", "rate", "mm", "cm", "n=", "p<", "vs")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"FAIL: file not found: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"FAIL: JSON root must be object: {path}")
    return obj


def _is_numeric_risk_text(text: str) -> bool:
    t = text.lower()
    if any(tok in t for tok in _NUMERIC_TOKENS):
        return True
    return any(ch.isdigit() for ch in t)


def _norm_locator(raw: Any) -> Dict[str, int]:
    loc = raw if isinstance(raw, dict) else {}
    return {
        "page_start": int(loc.get("page_start", 0) or 0),
        "page_end": int(loc.get("page_end", 0) or 0),
    }


def _stable_sort_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            str(x.get("doc_id") or ""),
            str(x.get("item_id") or ""),
            int((x.get("locator") or {}).get("page_start", 0) or 0),
            int((x.get("locator") or {}).get("page_end", 0) or 0),
            str(x.get("score")),
        ),
    )


def _stable_sort_visuals(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            str(x.get("asset_id") or ""),
            str(x.get("doc_id") or ""),
            str(x.get("type") or ""),
        ),
    )


def _gs_download_bytes(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    try:
        from google.cloud import storage  # type: ignore

        _, _, rest = uri.partition("gs://")
        bucket, _, blob = rest.partition("/")
        if not bucket or not blob:
            raise ValueError(f"Invalid gs:// URI: {uri}")
        return storage.Client().bucket(bucket).blob(blob).download_as_bytes()
    except Exception:
        return subprocess.check_output(["gsutil", "cat", uri])


def _doc_raw_uri(sqlite_path: Path, doc_id: str) -> Optional[str]:
    if not sqlite_path.exists():
        return None
    with sqlite3.connect(str(sqlite_path)) as con:
        row = con.execute("SELECT gcs_raw_path FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if not row:
        return None
    raw = str(row[0] or "").strip()
    return raw or None


def _ensure_page_preview(
    *,
    doc_id: str,
    version: str,
    page: int,
    cache_root: Path,
    sqlite_path: Path,
    dpi: int,
    pdf_cache: Dict[str, bytes],
) -> Optional[Path]:
    if page <= 0:
        return None
    out_path = cache_root / doc_id / version / f"p{page:04d}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    raw_uri = _doc_raw_uri(sqlite_path, doc_id)
    if not raw_uri:
        return None

    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception:
        return None

    cache_key = f"{doc_id}|{raw_uri}"
    if cache_key not in pdf_cache:
        try:
            pdf_cache[cache_key] = _gs_download_bytes(raw_uri)
        except Exception:
            return None

    try:
        pdf = pdfium.PdfDocument(pdf_cache[cache_key])
    except Exception:
        return None

    if page > len(pdf):
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = max(1.0, float(dpi) / 72.0)
    try:
        pil = pdf[page - 1].render(scale=scale).to_pil()
        pil.save(out_path, format="PNG")
        return out_path
    except Exception:
        return None


def _extract_bundle(bundle_zip: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    return out_dir


def _to_rel(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(str(path), str(base_dir))
    except Exception:
        return str(path)


def _normalize_for_viewer(
    *,
    audit: Dict[str, Any],
    output_html: Path,
    sqlite_path: Path,
    page_preview_root: Path,
    page_preview_dpi: int,
    bundle_extract_dir: Optional[Path],
) -> Dict[str, Any]:
    slides_raw = audit.get("slides") if isinstance(audit.get("slides"), list) else []
    request = audit.get("request") if isinstance(audit.get("request"), dict) else {}

    deck_id = str(audit.get("deck_id") or output_html.stem.replace(".audit", "")).strip() or "deck"
    version = str(request.get("version") or "unknown_version").strip() or "unknown_version"

    pdf_cache: Dict[str, bytes] = {}
    slides_out: List[Dict[str, Any]] = []

    for idx, raw_slide in enumerate(sorted(slides_raw, key=lambda s: int((s or {}).get("slide_index", 0) or 0)), start=1):
        slide = raw_slide if isinstance(raw_slide, dict) else {}
        slide_index = int(slide.get("slide_index", idx) or idx)
        slide_id = f"s{slide_index:03d}"
        title = str(slide.get("title") or "").strip() or f"Slide {slide_index}"

        bullets_raw = slide.get("bullets") if isinstance(slide.get("bullets"), list) else []
        bullets_out: List[Dict[str, Any]] = []
        warning_flags: set[str] = set()

        for b_idx, raw_bullet in enumerate(bullets_raw, start=1):
            bullet = raw_bullet if isinstance(raw_bullet, dict) else {}
            text = str(bullet.get("text") or "").strip()
            evidence_raw = bullet.get("evidence_items") if isinstance(bullet.get("evidence_items"), list) else []
            visual_ids = sorted({str(x).strip() for x in (bullet.get("visual_asset_ids") or []) if str(x).strip()})

            evidence_out: List[Dict[str, Any]] = []
            for ev in _stable_sort_evidence([x for x in evidence_raw if isinstance(x, dict)]):
                locator = _norm_locator(ev.get("locator"))
                doc_id = str(ev.get("doc_id") or "").strip()
                page = int(locator.get("page_start", 0) or 0)
                preview_rel: Optional[str] = None
                if doc_id and page > 0:
                    prev_path = _ensure_page_preview(
                        doc_id=doc_id,
                        version=version,
                        page=page,
                        cache_root=page_preview_root,
                        sqlite_path=sqlite_path,
                        dpi=page_preview_dpi,
                        pdf_cache=pdf_cache,
                    )
                    if prev_path is not None and prev_path.exists():
                        preview_rel = _to_rel(prev_path, output_html.parent)

                evidence_out.append(
                    {
                        "item_id": str(ev.get("item_id") or "").strip() or None,
                        "doc_id": doc_id or None,
                        "locator": locator,
                        "score": ev.get("score"),
                        "preview_page_path": preview_rel,
                    }
                )

            if not evidence_out:
                warning_flags.add("missing_evidence")
            if _is_numeric_risk_text(text) and len(evidence_out) <= 1:
                warning_flags.add("high_risk_numeric")

            bullets_out.append(
                {
                    "bullet_id": f"{slide_id}.b{b_idx:03d}",
                    "text": text,
                    "evidence_items": evidence_out,
                    "visual_asset_ids": visual_ids,
                    "high_risk": bool(_is_numeric_risk_text(text) and len(evidence_out) <= 1),
                }
            )

        visuals_raw = slide.get("visuals") if isinstance(slide.get("visuals"), list) else []
        visuals_out: List[Dict[str, Any]] = []
        for visual in _stable_sort_visuals([x for x in visuals_raw if isinstance(x, dict)]):
            asset_id = str(visual.get("asset_id") or "").strip()
            render_path = str(visual.get("render_path") or "").strip()
            local_thumb: Optional[str] = None
            table_rel: Optional[str] = None

            if render_path:
                rp = Path(render_path)
                if rp.exists():
                    local_thumb = _to_rel(rp, output_html.parent)

            if bundle_extract_dir and asset_id:
                if local_thumb is None:
                    for ext in (".png", ".jpg", ".jpeg", ".webp"):
                        p = bundle_extract_dir / "assets" / f"{asset_id}{ext}"
                        if p.exists():
                            local_thumb = _to_rel(p, output_html.parent)
                            break
                table_path = bundle_extract_dir / "tables" / f"{asset_id}.json"
                if table_path.exists():
                    table_rel = _to_rel(table_path, output_html.parent)

            exists = bool(visual.get("exists"))
            if local_thumb:
                exists = True
            if not exists:
                warning_flags.add("missing_visual_asset")

            visuals_out.append(
                {
                    "asset_id": asset_id or None,
                    "type": str(visual.get("type") or "").strip() or None,
                    "doc_id": str(visual.get("doc_id") or "").strip() or None,
                    "locator": _norm_locator(visual.get("locator")),
                    "thumbnail_path": local_thumb,
                    "table_json_path": table_rel,
                    "exists": exists,
                }
            )

        slides_out.append(
            {
                "slide_id": slide_id,
                "slide_index": slide_index,
                "title": title,
                "visual_role": str(slide.get("visual_role") or "").strip() or "illustrative",
                "warning_flags": sorted(warning_flags),
                "bullets": bullets_out,
                "visuals": visuals_out,
            }
        )

    return {
        "deck_id": deck_id,
        "deck_path": str(audit.get("deck_path") or "").strip() or None,
        "request": {
            "mode": str(request.get("mode") or "").strip() or None,
            "query": str(request.get("query") or "").strip() or None,
            "outline_title": str(request.get("outline_title") or "").strip() or None,
            "version": version,
            "specialty": str(request.get("specialty") or "").strip() or None,
        },
        "slides": slides_out,
        "summary": audit.get("summary") if isinstance(audit.get("summary"), dict) else {},
    }


def _review_template(data: Dict[str, Any], audit_json_path: Path) -> Dict[str, Any]:
    reviews: List[Dict[str, Any]] = []
    for slide in data.get("slides", []):
        if not isinstance(slide, dict):
            continue
        slide_id = str(slide.get("slide_id") or "").strip()
        for bullet in slide.get("bullets", []):
            if not isinstance(bullet, dict):
                continue
            reviews.append(
                {
                    "slide_id": slide_id,
                    "bullet_id": str(bullet.get("bullet_id") or "").strip(),
                    "status": None,
                    "note": "",
                    "suggested_rewrite": "",
                }
            )

    return {
        "deck_id": str(data.get("deck_id") or "deck"),
        "audit_json": str(audit_json_path),
        "updated_at": _utc_now_iso(),
        "reviews": reviews,
    }


def _html_page(data: Dict[str, Any], review_path: Path) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True)
    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Audit Review Viewer - __DECK_ID__</title>
  <style>
    :root {{ --bg:#f4f6f8; --card:#ffffff; --ink:#18212b; --muted:#617082; --line:#d7dde5; --warn:#9a5d00; --risk:#8f1f1f; --ok:#0b6e4f; }}
    body {{ margin:0; font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    .layout {{ display:grid; grid-template-columns:320px 1fr; min-height:100vh; }}
    aside {{ border-right:1px solid var(--line); background:#eef2f6; padding:14px; overflow:auto; }}
    main {{ padding:16px; overflow:auto; }}
    .slide-btn {{ width:100%; text-align:left; border:1px solid var(--line); background:#fff; border-radius:10px; padding:10px; margin:0 0 8px 0; cursor:pointer; }}
    .slide-btn.active {{ border-color:#1f5ea7; box-shadow:0 0 0 1px #1f5ea7 inset; }}
    .badge {{ display:inline-block; font-size:11px; padding:1px 6px; border-radius:999px; margin-left:5px; }}
    .badge.warn {{ background:#ffe9ca; color:var(--warn); }}
    .badge.risk {{ background:#ffdce0; color:var(--risk); }}
    .badge.ok {{ background:#d9f7ec; color:var(--ok); }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; margin-bottom:12px; }}
    h1,h2,h3 {{ margin:0 0 8px 0; }}
    .muted {{ color:var(--muted); }}
    .row {{ display:flex; gap:10px; align-items:flex-start; flex-wrap:wrap; }}
    .grow {{ flex:1 1 320px; min-width:280px; }}
    ul {{ margin:6px 0 0 18px; }}
    .ev {{ border:1px solid var(--line); border-radius:8px; padding:8px; margin:6px 0; background:#fbfcfd; }}
    .vis-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; }}
    .thumb {{ width:100%; max-height:220px; object-fit:contain; border:1px solid var(--line); background:#fff; border-radius:8px; }}
    textarea, select {{ width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:8px; padding:7px; font:13px/1.35 inherit; }}
    button {{ border:1px solid #174b86; background:#1e63ae; color:#fff; border-radius:8px; padding:7px 10px; cursor:pointer; }}
    button.secondary {{ background:#fff; color:#1b3d65; border-color:#9bb4cd; }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
    .small {{ font-size:12px; }}
  </style>
</head>
<body>
  <script id=\"audit-json\" type=\"application/json\">__PAYLOAD__</script>
  <div class=\"layout\">
    <aside>
      <h3>Slides</h3>
      <div id=\"slide-list\"></div>
    </aside>
    <main>
      <div class=\"toolbar\">
        <button id=\"download-review\">Download review JSON</button>
        <button class=\"secondary\" id=\"save-review\">Save review file</button>
        <button class=\"secondary\" id=\"load-review\">Load review JSON</button>
        <input id=\"load-review-input\" type=\"file\" accept=\"application/json\" style=\"display:none\" />
      </div>
      <div class=\"small muted\">Default review target: __REVIEW_NAME__</div>
      <div id=\"content\"></div>
    </main>
  </div>

  <script>
  const DATA = JSON.parse(document.getElementById('audit-json').textContent || '{}');
  const slides = Array.isArray(DATA.slides) ? DATA.slides : [];
  const reviewByBullet = new Map();

  function defaultReview() {{
    return {{
      deck_id: DATA.deck_id || 'deck',
      audit_json: null,
      updated_at: null,
      reviews: []
    }};
  }}

  function applyReviewJson(obj) {{
    reviewByBullet.clear();
    const rows = Array.isArray(obj && obj.reviews) ? obj.reviews : [];
    for (const row of rows) {{
      if (!row || typeof row !== 'object') continue;
      const bulletId = String(row.bullet_id || '').trim();
      const status = String(row.status || '').trim();
      if (!bulletId) continue;
      reviewByBullet.set(bulletId, {{
        slide_id: String(row.slide_id || '').trim() || null,
        bullet_id: bulletId,
        status: status || null,
        note: String(row.note || ''),
        suggested_rewrite: String(row.suggested_rewrite || ''),
      }});
    }}
    renderActive(activeSlideIndex);
  }}

  function buildReviewPayload() {{
    const out = defaultReview();
    out.updated_at = new Date().toISOString();
    out.reviews = [];
    const ordered = Array.from(reviewByBullet.values()).sort((a, b) => (a.bullet_id || '').localeCompare(b.bullet_id || ''));
    for (const row of ordered) {{
      const status = String(row.status || '').trim();
      if (!['ok','needs_edit','reject'].includes(status)) continue;
      out.reviews.push({{
        slide_id: row.slide_id || null,
        bullet_id: row.bullet_id,
        status,
        note: String(row.note || ''),
        suggested_rewrite: String(row.suggested_rewrite || ''),
      }});
    }}
    return out;
  }}

  function warningBadge(flag) {{
    if (flag === 'high_risk_numeric') return '<span class="badge risk">high-risk numeric</span>';
    if (flag === 'missing_evidence') return '<span class="badge warn">missing evidence</span>';
    if (flag === 'missing_visual_asset') return '<span class="badge warn">missing visual</span>';
    return `<span class="badge warn">${{flag}}</span>`;
  }}

  let activeSlideIndex = 0;

  function renderSlideList() {{
    const box = document.getElementById('slide-list');
    box.innerHTML = '';
    slides.forEach((slide, idx) => {{
      const btn = document.createElement('button');
      btn.className = 'slide-btn' + (idx === activeSlideIndex ? ' active' : '');
      const flags = Array.isArray(slide.warning_flags) ? slide.warning_flags : [];
      btn.innerHTML = `<strong>${{slide.slide_index}}. ${{slide.title || 'Untitled'}}</strong><br>${{flags.map(warningBadge).join(' ')}}`;
      btn.onclick = () => {{ activeSlideIndex = idx; renderSlideList(); renderActive(idx); }};
      box.appendChild(btn);
    }});
  }}

  function reviewRowForBullet(slide, bullet) {{
    const bulletId = String(bullet.bullet_id || '').trim();
    const existing = reviewByBullet.get(bulletId) || {{ slide_id: slide.slide_id, bullet_id: bulletId, status: null, note: '', suggested_rewrite: '' }};

    const wrapper = document.createElement('div');
    wrapper.className = 'card';

    const evItems = Array.isArray(bullet.evidence_items) ? bullet.evidence_items : [];
    const evHtml = evItems.map((ev) => {{
      const loc = ev && ev.locator ? ev.locator : {{}};
      const p1 = Number(loc.page_start || 0);
      const p2 = Number(loc.page_end || 0);
      const pageLabel = p1 > 0 ? (p1 === p2 ? `p.${{p1}}` : `p.${{p1}}-${{p2}}`) : 'n/a';
      const preview = ev && ev.preview_page_path ? `<a href="${{ev.preview_page_path}}" target="_blank" rel="noopener">Preview page</a>` : '<span class="muted">Preview page n/a</span>';
      return `<div class="ev"><div><strong>${{ev.doc_id || 'n/a'}}</strong> · item=${{ev.item_id || 'n/a'}} · ${{pageLabel}} · score=${{ev.score ?? 'n/a'}}</div><div>${{preview}}</div></div>`;
    }}).join('');

    wrapper.innerHTML = `
      <div><strong>${{bullet.text || '(empty bullet)'}}</strong>${{bullet.high_risk ? ' <span class="badge risk">high risk</span>' : ''}}</div>
      <div class="muted small">bullet_id=${{bulletId}}</div>
      <div>${{evHtml || '<div class="muted">No evidence items</div>'}}</div>
      <div class="row">
        <div class="grow">
          <label class="small muted">Status</label>
          <select data-k="status">
            <option value="">(not reviewed)</option>
            <option value="ok">ok</option>
            <option value="needs_edit">needs_edit</option>
            <option value="reject">reject</option>
          </select>
        </div>
        <div class="grow">
          <label class="small muted">Note</label>
          <textarea rows="2" data-k="note"></textarea>
        </div>
        <div class="grow">
          <label class="small muted">Suggested rewrite (optional)</label>
          <textarea rows="2" data-k="suggested_rewrite"></textarea>
        </div>
      </div>
    `;

    const sel = wrapper.querySelector('select[data-k="status"]');
    const note = wrapper.querySelector('textarea[data-k="note"]');
    const rewrite = wrapper.querySelector('textarea[data-k="suggested_rewrite"]');
    if (sel) sel.value = existing.status || '';
    if (note) note.value = existing.note || '';
    if (rewrite) rewrite.value = existing.suggested_rewrite || '';

    function commit() {{
      reviewByBullet.set(bulletId, {{
        slide_id: slide.slide_id || null,
        bullet_id: bulletId,
        status: sel ? sel.value : null,
        note: note ? note.value : '',
        suggested_rewrite: rewrite ? rewrite.value : '',
      }});
    }}

    if (sel) sel.addEventListener('change', commit);
    if (note) note.addEventListener('input', commit);
    if (rewrite) rewrite.addEventListener('input', commit);

    return wrapper;
  }}

  function renderActive(idx) {{
    const slide = slides[idx];
    const box = document.getElementById('content');
    if (!slide) {{ box.innerHTML = '<div class="card">No slide</div>'; return; }}

    const flags = Array.isArray(slide.warning_flags) ? slide.warning_flags : [];
    const visuals = Array.isArray(slide.visuals) ? slide.visuals : [];

    box.innerHTML = `
      <div class="card">
        <h2>${{slide.slide_index}}. ${{slide.title || 'Untitled'}}</h2>
        <div class="muted">slide_id=${{slide.slide_id}} · visual_role=${{slide.visual_role || 'illustrative'}}</div>
        <div>${{flags.map(warningBadge).join(' ') || '<span class="badge ok">no warnings</span>'}}</div>
      </div>
      <div class="card"><h3>Visuals</h3><div id="visuals"></div></div>
      <div class="card"><h3>Bullets & Evidence</h3><div id="bullets"></div></div>
    `;

    const visualBox = document.getElementById('visuals');
    if (visualBox) {{
      if (!visuals.length) {{
        visualBox.innerHTML = '<div class="muted">No visuals</div>';
      }} else {{
        visualBox.className = 'vis-grid';
        for (const v of visuals) {{
          const card = document.createElement('div');
          card.className = 'card';
          const thumb = v.thumbnail_path ? `<img class="thumb" src="${{v.thumbnail_path}}" alt="${{v.asset_id || 'asset'}}" />` : '<div class="muted">thumbnail n/a</div>';
          const tableLink = v.table_json_path ? `<a href="${{v.table_json_path}}" target="_blank" rel="noopener">tables/${{v.asset_id || ''}}.json</a>` : '<span class="muted">table json n/a</span>';
          const loc = v.locator || {{}};
          card.innerHTML = `
            <div><strong>${{v.asset_id || 'n/a'}}</strong> · ${{v.type || 'n/a'}} · ${{v.doc_id || 'n/a'}}</div>
            <div class="small muted">locator p.${{loc.page_start || 0}}-${{loc.page_end || 0}}</div>
            <div>${{thumb}}</div>
            <div class="small">${{tableLink}}</div>
          `;
          visualBox.appendChild(card);
        }}
      }}
    }}

    const bulletsBox = document.getElementById('bullets');
    if (bulletsBox) {{
      bulletsBox.innerHTML = '';
      const bullets = Array.isArray(slide.bullets) ? slide.bullets : [];
      for (const b of bullets) bulletsBox.appendChild(reviewRowForBullet(slide, b));
    }}
  }}

  function downloadText(filename, text) {{
    const blob = new Blob([text], {{ type: 'application/json;charset=utf-8' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }}

  async function saveWithPicker(filename, text) {{
    if (!window.showSaveFilePicker) return false;
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: filename,
        types: [{{ description: 'JSON', accept: {{ 'application/json': ['.json'] }} }}],
      }});
      const writable = await handle.createWritable();
      await writable.write(text);
      await writable.close();
      return true;
    }} catch (_e) {{
      return false;
    }}
  }}

  document.getElementById('download-review').addEventListener('click', () => {{
    const payload = buildReviewPayload();
    downloadText(`${{DATA.deck_id || 'deck'}}.review.json`, JSON.stringify(payload, null, 2));
  }});

  document.getElementById('save-review').addEventListener('click', async () => {{
    const payload = buildReviewPayload();
    const text = JSON.stringify(payload, null, 2);
    const filename = `${{DATA.deck_id || 'deck'}}.review.json`;
    const ok = await saveWithPicker(filename, text);
    if (!ok) downloadText(filename, text);
  }});

  document.getElementById('load-review').addEventListener('click', () => document.getElementById('load-review-input').click());
  document.getElementById('load-review-input').addEventListener('change', async (ev) => {{
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {{
      const text = await file.text();
      const obj = JSON.parse(text);
      applyReviewJson(obj);
    }} catch (_e) {{
      alert('Invalid review JSON');
    }}
    ev.target.value = '';
  }});

  renderSlideList();
  renderActive(activeSlideIndex);
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__DECK_ID__", str(data.get("deck_id") or "deck"))
        .replace("__PAYLOAD__", payload)
        .replace("__REVIEW_NAME__", review_path.name)
        .replace("{{", "{")
        .replace("}}", "}")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build static audit review viewer HTML + review template + page previews.")
    ap.add_argument("--audit-json", required=True, help="Path to <deck_id>.audit.json")
    ap.add_argument("--evidence-zip", default="", help="Optional <deck_id>.evidence.zip")
    ap.add_argument("--output-html", default="", help="Output viewer HTML path (default: next to audit)")
    ap.add_argument("--output-review", default="", help="Output review JSON path (default: <deck_id>.review.json)")
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_PATH", "catalog.sqlite3"))
    ap.add_argument("--page-preview-root", default="out/assets/pages")
    ap.add_argument("--page-preview-dpi", type=int, default=120)
    args = ap.parse_args()

    audit_path = Path(args.audit_json)
    audit = _load_json(audit_path)
    deck_id = str(audit.get("deck_id") or audit_path.stem.replace(".audit", "")).strip() or "deck"

    output_html = Path(args.output_html) if args.output_html else audit_path.with_suffix(".html")
    output_review = Path(args.output_review) if args.output_review else audit_path.with_name(f"{deck_id}.review.json")
    sqlite_path = Path(args.sqlite)

    bundle_extract_dir: Optional[Path] = None
    evidence_zip = Path(args.evidence_zip) if str(args.evidence_zip).strip() else None
    if evidence_zip and evidence_zip.exists():
        bundle_extract_dir = output_html.with_suffix("").with_name(f"{deck_id}.evidence")
        _extract_bundle(evidence_zip, bundle_extract_dir)

    data = _normalize_for_viewer(
        audit=audit,
        output_html=output_html,
        sqlite_path=sqlite_path,
        page_preview_root=Path(args.page_preview_root),
        page_preview_dpi=max(72, int(args.page_preview_dpi)),
        bundle_extract_dir=bundle_extract_dir,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_html_page(data, output_review), encoding="utf-8")

    review_template = _review_template(data, audit_path)
    output_review.parent.mkdir(parents=True, exist_ok=True)
    output_review.write_text(json.dumps(review_template, ensure_ascii=False, indent=2), encoding="utf-8")

    slides = data.get("slides") if isinstance(data.get("slides"), list) else []
    bullets = sum(len(s.get("bullets") or []) for s in slides if isinstance(s, dict))
    high_risk = sum(
        1
        for s in slides
        if isinstance(s, dict)
        for b in (s.get("bullets") or [])
        if isinstance(b, dict) and bool(b.get("high_risk"))
    )

    print("PASS")
    print(f"html={output_html}")
    print(f"review_template={output_review}")
    if bundle_extract_dir:
        print(f"evidence_dir={bundle_extract_dir}")
    print(f"slides={len(slides)} bullets={bullets} high_risk={high_risk}")


if __name__ == "__main__":
    main()
