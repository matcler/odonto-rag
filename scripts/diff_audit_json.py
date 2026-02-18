#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_VOLATILE_KEYS = {"timestamp", "run_id"}


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FAIL: file not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"FAIL: root must be object in {path}")
    return payload


def _drop_volatile(node: Any) -> Any:
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k in _VOLATILE_KEYS:
                continue
            out[k] = _drop_volatile(v)
        return out
    if isinstance(node, list):
        return [_drop_volatile(x) for x in node]
    return node


def _norm_locator(raw: Any) -> Dict[str, Any]:
    loc = raw if isinstance(raw, dict) else {}
    return {
        "page_start": int(loc.get("page_start", 0) or 0),
        "page_end": int(loc.get("page_end", 0) or 0),
    }


def _norm_evidence_items(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    items: List[Dict[str, Any]] = []
    for ev in raw:
        if not isinstance(ev, dict):
            continue
        item_id = str(ev.get("item_id") or "").strip()
        if not item_id:
            continue
        items.append(
            {
                "item_id": item_id,
                "doc_id": str(ev.get("doc_id") or "").strip() or None,
                "locator": _norm_locator(ev.get("locator")),
                "score": ev.get("score"),
            }
        )
    items.sort(
        key=lambda x: (
            x.get("item_id") or "",
            x.get("doc_id") or "",
            x.get("locator", {}).get("page_start", 0),
            x.get("locator", {}).get("page_end", 0),
            str(x.get("score")),
        )
    )
    return items


def _norm_bullets(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    bullets: List[Dict[str, Any]] = []
    for bullet in raw:
        if not isinstance(bullet, dict):
            continue
        text = str(bullet.get("text") or "").strip()
        visual_ids = [str(v).strip() for v in (bullet.get("visual_asset_ids") or []) if str(v).strip()]
        bullets.append(
            {
                "text": text,
                "evidence_items": _norm_evidence_items(bullet.get("evidence_items")),
                "visual_asset_ids": sorted(set(visual_ids)),
            }
        )
    bullets.sort(key=lambda x: (x.get("text") or "", len(x.get("evidence_items", []))))
    return bullets


def _norm_visuals(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    visuals: List[Dict[str, Any]] = []
    for v in raw:
        if not isinstance(v, dict):
            continue
        visuals.append(
            {
                "asset_id": str(v.get("asset_id") or "").strip() or None,
                "type": str(v.get("type") or "").strip() or None,
                "doc_id": str(v.get("doc_id") or "").strip() or None,
                "locator": _norm_locator(v.get("locator")),
                "specialty": str(v.get("specialty") or "").strip() or None,
                "exists": bool(v.get("exists")),
            }
        )
    visuals.sort(key=lambda x: (x.get("asset_id") or "", x.get("doc_id") or "", x.get("type") or ""))
    return visuals


def _norm_slides(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    slides: List[Dict[str, Any]] = []
    for idx, slide in enumerate(raw, start=1):
        if not isinstance(slide, dict):
            continue
        slide_index = int(slide.get("slide_index", idx) or idx)
        slides.append(
            {
                "slide_index": slide_index,
                "title": str(slide.get("title") or "").strip(),
                "visual_role": str(slide.get("visual_role") or "").strip() or "illustrative",
                "bullets": _norm_bullets(slide.get("bullets")),
                "visuals": _norm_visuals(slide.get("visuals")),
            }
        )
    slides.sort(key=lambda x: (x.get("slide_index", 0), x.get("title") or ""))
    return slides


def normalize_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = _drop_volatile(payload)
    request = base.get("request") if isinstance(base.get("request"), dict) else {}
    env_profile = request.get("env_profile") if isinstance(request.get("env_profile"), dict) else {}
    summary = base.get("summary") if isinstance(base.get("summary"), dict) else {}

    gates = summary.get("gates_applied") if isinstance(summary.get("gates_applied"), dict) else {}
    normalized = {
        "deck_id": str(base.get("deck_id") or "").strip() or None,
        "request": {
            "mode": str(request.get("mode") or "").strip() or None,
            "query": str(request.get("query") or "").strip() or None,
            "outline_title": str(request.get("outline_title") or "").strip() or None,
            "version": str(request.get("version") or "").strip() or None,
            "specialty": str(request.get("specialty") or "").strip() or None,
            "env_profile": {k: bool(v) for k, v in sorted(env_profile.items())},
        },
        "slides": _norm_slides(base.get("slides")),
        "summary": {
            "unique_docs": sorted({str(x).strip() for x in (summary.get("unique_docs") or []) if str(x).strip()}),
            "unique_items": sorted({str(x).strip() for x in (summary.get("unique_items") or []) if str(x).strip()}),
            "unique_assets": sorted({str(x).strip() for x in (summary.get("unique_assets") or []) if str(x).strip()}),
            "missing_asset_count": int(summary.get("missing_asset_count", 0) or 0),
            "missing_assets": sorted({str(x).strip() for x in (summary.get("missing_assets") or []) if str(x).strip()}),
            "gates_applied": {k: bool(v) for k, v in sorted(gates.items())},
        },
    }
    return normalized


def normalized_sha256(payload: Dict[str, Any]) -> str:
    normalized = normalize_audit(payload)
    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bullets_map(slide: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for b in slide.get("bullets", []):
        if not isinstance(b, dict):
            continue
        text = str(b.get("text") or "").strip()
        if text:
            out[text] = b
    return out


def _evidence_signature(bullet: Dict[str, Any]) -> List[Tuple[str, str, int, int]]:
    sig: List[Tuple[str, str, int, int]] = []
    for ev in bullet.get("evidence_items", []):
        if not isinstance(ev, dict):
            continue
        loc = ev.get("locator") if isinstance(ev.get("locator"), dict) else {}
        sig.append(
            (
                str(ev.get("item_id") or "").strip(),
                str(ev.get("doc_id") or "").strip(),
                int(loc.get("page_start", 0) or 0),
                int(loc.get("page_end", 0) or 0),
            )
        )
    sig.sort()
    return sig


def semantic_diff(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    changes: List[str] = []

    old_slides = {int(s.get("slide_index", 0) or 0): s for s in old.get("slides", []) if isinstance(s, dict)}
    new_slides = {int(s.get("slide_index", 0) or 0): s for s in new.get("slides", []) if isinstance(s, dict)}

    for idx in sorted(set(old_slides.keys()) | set(new_slides.keys())):
        s_old = old_slides.get(idx)
        s_new = new_slides.get(idx)
        if s_old is None:
            changes.append(f"slide {idx}: added")
            continue
        if s_new is None:
            changes.append(f"slide {idx}: removed")
            continue

        old_texts = [str(b.get("text") or "") for b in s_old.get("bullets", []) if isinstance(b, dict)]
        new_texts = [str(b.get("text") or "") for b in s_new.get("bullets", []) if isinstance(b, dict)]
        if old_texts != new_texts:
            changes.append(f"slide {idx}: bullet text changed")

        old_bullets = _bullets_map(s_old)
        new_bullets = _bullets_map(s_new)
        for text in sorted(set(old_bullets.keys()) | set(new_bullets.keys())):
            b_old = old_bullets.get(text)
            b_new = new_bullets.get(text)
            if b_old is None or b_new is None:
                continue
            if _evidence_signature(b_old) != _evidence_signature(b_new):
                changes.append(f"slide {idx}: evidence items changed for bullet '{text}'")
            old_links = sorted(set(str(x).strip() for x in (b_old.get("visual_asset_ids") or []) if str(x).strip()))
            new_links = sorted(set(str(x).strip() for x in (b_new.get("visual_asset_ids") or []) if str(x).strip()))
            if old_links != new_links:
                changes.append(f"slide {idx}: visual links changed for bullet '{text}'")

        old_visuals = [
            (str(v.get("asset_id") or ""), str(v.get("doc_id") or ""), str(v.get("type") or ""), bool(v.get("exists")))
            for v in s_old.get("visuals", [])
            if isinstance(v, dict)
        ]
        new_visuals = [
            (str(v.get("asset_id") or ""), str(v.get("doc_id") or ""), str(v.get("type") or ""), bool(v.get("exists")))
            for v in s_new.get("visuals", [])
            if isinstance(v, dict)
        ]
        if sorted(old_visuals) != sorted(new_visuals):
            changes.append(f"slide {idx}: visual links changed")

    old_missing = int((old.get("summary") or {}).get("missing_asset_count", 0) or 0)
    new_missing = int((new.get("summary") or {}).get("missing_asset_count", 0) or 0)
    if old_missing != new_missing:
        changes.append(f"missing_asset_count changed: {old_missing} -> {new_missing}")

    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic semantic diff for *.audit.json with canonical normalization")
    ap.add_argument("old", nargs="?", help="Old audit JSON")
    ap.add_argument("new", nargs="?", help="New audit JSON")
    ap.add_argument("--sha256", default="", help="Print normalized SHA256 for a single audit JSON and exit")
    ap.add_argument("--fail-on-diff", action="store_true", help="Exit 1 when semantic or normalized diff is detected")
    args = ap.parse_args()

    if args.sha256:
        payload = _load_json(args.sha256)
        print(normalized_sha256(payload))
        return

    if not args.old or not args.new:
        raise SystemExit("FAIL: provide OLD NEW audit JSON paths (or --sha256 <file>)")

    old_raw = _load_json(args.old)
    new_raw = _load_json(args.new)
    old_norm = normalize_audit(old_raw)
    new_norm = normalize_audit(new_raw)

    old_sha = normalized_sha256(old_raw)
    new_sha = normalized_sha256(new_raw)

    print(f"old_sha256={old_sha}")
    print(f"new_sha256={new_sha}")

    changes = semantic_diff(old_norm, new_norm)
    if not changes and old_norm != new_norm:
        changes.append("normalized audit changed (non-semantic fields)")

    if changes:
        print("DIFF")
        for c in changes:
            print(f"- {c}")
        if args.fail_on_diff:
            raise SystemExit(1)
    else:
        print("NO_DIFF")


if __name__ == "__main__":
    main()
