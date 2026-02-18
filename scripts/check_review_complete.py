#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_STATUSES = {"ok", "needs_edit", "reject"}
_NUMERIC_HINTS = ("%", "mean", "median", "ratio", "rate", "mm", "cm", "n=", "p<", "vs")


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"FAIL: {label} file not found: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"FAIL: {label} root must be a JSON object")
    return obj


def _is_high_risk(text: str, evidence_count: int) -> bool:
    if evidence_count > 1:
        return False
    t = text.lower()
    if any(tok in t for tok in _NUMERIC_HINTS):
        return True
    return any(ch.isdigit() for ch in t)


def _audit_bullets(audit: Dict[str, Any]) -> List[Tuple[str, str, str, int, bool]]:
    slides = audit.get("slides") if isinstance(audit.get("slides"), list) else []
    out: List[Tuple[str, str, str, int, bool]] = []
    for s_idx, slide in enumerate(slides, start=1):
        s_obj = slide if isinstance(slide, dict) else {}
        slide_index = int(s_obj.get("slide_index", s_idx) or s_idx)
        slide_id = f"s{slide_index:03d}"
        bullets = s_obj.get("bullets") if isinstance(s_obj.get("bullets"), list) else []
        for b_idx, bullet in enumerate(bullets, start=1):
            b_obj = bullet if isinstance(bullet, dict) else {}
            bullet_id = f"{slide_id}.b{b_idx:03d}"
            text = str(b_obj.get("text") or "").strip()
            evidence = b_obj.get("evidence_items") if isinstance(b_obj.get("evidence_items"), list) else []
            ev_count = len([x for x in evidence if isinstance(x, dict)])
            out.append((slide_id, bullet_id, text, ev_count, _is_high_risk(text, ev_count)))
    return out


def _review_map(review: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = review.get("reviews") if isinstance(review.get("reviews"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        bullet_id = str(raw.get("bullet_id") or "").strip()
        if not bullet_id:
            continue
        status = str(raw.get("status") or "").strip()
        out[bullet_id] = {
            "slide_id": str(raw.get("slide_id") or "").strip() or None,
            "bullet_id": bullet_id,
            "status": status if status in _STATUSES else None,
            "note": str(raw.get("note") or ""),
            "suggested_rewrite": str(raw.get("suggested_rewrite") or ""),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Optional human gate: fail if review contains reject, or if high-risk bullets have no review status."
        )
    )
    ap.add_argument("--audit-json", required=True, help="Path to <deck_id>.audit.json")
    ap.add_argument(
        "--review-json",
        default="",
        help="Path to <deck_id>.review.json (default: derive from audit deck_id)",
    )
    args = ap.parse_args()

    audit_path = Path(args.audit_json)
    audit = _load_json(audit_path, "audit")
    deck_id = str(audit.get("deck_id") or audit_path.stem.replace(".audit", "")).strip() or "deck"
    review_path = Path(args.review_json) if args.review_json else audit_path.with_name(f"{deck_id}.review.json")
    review = _load_json(review_path, "review")

    bullets = _audit_bullets(audit)
    by_bullet = _review_map(review)

    errors: List[str] = []
    reject_count = 0
    reviewed_count = 0
    high_risk_total = 0
    high_risk_missing = 0

    for slide_id, bullet_id, text, _ev_count, high_risk in bullets:
        row = by_bullet.get(bullet_id)
        status = str((row or {}).get("status") or "").strip()
        if status:
            reviewed_count += 1
        if status == "reject":
            reject_count += 1
            errors.append(f"reject: {bullet_id} ({text[:90]})")
        if high_risk:
            high_risk_total += 1
            if status not in _STATUSES:
                high_risk_missing += 1
                errors.append(f"missing high-risk review: {bullet_id} ({text[:90]})")

    if errors:
        print("FAIL")
        for err in errors:
            print(f"- {err}")
        print(
            f"summary: bullets={len(bullets)} reviewed={reviewed_count} "
            f"high_risk={high_risk_total} high_risk_missing={high_risk_missing} rejects={reject_count}"
        )
        raise SystemExit(1)

    print("PASS")
    print(
        f"bullets={len(bullets)} reviewed={reviewed_count} "
        f"high_risk={high_risk_total} high_risk_missing=0 rejects=0"
    )


if __name__ == "__main__":
    main()
