#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FAIL: audit file not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("FAIL: audit root must be a JSON object")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate S6 audit pack: bullets evidence, visuals existence/specialty, missing-asset policy."
    )
    ap.add_argument("--audit-json", required=True, help="Path to <deck_id>.audit.json")
    ap.add_argument("--specialty", default="", help="Expected specialty (optional)")
    ap.add_argument(
        "--allow-missing-assets",
        action="store_true",
        help="Allow non-zero missing_asset_count in summary",
    )
    args = ap.parse_args()

    audit = _load_json(args.audit_json)
    slides = audit.get("slides")
    if not isinstance(slides, list) or not slides:
        raise SystemExit("FAIL: audit has no slides")

    expected_specialty = args.specialty.strip().lower()
    errors: List[str] = []
    missing_assets_runtime = 0

    for slide_idx, slide in enumerate(slides, start=1):
        if not isinstance(slide, dict):
            errors.append(f"slide {slide_idx}: not an object")
            continue
        bullets = slide.get("bullets")
        if not isinstance(bullets, list) or not bullets:
            errors.append(f"slide {slide_idx}: no bullets")
        else:
            for bullet_idx, bullet in enumerate(bullets, start=1):
                if not isinstance(bullet, dict):
                    errors.append(f"slide {slide_idx} bullet {bullet_idx}: not an object")
                    continue
                evidence = bullet.get("evidence_items")
                if not isinstance(evidence, list) or not evidence:
                    errors.append(f"slide {slide_idx} bullet {bullet_idx}: no evidence_items")
                    continue
                for ev in evidence:
                    if not isinstance(ev, dict):
                        errors.append(f"slide {slide_idx} bullet {bullet_idx}: invalid evidence item")
                        continue
                    if not str(ev.get("item_id") or "").strip():
                        errors.append(f"slide {slide_idx} bullet {bullet_idx}: evidence missing item_id")
                    if not str(ev.get("doc_id") or "").strip():
                        errors.append(f"slide {slide_idx} bullet {bullet_idx}: evidence missing doc_id")
                    locator = ev.get("locator")
                    if not isinstance(locator, dict):
                        errors.append(f"slide {slide_idx} bullet {bullet_idx}: evidence missing locator")

        visuals = slide.get("visuals")
        if not isinstance(visuals, list):
            errors.append(f"slide {slide_idx}: visuals must be a list")
            continue
        for visual in visuals:
            if not isinstance(visual, dict):
                errors.append(f"slide {slide_idx}: invalid visual object")
                continue
            asset_id = str(visual.get("asset_id") or "").strip()
            exists = bool(visual.get("exists"))
            if not asset_id:
                errors.append(f"slide {slide_idx}: visual missing asset_id")
            if not exists:
                missing_assets_runtime += 1
                render_path = str(visual.get("render_path") or "").strip()
                if render_path and Path(render_path).exists():
                    exists = True
            if expected_specialty:
                visual_specialty = str(visual.get("specialty") or "").strip().lower()
                if visual_specialty and visual_specialty != expected_specialty:
                    errors.append(
                        f"slide {slide_idx}: visual {asset_id or '<missing>'} specialty mismatch "
                        f"({visual_specialty} != {expected_specialty})"
                    )

    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    reported_missing = int(summary.get("missing_asset_count", 0))
    if reported_missing != missing_assets_runtime:
        errors.append(
            f"summary.missing_asset_count={reported_missing} != runtime_detected={missing_assets_runtime}"
        )
    if reported_missing > 0 and not args.allow_missing_assets:
        errors.append(
            "missing_asset_count > 0 but policy disallows missing assets "
            "(pass --allow-missing-assets to override)"
        )

    if errors:
        print("FAIL")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("PASS")
    print(f"slides={len(slides)}")
    print(f"missing_asset_count={reported_missing}")


if __name__ == "__main__":
    main()
