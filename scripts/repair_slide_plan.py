#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "onto",
    "about",
    "over",
    "under",
    "using",
    "use",
    "how",
    "why",
    "what",
    "when",
    "where",
    "which",
    "while",
    "your",
    "their",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "not",
    "are",
    "was",
    "is",
    "of",
    "to",
    "in",
    "on",
    "a",
    "an",
    "or",
    "as",
    "by",
    "at",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"FAIL: file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"FAIL: root must be object in {path}")
    return payload


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in _TRUE_VALUES


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS and not tok.isdigit()
    }


def _contains_numeric(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _as_list_of_lists(raw: Any, size: int) -> List[List[Any]]:
    out: List[List[Any]] = []
    if isinstance(raw, list):
        for item in raw[:size]:
            if isinstance(item, list):
                out.append(list(item))
            else:
                out.append([])
    while len(out) < size:
        out.append([])
    return out


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    txt = " ".join(str(text or "").split()).strip()
    if len(txt) <= max_chars:
        return txt
    if max_chars <= 1:
        return "…"
    return txt[: max_chars - 1].rstrip() + "…"


def _split_or_truncate(text: str, max_chars: int) -> List[str]:
    txt = " ".join(str(text or "").split()).strip()
    if len(txt) <= max_chars:
        return [txt]

    # Deterministic split before truncation.
    candidates: List[int] = []
    for sep in (". ", "; ", ": ", ", "):
        start = 0
        while True:
            idx = txt.find(sep, start)
            if idx < 0:
                break
            split_at = idx + len(sep)
            if int(0.55 * max_chars) <= split_at <= max_chars:
                candidates.append(split_at)
            start = idx + 1
    if candidates:
        split_at = min(candidates)
        head = txt[:split_at].strip()
        tail = txt[split_at:].strip()
        if head and tail:
            return [_truncate_with_ellipsis(head, max_chars), _truncate_with_ellipsis(tail, max_chars)]

    return [_truncate_with_ellipsis(txt, max_chars)]


def _score_value(item: Dict[str, Any]) -> float:
    raw = item.get("score")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except Exception:
            return 0.0
    return 0.0


def _stable_evidence_sort_key(item: Dict[str, Any], idx: int) -> Tuple[float, str, str, int, int, int]:
    loc = item.get("locator") if isinstance(item.get("locator"), dict) else {}
    page_start = int(loc.get("page_start", 0) or 0)
    page_end = int(loc.get("page_end", 0) or 0)
    return (
        -_score_value(item),
        str(item.get("item_id") or ""),
        str(item.get("doc_id") or ""),
        page_start,
        page_end,
        idx,
    )


def _extract_table_rows(visual: Dict[str, Any]) -> Optional[List[List[Any]]]:
    rows = visual.get("table_rows")
    if isinstance(rows, list):
        return [list(r) if isinstance(r, list) else [r] for r in rows]
    table_obj = visual.get("table")
    if isinstance(table_obj, dict) and isinstance(table_obj.get("rows"), list):
        data = table_obj.get("rows")
        return [list(r) if isinstance(r, list) else [r] for r in data]
    return None


def _write_table_rows(visual: Dict[str, Any], rows: List[List[Any]]) -> None:
    visual["table_rows"] = rows
    table_obj = visual.get("table")
    if isinstance(table_obj, dict):
        table_obj["rows"] = rows


def _visual_tokens(visual: Dict[str, Any]) -> set[str]:
    pieces = [
        str(visual.get("asset_id") or ""),
        str(visual.get("asset_type") or ""),
        str(visual.get("caption") or ""),
        str(visual.get("doc_id") or ""),
    ]
    rows = _extract_table_rows(visual)
    if rows and rows[0]:
        pieces.extend(str(cell or "") for cell in rows[0][:6])
    return _tokenize(" ".join(pieces))


def _repair_plan(plan: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    repaired = copy.deepcopy(plan)
    slides = repaired.get("slides") if isinstance(repaired.get("slides"), list) else []
    repairs: List[Dict[str, Any]] = []

    for s_idx, raw_slide in enumerate(slides, start=1):
        if not isinstance(raw_slide, dict):
            continue
        slide = raw_slide
        slide_no = int(slide.get("slide_no", s_idx) or s_idx)

        bullets_raw = slide.get("bullets") if isinstance(slide.get("bullets"), list) else []
        bullets = [" ".join(str(b or "").split()).strip() for b in bullets_raw]
        src_ids = _as_list_of_lists(slide.get("bullet_source_item_ids"), len(bullets))
        src_structured = _as_list_of_lists(slide.get("bullet_sources_structured"), len(bullets))
        vis_links = _as_list_of_lists(slide.get("bullet_visual_asset_ids"), len(bullets))

        # Readability: split/truncate long bullets deterministically.
        new_bullets: List[str] = []
        new_src_ids: List[List[Any]] = []
        new_src_structured: List[List[Any]] = []
        new_vis_links: List[List[Any]] = []

        for b_idx, text in enumerate(bullets, start=1):
            split_parts = _split_or_truncate(text, args.long_bullet_chars)
            if len(split_parts) > 1:
                repairs.append(
                    {
                        "slide_no": slide_no,
                        "bullet_index": b_idx,
                        "type": "split_long_bullet",
                        "before_chars": len(text),
                        "after_parts": len(split_parts),
                        "max_chars": args.long_bullet_chars,
                    }
                )
            elif split_parts[0] != text:
                repairs.append(
                    {
                        "slide_no": slide_no,
                        "bullet_index": b_idx,
                        "type": "truncate_long_bullet",
                        "before_chars": len(text),
                        "after_chars": len(split_parts[0]),
                        "max_chars": args.long_bullet_chars,
                    }
                )

            for part in split_parts:
                new_bullets.append(part)
                new_src_ids.append(list(src_ids[b_idx - 1]))
                new_src_structured.append(list(src_structured[b_idx - 1]))
                new_vis_links.append(list(vis_links[b_idx - 1]))

        bullets = new_bullets
        src_ids = new_src_ids
        src_structured = new_src_structured
        vis_links = new_vis_links

        # Evidence overload (hard only): keep top-N evidence by score, stable tie-break.
        if args.mode == "hard" and args.max_evidence_per_bullet > 0:
            for b_idx in range(len(bullets)):
                structured = src_structured[b_idx]
                if len(structured) > args.max_evidence_per_bullet:
                    structured_dicts = [x for x in structured if isinstance(x, dict)]
                    enumerated = list(enumerate(structured_dicts))
                    enumerated.sort(key=lambda pair: _stable_evidence_sort_key(pair[1], pair[0]))
                    keep = [item for _, item in enumerated[: args.max_evidence_per_bullet]]
                    keep_item_ids = [str(x.get("item_id") or "").strip() for x in keep if str(x.get("item_id") or "").strip()]
                    src_structured[b_idx] = keep
                    src_ids[b_idx] = keep_item_ids
                    repairs.append(
                        {
                            "slide_no": slide_no,
                            "bullet_index": b_idx + 1,
                            "type": "trim_evidence_per_bullet",
                            "before": len(structured),
                            "after": len(keep),
                            "max_evidence_per_bullet": args.max_evidence_per_bullet,
                        }
                    )
                elif len(src_ids[b_idx]) > args.max_evidence_per_bullet:
                    src_ids[b_idx] = src_ids[b_idx][: args.max_evidence_per_bullet]
                    repairs.append(
                        {
                            "slide_no": slide_no,
                            "bullet_index": b_idx + 1,
                            "type": "trim_source_item_ids",
                            "after": len(src_ids[b_idx]),
                            "max_evidence_per_bullet": args.max_evidence_per_bullet,
                        }
                    )

        # Readability cap per slide.
        if args.max_bullets_per_slide > 0 and len(bullets) > args.max_bullets_per_slide:
            before_count = len(bullets)
            bullets = bullets[: args.max_bullets_per_slide]
            src_ids = src_ids[: args.max_bullets_per_slide]
            src_structured = src_structured[: args.max_bullets_per_slide]
            vis_links = vis_links[: args.max_bullets_per_slide]
            repairs.append(
                {
                    "slide_no": slide_no,
                    "type": "cap_bullets_per_slide",
                    "before": before_count,
                    "after": len(bullets),
                    "max_bullets_per_slide": args.max_bullets_per_slide,
                }
            )

        slide["bullets"] = bullets
        slide["bullet_source_item_ids"] = src_ids
        slide["bullet_sources_structured"] = src_structured
        slide["bullet_visual_asset_ids"] = vis_links

        # Visual coherence (soft+hard): deterministic linking for evidence slides with no links.
        role = str(slide.get("visual_role") or "").strip().lower()
        visuals = [v for v in (slide.get("visuals") or []) if isinstance(v, dict)]
        if role == "evidence" and visuals:
            any_linked = any(bool(vs) for vs in vis_links)
            if not any_linked:
                visuals_by_id = {
                    str(v.get("asset_id") or "").strip(): v
                    for v in visuals
                    if str(v.get("asset_id") or "").strip()
                }
                table_ids = sorted(
                    aid
                    for aid, v in visuals_by_id.items()
                    if str(v.get("asset_type") or "").strip().lower() == "table"
                )
                v_tokens = {aid: _visual_tokens(v) for aid, v in visuals_by_id.items()}

                linked_count = 0
                for b_idx, text in enumerate(bullets, start=1):
                    current = [str(x).strip() for x in vis_links[b_idx - 1] if str(x).strip()]
                    if current:
                        continue
                    if _contains_numeric(text) and table_ids:
                        vis_links[b_idx - 1] = [table_ids[0]]
                        linked_count += 1
                        repairs.append(
                            {
                                "slide_no": slide_no,
                                "bullet_index": b_idx,
                                "type": "link_numeric_bullet_to_table",
                                "asset_id": table_ids[0],
                            }
                        )
                        continue

                    b_tokens = _tokenize(text)
                    ranked: List[Tuple[int, str]] = []
                    for aid in sorted(visuals_by_id.keys()):
                        overlap = len(b_tokens.intersection(v_tokens.get(aid, set())))
                        if overlap > 0:
                            ranked.append((overlap, aid))
                    if ranked:
                        ranked.sort(key=lambda x: (-x[0], x[1]))
                        chosen = ranked[0][1]
                        vis_links[b_idx - 1] = [chosen]
                        linked_count += 1
                        repairs.append(
                            {
                                "slide_no": slide_no,
                                "bullet_index": b_idx,
                                "type": "link_bullet_to_visual_by_keyword",
                                "asset_id": chosen,
                                "keyword_overlap": ranked[0][0],
                            }
                        )

                slide["bullet_visual_asset_ids"] = vis_links
                if linked_count == 0 and args.downgrade_visual_role_on_no_link:
                    slide["visual_role"] = "illustrative"
                    repairs.append(
                        {
                            "slide_no": slide_no,
                            "type": "downgrade_visual_role_on_no_link",
                            "before": "evidence",
                            "after": "illustrative",
                        }
                    )

        # Table density (hard only): deterministic compact policy.
        if args.mode == "hard":
            for visual in visuals:
                asset_id = str(visual.get("asset_id") or "").strip() or None
                vtype = str(visual.get("asset_type") or "").strip().lower()
                if vtype != "table":
                    continue
                rows = _extract_table_rows(visual)
                if not rows:
                    continue
                row_count = len(rows)
                col_count = max((len(r) for r in rows), default=0)
                cells = row_count * col_count
                if (
                    row_count > args.table_max_rows
                    or col_count > args.table_max_cols
                    or cells > args.table_max_cells
                ):
                    compact_rows = [list(r)[: args.table_max_cols] for r in rows[: args.table_max_rows]]
                    _write_table_rows(visual, compact_rows)
                    visual["repair_table_mode"] = "compact"
                    repairs.append(
                        {
                            "slide_no": slide_no,
                            "type": "compact_table",
                            "asset_id": asset_id,
                            "before": {"rows": row_count, "cols": col_count, "cells": cells},
                            "after": {
                                "rows": len(compact_rows),
                                "cols": max((len(r) for r in compact_rows), default=0),
                                "cells": len(compact_rows)
                                * max((len(r) for r in compact_rows), default=0),
                            },
                            "limits": {
                                "rows": args.table_max_rows,
                                "cols": args.table_max_cols,
                                "cells": args.table_max_cells,
                            },
                            "policy": "compact_mode",
                        }
                    )

    return repaired, repairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic slide plan normalizer/repair")
    ap.add_argument("--plan-json", required=True, help="Input plan JSON")
    ap.add_argument("--out-plan-json", default="", help="Output repaired plan path")
    ap.add_argument("--out-repairs-json", default="", help="Output machine-readable repairs JSON")
    ap.add_argument("--mode", choices=["soft", "hard"], default="soft")

    ap.add_argument("--max-bullets-per-slide", type=int, default=_env_int("SLIDES_REPAIR_MAX_BULLETS_PER_SLIDE", 6))
    ap.add_argument("--max-evidence-per-bullet", type=int, default=_env_int("SLIDES_REPAIR_MAX_EVIDENCE_PER_BULLET", 3))
    ap.add_argument("--long-bullet-chars", type=int, default=_env_int("SLIDES_REPAIR_LONG_BULLET_CHARS", 140))
    ap.add_argument(
        "--downgrade-visual-role-on-no-link",
        type=int,
        choices=[0, 1],
        default=1 if _env_flag("SLIDES_REPAIR_DOWNGRADE_VISUAL_ROLE_ON_NO_LINK", True) else 0,
    )
    ap.add_argument("--table-max-rows", type=int, default=12)
    ap.add_argument("--table-max-cols", type=int, default=6)
    ap.add_argument("--table-max-cells", type=int, default=60)
    args = ap.parse_args()

    in_path = Path(args.plan_json)
    plan = _load_json(in_path)

    out_plan = Path(args.out_plan_json) if args.out_plan_json else in_path.with_suffix("").with_suffix(".repaired.json")
    out_repairs = (
        Path(args.out_repairs_json)
        if args.out_repairs_json
        else out_plan.with_name(out_plan.name.replace(".json", ".repairs.json"))
    )
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_repairs.parent.mkdir(parents=True, exist_ok=True)

    args.downgrade_visual_role_on_no_link = bool(args.downgrade_visual_role_on_no_link)
    repaired, repairs = _repair_plan(plan, args)

    out_plan.write_text(json.dumps(repaired, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    repairs_payload = {
        "mode": args.mode,
        "input_plan": str(in_path),
        "output_plan": str(out_plan),
        "policy": {
            "max_bullets_per_slide": args.max_bullets_per_slide,
            "max_evidence_per_bullet": args.max_evidence_per_bullet,
            "long_bullet_chars": args.long_bullet_chars,
            "downgrade_visual_role_on_no_link": bool(args.downgrade_visual_role_on_no_link),
            "table_max_rows": args.table_max_rows,
            "table_max_cols": args.table_max_cols,
            "table_max_cells": args.table_max_cells,
        },
        "repairs": repairs,
        "repairs_count": len(repairs),
    }
    out_repairs.write_text(json.dumps(repairs_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("PASS")
    print(f"out_plan={out_plan}")
    print(f"out_repairs={out_repairs}")
    print(f"repairs_count={len(repairs)}")


if __name__ == "__main__":
    main()
