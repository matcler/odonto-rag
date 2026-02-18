#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

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

_BBOX_KEYS = {"bbox", "normalized_bbox", "vertices", "normalized_vertices", "x0", "x1", "y0", "y1"}
_SEGMENT_KEYS = {
    "segment",
    "segment_id",
    "paragraph_id",
    "line_id",
    "char_start",
    "char_end",
    "text_start",
    "text_end",
    "offset_start",
    "offset_end",
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


def _safe_name(value: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in value)
    return (out.strip("._") or "quality").lower()


def _tokenize(text: str) -> Set[str]:
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS and not tok.isdigit()
    }
    return tokens


def _contains_numeric(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _bool_trueish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _has_nonempty(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _locator_flags(raw: Any) -> Tuple[bool, bool, bool]:
    loc = raw if isinstance(raw, dict) else {}
    page_start = int(loc.get("page_start", 0) or 0)
    page_end = int(loc.get("page_end", 0) or 0)
    has_page = page_start > 0 or page_end > 0
    has_bbox = any(k in loc and _has_nonempty(loc.get(k)) for k in _BBOX_KEYS)
    has_segment = any(k in loc and _has_nonempty(loc.get(k)) for k in _SEGMENT_KEYS)
    is_complete = has_page and (has_bbox or has_segment)
    page_only = has_page and not (has_bbox or has_segment)
    return has_page, is_complete, page_only


def _find_path(obj: Any, path: List[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _first_int(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float):
        as_int = int(raw)
        return as_int if as_int > 0 else None
    if isinstance(raw, str):
        txt = raw.strip()
        if txt.isdigit():
            parsed = int(txt)
            return parsed if parsed > 0 else None
    return None


def _extract_table_dims(visual: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    row_paths = [
        ["rows"],
        ["n_rows"],
        ["row_count"],
        ["table", "rows"],
        ["table", "n_rows"],
        ["table", "row_count"],
        ["meta", "rows"],
        ["meta", "n_rows"],
        ["meta", "row_count"],
        ["meta", "table", "rows"],
        ["meta", "table", "n_rows"],
        ["meta", "table", "row_count"],
    ]
    col_paths = [
        ["cols"],
        ["n_cols"],
        ["col_count"],
        ["columns"],
        ["table", "cols"],
        ["table", "n_cols"],
        ["table", "col_count"],
        ["table", "columns"],
        ["meta", "cols"],
        ["meta", "n_cols"],
        ["meta", "col_count"],
        ["meta", "columns"],
        ["meta", "table", "cols"],
        ["meta", "table", "n_cols"],
        ["meta", "table", "col_count"],
        ["meta", "table", "columns"],
    ]
    rows = next((_first_int(_find_path(visual, p)) for p in row_paths if _first_int(_find_path(visual, p)) is not None), None)
    cols = next((_first_int(_find_path(visual, p)) for p in col_paths if _first_int(_find_path(visual, p)) is not None), None)
    return rows, cols


def _flatten_strings(node: Any) -> List[str]:
    out: List[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for v in cur.values():
                stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                stack.append(v)
        elif isinstance(cur, str):
            txt = cur.strip()
            if txt:
                out.append(txt.lower())
    return out


def _is_fallback_visual(visual: Dict[str, Any]) -> bool:
    for key in ("is_fallback", "fallback", "used_fallback_detector"):
        if key in visual and _bool_trueish(visual.get(key)):
            return True
    haystack = " ".join(_flatten_strings(visual))
    return "pdf_fallback_connected_components" in haystack or "fallback_connected_components" in haystack


def _ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _build_quality_report(audit: Dict[str, Any], args: argparse.Namespace, audit_path: Path) -> Dict[str, Any]:
    slides = audit.get("slides") if isinstance(audit.get("slides"), list) else []
    request = audit.get("request") if isinstance(audit.get("request"), dict) else {}
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    env_profile = request.get("env_profile") if isinstance(request.get("env_profile"), dict) else {}

    deck_tokens: Set[str] = set()
    query_tokens = _tokenize(str(request.get("query") or ""))
    outline_tokens = _tokenize(str(request.get("outline_title") or ""))
    reference_tokens = query_tokens or outline_tokens

    evidence_per_bullet: List[int] = []
    bullet_lengths: List[int] = []
    bullet_token_lengths: List[int] = []
    bullets_per_slide: List[int] = []
    slide_doc_counts: List[Tuple[int, int]] = []
    table_shapes: List[Dict[str, Any]] = []

    unique_docs_deck: Set[str] = set()
    total_bullets = 0
    total_evidence = 0
    evidence_with_page = 0
    evidence_with_complete_locator = 0
    evidence_page_only_locator = 0
    single_evidence_bullets = 0
    bullets_missing_evidence = 0
    slides_over_bullet_cap = 0
    long_bullets = 0
    long_bullet_slides: Set[int] = set()
    high_doc_diversity_slides: List[int] = []
    off_topic_slides: List[int] = []

    numeric_bullets = 0
    numeric_bullets_linked_table = 0
    evidence_role_slides = 0
    evidence_role_unlinked_slides = 0

    total_visual_assets = 0
    fallback_visual_assets = 0
    missing_assets_runtime = 0
    dense_tables = 0

    for slide_idx, raw_slide in enumerate(slides, start=1):
        slide = raw_slide if isinstance(raw_slide, dict) else {}
        title = str(slide.get("title") or "").strip()
        slide_tokens = _tokenize(title)
        slide_docs: Set[str] = set()
        bullets = slide.get("bullets") if isinstance(slide.get("bullets"), list) else []
        visuals = slide.get("visuals") if isinstance(slide.get("visuals"), list) else []

        bullets_per_slide.append(len(bullets))
        if len(bullets) > args.warn_bullets_per_slide_cap:
            slides_over_bullet_cap += 1

        visual_type_by_asset: Dict[str, str] = {}
        for visual in visuals:
            if not isinstance(visual, dict):
                continue
            total_visual_assets += 1
            asset_id = str(visual.get("asset_id") or "").strip()
            vtype = str(visual.get("type") or "").strip().lower()
            if asset_id:
                visual_type_by_asset[asset_id] = vtype
            doc_id = str(visual.get("doc_id") or "").strip()
            if doc_id:
                slide_docs.add(doc_id)
                unique_docs_deck.add(doc_id)
            if not bool(visual.get("exists")):
                missing_assets_runtime += 1
            if _is_fallback_visual(visual):
                fallback_visual_assets += 1
            if vtype == "table":
                rows, cols = _extract_table_dims(visual)
                if rows is not None and cols is not None:
                    cells = rows * cols
                    if cells >= args.warn_table_dense_cells:
                        dense_tables += 1
                    table_shapes.append(
                        {
                            "slide_index": slide_idx,
                            "asset_id": asset_id or None,
                            "rows": rows,
                            "cols": cols,
                            "cells": cells,
                        }
                    )

        any_bullet_visual_link = False
        for bullet_idx, raw_bullet in enumerate(bullets, start=1):
            bullet = raw_bullet if isinstance(raw_bullet, dict) else {}
            text = str(bullet.get("text") or "").strip()
            ev_items = bullet.get("evidence_items") if isinstance(bullet.get("evidence_items"), list) else []
            visual_ids = [str(v).strip() for v in (bullet.get("visual_asset_ids") or []) if str(v).strip()]
            if visual_ids:
                any_bullet_visual_link = True

            total_bullets += 1
            ev_count = len(ev_items)
            evidence_per_bullet.append(ev_count)
            if ev_count == 1:
                single_evidence_bullets += 1
            if ev_count < args.hard_min_evidence_per_bullet:
                bullets_missing_evidence += 1

            bullet_lengths.append(len(text))
            bullet_tokens = _tokenize(text)
            bullet_token_lengths.append(len(bullet_tokens))
            slide_tokens.update(bullet_tokens)
            if len(text) > args.warn_long_bullet_chars:
                long_bullets += 1
                long_bullet_slides.add(slide_idx)

            if _contains_numeric(text):
                numeric_bullets += 1
                if any(visual_type_by_asset.get(v_id) == "table" for v_id in visual_ids):
                    numeric_bullets_linked_table += 1

            for ev in ev_items:
                if not isinstance(ev, dict):
                    continue
                total_evidence += 1
                doc_id = str(ev.get("doc_id") or "").strip()
                if doc_id:
                    slide_docs.add(doc_id)
                    unique_docs_deck.add(doc_id)
                has_page, complete, page_only = _locator_flags(ev.get("locator"))
                if has_page:
                    evidence_with_page += 1
                if complete:
                    evidence_with_complete_locator += 1
                if page_only:
                    evidence_page_only_locator += 1

        if reference_tokens:
            overlap = len(reference_tokens & slide_tokens) / len(reference_tokens)
            if overlap < args.warn_min_slide_query_overlap:
                off_topic_slides.append(slide_idx)

        deck_tokens.update(slide_tokens)
        slide_doc_counts.append((slide_idx, len(slide_docs)))
        if len(slide_docs) > args.warn_max_docs_per_slide:
            high_doc_diversity_slides.append(slide_idx)

        role = str(slide.get("visual_role") or "").strip().lower()
        if role == "evidence":
            evidence_role_slides += 1
            if not any_bullet_visual_link:
                evidence_role_unlinked_slides += 1

    summary_missing_assets = int(summary.get("missing_asset_count", 0) or 0)
    strict_missing_limit = args.hard_max_missing_assets
    if strict_missing_limit < 0 and args.strict_missing_assets:
        strict_missing_limit = 0

    query_coverage = _ratio(len(query_tokens & deck_tokens), len(query_tokens))
    outline_coverage = _ratio(len(outline_tokens & deck_tokens), len(outline_tokens))

    min_evidence = min(evidence_per_bullet) if evidence_per_bullet else 0
    max_evidence = max(evidence_per_bullet) if evidence_per_bullet else 0
    mean_evidence = mean(evidence_per_bullet) if evidence_per_bullet else 0.0

    min_bullets_slide = min(bullets_per_slide) if bullets_per_slide else 0
    max_bullets_slide = max(bullets_per_slide) if bullets_per_slide else 0
    mean_bullets_slide = mean(bullets_per_slide) if bullets_per_slide else 0.0

    mean_bullet_chars = mean(bullet_lengths) if bullet_lengths else 0.0
    max_bullet_chars = max(bullet_lengths) if bullet_lengths else 0
    mean_bullet_tokens = mean(bullet_token_lengths) if bullet_token_lengths else 0.0
    max_bullet_tokens = max(bullet_token_lengths) if bullet_token_lengths else 0

    single_evidence_ratio = _ratio(single_evidence_bullets, total_bullets)
    fallback_ratio = _ratio(fallback_visual_assets, total_visual_assets)
    complete_locator_ratio = _ratio(evidence_with_complete_locator, total_evidence)
    page_only_locator_ratio = _ratio(evidence_page_only_locator, evidence_with_page)
    numeric_table_link_ratio = _ratio(numeric_bullets_linked_table, numeric_bullets)
    evidence_role_unlinked_ratio = _ratio(evidence_role_unlinked_slides, evidence_role_slides)

    problematic_slide_count = len(set(high_doc_diversity_slides) | set(off_topic_slides) | set(long_bullet_slides))

    warnings: List[str] = []
    if single_evidence_ratio is not None and single_evidence_ratio > args.warn_max_single_evidence_ratio:
        warnings.append(
            f"thin evidence risk: single-evidence bullets {_fmt_ratio(single_evidence_ratio)} > {_fmt_ratio(args.warn_max_single_evidence_ratio)}"
        )
    if fallback_ratio is not None and fallback_ratio > args.warn_max_fallback_asset_ratio:
        warnings.append(
            f"fallback visual ratio {_fmt_ratio(fallback_ratio)} > {_fmt_ratio(args.warn_max_fallback_asset_ratio)}"
        )
    if page_only_locator_ratio is not None and page_only_locator_ratio > args.warn_max_page_only_locator_ratio:
        warnings.append(
            f"page-only locator ratio {_fmt_ratio(page_only_locator_ratio)} > {_fmt_ratio(args.warn_max_page_only_locator_ratio)}"
        )
    if slides_over_bullet_cap > args.warn_max_slides_over_bullets_cap:
        warnings.append(
            f"slides over bullet cap: {slides_over_bullet_cap} > {args.warn_max_slides_over_bullets_cap} (cap={args.warn_bullets_per_slide_cap})"
        )
    if long_bullets > args.warn_max_long_bullets:
        warnings.append(f"long bullets: {long_bullets} > {args.warn_max_long_bullets} (chars>{args.warn_long_bullet_chars})")
    if evidence_role_unlinked_slides > 0:
        warnings.append(
            f"visual_role=evidence without bullet links: {evidence_role_unlinked_slides}/{evidence_role_slides}"
        )
    if high_doc_diversity_slides:
        warnings.append(
            "high source diversity per slide: " + ",".join(str(x) for x in sorted(high_doc_diversity_slides))
        )
    if dense_tables > 0:
        warnings.append(f"dense tables detected: {dense_tables} (cells>={args.warn_table_dense_cells})")
    if off_topic_slides:
        warnings.append(
            "low query/outline overlap slides: " + ",".join(str(x) for x in sorted(off_topic_slides))
        )
    if problematic_slide_count >= args.warn_max_problematic_slides:
        warnings.append(
            f"problematic slide count {problematic_slide_count} >= {args.warn_max_problematic_slides}"
        )

    hard_failures: List[str] = []
    if min_evidence < args.hard_min_evidence_per_bullet:
        hard_failures.append(
            f"min evidence per bullet {min_evidence} < hard threshold {args.hard_min_evidence_per_bullet}"
        )
    if strict_missing_limit >= 0 and summary_missing_assets > strict_missing_limit:
        hard_failures.append(
            f"missing_asset_count {summary_missing_assets} > hard threshold {strict_missing_limit}"
        )

    quality = {
        "audit_path": str(audit_path),
        "deck_id": str(audit.get("deck_id") or "").strip() or None,
        "request": {
            "query": str(request.get("query") or "").strip() or None,
            "outline_title": str(request.get("outline_title") or "").strip() or None,
            "specialty": str(request.get("specialty") or "").strip() or None,
            "version": str(request.get("version") or "").strip() or None,
            "mode": str(request.get("mode") or "").strip() or None,
            "env_profile": {k: bool(v) for k, v in sorted(env_profile.items())},
        },
        "counts": {
            "slides": len(slides),
            "bullets": total_bullets,
            "evidence_items": total_evidence,
            "visual_assets": total_visual_assets,
            "fallback_visual_assets": fallback_visual_assets,
            "missing_asset_count_summary": summary_missing_assets,
            "missing_asset_count_runtime": missing_assets_runtime,
        },
        "metrics": {
            "evidence_density": {
                "per_bullet_mean": round(mean_evidence, 4),
                "per_bullet_min": min_evidence,
                "per_bullet_max": max_evidence,
                "single_evidence_bullets_ratio": single_evidence_ratio,
            },
            "source_diversity": {
                "unique_docs_deck": len(unique_docs_deck),
                "unique_docs_per_slide": [{"slide_index": idx, "doc_count": count} for idx, count in slide_doc_counts],
            },
            "locator_quality": {
                "with_page_ratio": _ratio(evidence_with_page, total_evidence),
                "complete_locator_ratio": complete_locator_ratio,
                "page_only_locator_ratio": page_only_locator_ratio,
            },
            "visual_coherence": {
                "evidence_role_slides": evidence_role_slides,
                "evidence_role_unlinked_slides": evidence_role_unlinked_slides,
                "evidence_role_unlinked_ratio": evidence_role_unlinked_ratio,
                "numeric_bullets": numeric_bullets,
                "numeric_bullets_linked_to_table": numeric_bullets_linked_table,
                "numeric_table_link_ratio": numeric_table_link_ratio,
            },
            "readability": {
                "bullets_per_slide_mean": round(mean_bullets_slide, 4),
                "bullets_per_slide_min": min_bullets_slide,
                "bullets_per_slide_max": max_bullets_slide,
                "slides_over_bullets_cap": slides_over_bullet_cap,
                "bullet_chars_mean": round(mean_bullet_chars, 2),
                "bullet_chars_max": max_bullet_chars,
                "bullet_tokens_mean": round(mean_bullet_tokens, 2),
                "bullet_tokens_max": max_bullet_tokens,
                "long_bullets": long_bullets,
                "table_shapes": table_shapes,
                "dense_tables": dense_tables,
            },
            "fallback_rate": {
                "fallback_assets_ratio": fallback_ratio,
                "fallback_assets_count": fallback_visual_assets,
                "visual_assets_count": total_visual_assets,
            },
            "coverage": {
                "query_token_coverage_ratio": query_coverage,
                "outline_token_coverage_ratio": outline_coverage,
                "off_topic_slides": sorted(set(off_topic_slides)),
                "reference_token_count": len(reference_tokens),
            },
        },
        "thresholds": {
            "hard_min_evidence_per_bullet": args.hard_min_evidence_per_bullet,
            "hard_max_missing_assets": strict_missing_limit,
            "warn_max_single_evidence_ratio": args.warn_max_single_evidence_ratio,
            "warn_max_fallback_asset_ratio": args.warn_max_fallback_asset_ratio,
            "warn_max_page_only_locator_ratio": args.warn_max_page_only_locator_ratio,
            "warn_max_docs_per_slide": args.warn_max_docs_per_slide,
            "warn_min_slide_query_overlap": args.warn_min_slide_query_overlap,
            "warn_bullets_per_slide_cap": args.warn_bullets_per_slide_cap,
            "warn_max_slides_over_bullets_cap": args.warn_max_slides_over_bullets_cap,
            "warn_long_bullet_chars": args.warn_long_bullet_chars,
            "warn_max_long_bullets": args.warn_max_long_bullets,
            "warn_table_dense_cells": args.warn_table_dense_cells,
            "warn_max_problematic_slides": args.warn_max_problematic_slides,
        },
        "warnings": warnings,
        "gate": {
            "hard_failures": hard_failures,
            "warning_count": len(warnings),
            "pass_hard": len(hard_failures) == 0,
            "pass_with_warnings": len(hard_failures) == 0 and len(warnings) == 0,
        },
    }
    return quality


def _quality_txt(quality: Dict[str, Any]) -> str:
    counts = quality.get("counts") if isinstance(quality.get("counts"), dict) else {}
    metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    gate = quality.get("gate") if isinstance(quality.get("gate"), dict) else {}
    warnings = quality.get("warnings") if isinstance(quality.get("warnings"), list) else []
    evidence = metrics.get("evidence_density") if isinstance(metrics.get("evidence_density"), dict) else {}
    locator = metrics.get("locator_quality") if isinstance(metrics.get("locator_quality"), dict) else {}
    visual = metrics.get("visual_coherence") if isinstance(metrics.get("visual_coherence"), dict) else {}
    read = metrics.get("readability") if isinstance(metrics.get("readability"), dict) else {}
    fallback = metrics.get("fallback_rate") if isinstance(metrics.get("fallback_rate"), dict) else {}
    coverage = metrics.get("coverage") if isinstance(metrics.get("coverage"), dict) else {}
    source = metrics.get("source_diversity") if isinstance(metrics.get("source_diversity"), dict) else {}

    lines = [
        "Clinical Quality Evaluation",
        f"deck_id: {quality.get('deck_id') or 'n/a'}",
        f"audit: {quality.get('audit_path') or 'n/a'}",
        "",
        "Counts",
        f"- slides: {counts.get('slides', 0)}",
        f"- bullets: {counts.get('bullets', 0)}",
        f"- evidence_items: {counts.get('evidence_items', 0)}",
        f"- visual_assets: {counts.get('visual_assets', 0)}",
        f"- fallback_visual_assets: {counts.get('fallback_visual_assets', 0)}",
        f"- missing_asset_count(summary/runtime): {counts.get('missing_asset_count_summary', 0)}/{counts.get('missing_asset_count_runtime', 0)}",
        "",
        "Evidence",
        f"- evidence per bullet (mean/min/max): {evidence.get('per_bullet_mean', 0)}/{evidence.get('per_bullet_min', 0)}/{evidence.get('per_bullet_max', 0)}",
        f"- single-evidence bullets: {_fmt_ratio(evidence.get('single_evidence_bullets_ratio'))}",
        "",
        "Locator quality",
        f"- locator with page: {_fmt_ratio(locator.get('with_page_ratio'))}",
        f"- complete locator: {_fmt_ratio(locator.get('complete_locator_ratio'))}",
        f"- page-only locator: {_fmt_ratio(locator.get('page_only_locator_ratio'))}",
        "",
        "Visual coherence",
        f"- evidence-role slides (unlinked): {visual.get('evidence_role_slides', 0)} ({visual.get('evidence_role_unlinked_slides', 0)} unlinked)",
        f"- numeric bullets linked to table: {_fmt_ratio(visual.get('numeric_table_link_ratio'))} ({visual.get('numeric_bullets_linked_to_table', 0)}/{visual.get('numeric_bullets', 0)})",
        "",
        "Readability",
        f"- bullets per slide (mean/min/max): {read.get('bullets_per_slide_mean', 0)}/{read.get('bullets_per_slide_min', 0)}/{read.get('bullets_per_slide_max', 0)}",
        f"- slides over bullet cap: {read.get('slides_over_bullets_cap', 0)}",
        f"- long bullets: {read.get('long_bullets', 0)}",
        f"- dense tables: {read.get('dense_tables', 0)}",
        "",
        "Source diversity",
        f"- unique docs in deck: {source.get('unique_docs_deck', 0)}",
        "",
        "Coverage",
        f"- query token coverage: {_fmt_ratio(coverage.get('query_token_coverage_ratio'))}",
        f"- outline token coverage: {_fmt_ratio(coverage.get('outline_token_coverage_ratio'))}",
        f"- off-topic slides: {','.join(str(x) for x in coverage.get('off_topic_slides', [])) or 'none'}",
        "",
        "Fallback",
        f"- fallback asset ratio: {_fmt_ratio(fallback.get('fallback_assets_ratio'))}",
        "",
        "Gate",
        f"- pass_hard: {bool(gate.get('pass_hard'))}",
        f"- warning_count: {gate.get('warning_count', 0)}",
    ]
    hard = gate.get("hard_failures") if isinstance(gate.get("hard_failures"), list) else []
    if hard:
        lines.append("- hard_failures:")
        lines.extend([f"  - {h}" for h in hard])
    if warnings:
        lines.append("- warnings:")
        lines.extend([f"  - {w}" for w in warnings])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute deterministic clinical quality metrics from deck.audit.json")
    ap.add_argument("--audit-json", required=True, help="Path to input *.audit.json")
    ap.add_argument("--out-dir", default="", help="Output directory for quality.{json,txt}")
    ap.add_argument("--out-root", default="out/tests", help="Used with --test-name when --out-dir is not provided")
    ap.add_argument("--test-name", default="", help="Target folder name under --out-root")
    ap.add_argument("--fail-on-hard", action="store_true", help="Exit 1 on hard gate failures")

    ap.add_argument("--hard-min-evidence-per-bullet", type=int, default=1)
    ap.add_argument("--hard-max-missing-assets", type=int, default=-1, help="Set -1 to disable this hard gate")
    ap.add_argument("--strict-missing-assets", action="store_true", help="Shortcut for --hard-max-missing-assets=0")

    ap.add_argument("--warn-max-single-evidence-ratio", type=float, default=0.30)
    ap.add_argument("--warn-max-fallback-asset-ratio", type=float, default=0.20)
    ap.add_argument("--warn-max-page-only-locator-ratio", type=float, default=0.70)
    ap.add_argument("--warn-max-docs-per-slide", type=int, default=3)
    ap.add_argument("--warn-min-slide-query-overlap", type=float, default=0.20)
    ap.add_argument("--warn-bullets-per-slide-cap", type=int, default=7)
    ap.add_argument("--warn-max-slides-over-bullets-cap", type=int, default=2)
    ap.add_argument("--warn-long-bullet-chars", type=int, default=140)
    ap.add_argument("--warn-max-long-bullets", type=int, default=2)
    ap.add_argument("--warn-table-dense-cells", type=int, default=60)
    ap.add_argument("--warn-max-problematic-slides", type=int, default=2)
    args = ap.parse_args()

    audit_path = Path(args.audit_json)
    audit = _load_json(audit_path)
    if not args.out_dir and not args.test_name:
        out_dir = audit_path.parent
    elif args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.out_root) / _safe_name(args.test_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    quality = _build_quality_report(audit, args, audit_path)
    quality_json_path = out_dir / "quality.json"
    quality_txt_path = out_dir / "quality.txt"
    quality_json_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    quality_txt_path.write_text(_quality_txt(quality), encoding="utf-8")

    hard_failures = quality.get("gate", {}).get("hard_failures", [])
    if isinstance(hard_failures, list) and hard_failures:
        print("HARD_FAIL")
        for item in hard_failures:
            print(f"- {item}")
        print(f"quality_json={quality_json_path}")
        print(f"quality_txt={quality_txt_path}")
        if args.fail_on_hard:
            raise SystemExit(1)
    else:
        print("PASS")
        print(f"quality_json={quality_json_path}")
        print(f"quality_txt={quality_txt_path}")


if __name__ == "__main__":
    main()
