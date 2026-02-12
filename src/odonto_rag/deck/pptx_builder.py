from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or "rag_slides"


def _truncate_footer_label(text: str, limit: int = 200) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def build_pptx_from_slide_plan(
    slide_plan: Dict[str, Any],
    *,
    out_dir: str = "out",
    filename: Optional[str] = None,
) -> Path:
    slides = slide_plan.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("slide_plan['slides'] must be a non-empty list")

    raw_title = str(slide_plan.get("title") or "").strip()
    if raw_title:
        deck_title = raw_title
    else:
        outline_used = slide_plan.get("outline_used")
        if isinstance(outline_used, dict):
            outline_title = str(outline_used.get("title") or "").strip()
            deck_title = outline_title or "RAG Slides"
        else:
            deck_title = "RAG Slides"

    if filename is not None:
        generated_name = filename
        if not generated_name.lower().endswith(".pptx"):
            generated_name += ".pptx"
    else:
        payload = json.dumps(slide_plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        sha = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        generated_name = f"{_slugify(deck_title)}_{sha}.pptx"

    out_path = Path(out_dir) / generated_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    for idx, raw_slide in enumerate(slides, start=1):
        slide_data = raw_slide if isinstance(raw_slide, dict) else {}
        title = str(slide_data.get("title") or f"Slide {idx}").strip()
        bullets_raw = slide_data.get("bullets")
        bullets = [str(b).strip() for b in bullets_raw if str(b).strip()] if isinstance(bullets_raw, list) else []
        citations_raw = slide_data.get("citations")
        citations = [str(c).strip() for c in citations_raw if str(c).strip()] if isinstance(citations_raw, list) else []
        sources_raw = slide_data.get("sources")
        sources = [s for s in sources_raw if isinstance(s, dict)] if isinstance(sources_raw, list) else []

        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title

        body_shape = slide.placeholders[1]
        text_frame = body_shape.text_frame
        text_frame.clear()
        for bullet_idx, bullet in enumerate(bullets):
            paragraph = text_frame.paragraphs[0] if bullet_idx == 0 else text_frame.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0

        footer_lines: list[str] = []
        if citations:
            src_by_ref = {}
            for src in sources:
                ref_id = str(src.get("ref_id") or "").strip()
                if ref_id and ref_id not in src_by_ref:
                    src_by_ref[ref_id] = src

            grouped: Dict[str, Dict[str, Any]] = {}
            group_order: list[str] = []
            for c in citations:
                src = src_by_ref.get(c)
                if not src:
                    continue
                item_id = str(src.get("item_id") or "").strip()
                doc_id = str(src.get("doc_id") or "").strip()
                doc_citation = str(src.get("doc_citation") or "").strip()
                doc_filename = str(src.get("doc_filename") or "").strip()
                label = _truncate_footer_label(doc_citation or doc_filename or item_id or c)
                group_key = doc_id or f"fallback::{label}"
                if group_key not in grouped:
                    grouped[group_key] = {"label": label, "tokens": []}
                    group_order.append(group_key)
                if c not in grouped[group_key]["tokens"]:
                    grouped[group_key]["tokens"].append(c)

            footer_lines = [
                f"{grouped[key]['label']} ({', '.join(grouped[key]['tokens'])})"
                for key in group_order
                if grouped[key]["tokens"]
            ]

        if footer_lines:
            slide_width_in = float(prs.slide_width) / 914400.0
            slide_height_in = float(prs.slide_height) / 914400.0
            footer_height = min(1.2, max(0.3, 0.16 * len(footer_lines)))
            footer_box = slide.shapes.add_textbox(
                Inches(0.5),
                Inches(max(0.1, slide_height_in - footer_height - 0.1)),
                Inches(max(1.0, slide_width_in - 1.0)),
                Inches(footer_height),
            )
            footer_frame = footer_box.text_frame
            footer_frame.clear()
            footer_p = footer_frame.paragraphs[0]
            footer_p.text = "\n".join(footer_lines)
            footer_p.alignment = PP_ALIGN.LEFT
            for run in footer_p.runs:
                run.font.size = Pt(8)

        notes_parts = []
        speaker_notes = slide_data.get("speaker_notes")
        if speaker_notes is not None:
            notes_text = str(speaker_notes).strip()
            if notes_text:
                notes_parts.append(notes_text)

        if citations:
            src_by_ref = {}
            for src in sources:
                ref_id = str(src.get("ref_id") or "").strip()
                if ref_id and ref_id not in src_by_ref:
                    src_by_ref[ref_id] = src

            source_lines = ["Sources:"]
            for c in citations:
                src = src_by_ref.get(c)
                if not src:
                    source_lines.append(f"{c} (missing source)")
                    continue
                item_id = str(src.get("item_id") or "").strip()
                doc_citation = str(src.get("doc_citation") or "").strip()
                doc_filename = str(src.get("doc_filename") or "").strip()
                page_start = int(src.get("page_start", 0))
                page_end = int(src.get("page_end", 0))
                if page_start == page_end:
                    page_text = f"p. {page_start}"
                else:
                    page_text = f"p. {page_start}-{page_end}"
                label = doc_citation or doc_filename or item_id or c
                source_lines.append(f"{label} ({c} {page_text})")
            notes_parts.append("\n".join(source_lines))

        if notes_parts:
            slide.notes_slide.notes_text_frame.text = "\n\n".join(notes_parts)

    prs.save(str(out_path))
    return out_path
