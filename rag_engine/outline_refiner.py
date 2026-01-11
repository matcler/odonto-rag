from __future__ import annotations

import json
from typing import Callable, Dict, Any, List

from rag_engine.clinical_outline import extract_object, CLINICAL_OUTLINE_SCHEMA, build_json_repair_prompt


def _tp_count(outline: Dict[str, Any]) -> int:
    tps = outline.get("teaching_points", [])
    return len(tps) if isinstance(tps, list) else 0


def build_refine_prompt(outline: Dict[str, Any]) -> str:
    outline_json = json.dumps(outline, ensure_ascii=False, indent=2)

    return f"""
Return ONLY STRICT JSON (double quotes, no trailing commas). No extra text.

You are a senior dental clinician and educator.

TASK:
Refine the clinical outline below to be slide-ready.

HARD CONSTRAINTS (MUST):
- Keep the SAME number of teaching_points and the SAME order.
- Do NOT merge teaching points.
- Do NOT remove teaching points.
- Titles: decision-oriented, clinical, short.
- Each teaching point must have:
  - key_takeaways: 2–4
  - common_pitfalls: 1–3
  - what_to_do: 2–4
- Replace vague filler with chairside actions.

SCHEMA:
{CLINICAL_OUTLINE_SCHEMA}

INPUT OUTLINE JSON:
{outline_json}
""".strip()


def build_refine_regenerate_prompt(outline: Dict[str, Any]) -> str:
    outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
    return f"""
Return ONLY STRICT JSON (double quotes, no trailing commas). No extra text.

You MUST rewrite the outline below into a refined outline that matches the schema.

ABSOLUTE RULE:
- teaching_points count MUST be EXACTLY { _tp_count(outline) } and same order.

Targets:
- Titles: clinical and decision-oriented (avoid "Understand/Master/Optimize").
- key_takeaways: 2–4 short items
- common_pitfalls: 1–3 short items
- what_to_do: 2–4 actionable steps
- No filler.

SCHEMA:
{CLINICAL_OUTLINE_SCHEMA}

INPUT OUTLINE:
{outline_json}
""".strip()


def _validate_same_count(original: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    return _tp_count(original) > 0 and _tp_count(candidate) == _tp_count(original)


def _fallback_original(original: Dict[str, Any]) -> Dict[str, Any]:
    """
    Last resort: return the original outline (ensures slides generation continues).
    """
    return original


def refine_clinical_outline(
    *,
    outline: Dict[str, Any],
    llm_generate_text: Callable[[str, float, int], str],
) -> Dict[str, Any]:
    if _tp_count(outline) == 0:
        return outline

    # Attempt 1
    raw = llm_generate_text(build_refine_prompt(outline), 0.15, 2400)
    try:
        refined = extract_object(raw)
        if _validate_same_count(outline, refined):
            return refined
    except Exception:
        pass

    # Attempt 2: repair (works if JSON is "almost" correct)
    try:
        fixed = llm_generate_text(build_json_repair_prompt(raw), 0.0, 2600)
        refined2 = extract_object(fixed)
        if _validate_same_count(outline, refined2):
            return refined2
    except Exception:
        pass

    # Attempt 3: regenerate from original with explicit count
    try:
        raw2 = llm_generate_text(build_refine_regenerate_prompt(outline), 0.10, 2600)
        refined3 = extract_object(raw2)
        if _validate_same_count(outline, refined3):
            return refined3
    except Exception:
        pass

    # Last resort: do NOT block the pipeline
    return _fallback_original(outline)
