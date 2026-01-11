from __future__ import annotations

import ast
import json
from typing import Callable, Dict, Any, List


CLINICAL_OUTLINE_SCHEMA = """
{
  "topic": "string",
  "audience": "string",
  "teaching_points": [
    {
      "title": "string",
      "key_takeaways": ["string"],
      "common_pitfalls": ["string"],
      "what_to_do": ["string"]
    }
  ]
}
""".strip()


def _extract_first_object_best_effort(raw: str) -> str:
    raw = raw or ""
    start = raw.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")
    return raw[start:]


def extract_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    # 1) strict JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) candidate substring (best-effort)
    candidate = _extract_first_object_best_effort(raw).strip()

    # 2a) raw_decode JSON
    try:
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2b) python literal (single quotes) - may still fail on truncated strings
    try:
        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        raise ValueError(
            "Could not parse model output as JSON/Python dict. "
            f"Head: {candidate[:180]!r}"
        ) from e

    raise ValueError("Parsed value is not a dict.")


def build_json_repair_prompt(bad_output: str) -> str:
    return f"""
Return ONLY a valid STRICT JSON object. No markdown, no extra text.
Double quotes for keys/strings. No trailing commas.
Must be parseable by Python json.loads().

SCHEMA:
{CLINICAL_OUTLINE_SCHEMA}

INPUT TO CONVERT:
{bad_output}
""".strip()


def _clean_list(items, max_items: int = 6) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for it in items:
        s = str(it).replace("\n", " ").strip()
        if s:
            out.append(s)
    return out[:max_items]


def _normalize_outline(data: Dict[str, Any], topic: str, audience: str) -> Dict[str, Any]:
    data["topic"] = (data.get("topic") or topic).strip()
    data["audience"] = (data.get("audience") or audience).strip()

    cleaned = []
    for tp in data.get("teaching_points", []) or []:
        if not isinstance(tp, dict):
            continue
        title = str(tp.get("title") or "").strip()
        if not title:
            continue
        cleaned.append({
            "title": title[:80],
            "key_takeaways": _clean_list(tp.get("key_takeaways")),
            "common_pitfalls": _clean_list(tp.get("common_pitfalls")),
            "what_to_do": _clean_list(tp.get("what_to_do")),
        })

    data["teaching_points"] = cleaned
    return data


def _fallback_outline(topic: str, audience: str) -> Dict[str, Any]:
    """
    Deterministic safe outline (8 teaching points).
    Used ONLY when LLM output is irrecoverably malformed.
    """
    tps = [
        {
            "title": "Choose System: E&R vs Self-Etch vs Universal",
            "key_takeaways": ["Selection depends on substrate and margins", "Universal is mode-dependent", "Enamel often benefits from phosphoric acid"],
            "common_pitfalls": ["Using one protocol for all situations"],
            "what_to_do": ["Decide mode before starting", "Prefer selective enamel etch when indicated", "Keep protocols separated by system"],
        },
        {
            "title": "Etch-and-Rinse: Control Etch and Rinse",
            "key_takeaways": ["High enamel bond strength", "Technique sensitive on dentin", "Rinse thoroughly"],
            "common_pitfalls": ["Over-etching dentin", "Incomplete rinsing"],
            "what_to_do": ["Respect etch time", "Rinse completely", "Proceed without contamination"],
        },
        {
            "title": "E&R: Manage Dentin Moisture (Wet Bonding)",
            "key_takeaways": ["Dentin should be moist, not wet or dry", "Collagen collapse reduces infiltration"],
            "common_pitfalls": ["Over-drying after rinse", "Pooling water dilutes resin"],
            "what_to_do": ["Blot dry to ‘glistening’ dentin", "Avoid strong air blast", "Apply primer/adhesive promptly"],
        },
        {
            "title": "Self-Etch: When Simplicity Costs Enamel Bond",
            "key_takeaways": ["Reduced sensitivity", "Often weaker on uncut enamel"],
            "common_pitfalls": ["Skipping selective enamel etch", "Too short application time"],
            "what_to_do": ["Selective etch enamel margins", "Scrub for recommended time", "Follow dwell time strictly"],
        },
        {
            "title": "Isolation and Contamination Control",
            "key_takeaways": ["Saliva/blood kills bond strength", "Timing matters most after prep/etch"],
            "common_pitfalls": ["No rubber dam where needed", "Touching etched enamel"],
            "what_to_do": ["Use rubber dam when possible", "Re-etch/re-prime after contamination", "Control gingival fluid aggressively"],
        },
        {
            "title": "Solvent Evaporation and Film Thickness",
            "key_takeaways": ["Air-thinning drives solvent off", "Thin uniform layer improves fit"],
            "common_pitfalls": ["Pooling adhesive", "Insufficient air-drying"],
            "what_to_do": ["Air-thin 5–10s as per IFU", "Scrub then thin", "Avoid thick glossy puddles"],
        },
        {
            "title": "Light Curing: Energy Delivery and Placement",
            "key_takeaways": ["Intensity + time matter", "Distance/angle reduce energy"],
            "common_pitfalls": ["Under-curing", "Dirty/light tip degradation"],
            "what_to_do": ["Check light output periodically", "Keep tip close and perpendicular", "Cure for IFU time (longer if distance)"],
        },
        {
            "title": "Troubleshooting Failures and Sensitivity",
            "key_takeaways": ["Failures usually technique/contamination", "Sensitivity often from dentin desiccation/poor seal"],
            "common_pitfalls": ["Repeating same steps without root-cause review"],
            "what_to_do": ["Audit isolation and moisture control", "Verify curing and air-thinning", "Consider selective-etch/universal mode change"],
        },
    ]
    return {"topic": topic, "audience": audience, "teaching_points": tps}


def build_clinical_outline_prompt(*, topic: str, audience: str, context: str) -> str:
    return f"""
Return ONLY STRICT JSON (double quotes, no trailing commas). No extra text.

You are a senior dental clinician and educator.

TASK:
From the provided SOURCES, create a clinical teaching outline.

HARD REQUIREMENTS:
- teaching_points MUST be EXACTLY 8 items.
- Titles must be unique and decision-oriented.
- Each teaching point must have:
  - key_takeaways: 2–4 items
  - common_pitfalls: 1–3 items
  - what_to_do: 2–4 items
- Use short chairside phrases.
- No citations.

SCHEMA:
{CLINICAL_OUTLINE_SCHEMA}

TOPIC: {topic}
AUDIENCE: {audience}

SOURCES:
{context}
""".strip()


def generate_clinical_outline(
    *,
    topic: str,
    audience: str,
    context: str,
    llm_generate_text: Callable[[str, float, int], str],
) -> Dict[str, Any]:
    prompt = build_clinical_outline_prompt(topic=topic, audience=audience, context=context)

    raw = llm_generate_text(prompt, 0.10, 1800)

    # Attempt 1: parse
    try:
        data = _normalize_outline(extract_object(raw), topic, audience)
        if isinstance(data.get("teaching_points"), list) and len(data["teaching_points"]) >= 6:
            return data
    except Exception:
        pass

    # Attempt 2: repair (one shot)
    try:
        fixed = llm_generate_text(build_json_repair_prompt(raw), 0.0, 2200)
        data2 = _normalize_outline(extract_object(fixed), topic, audience)
        if isinstance(data2.get("teaching_points"), list) and len(data2["teaching_points"]) >= 6:
            return data2
    except Exception:
        pass

    # Last resort: deterministic fallback (keeps pipeline alive)
    return _fallback_outline(topic, audience)
