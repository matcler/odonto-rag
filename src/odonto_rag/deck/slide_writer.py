from __future__ import annotations

import json
import re
from typing import Callable, Dict, Any, List, Set


SLIDE_DECK_SCHEMA = """
{
  "topic": "string",
  "audience": "string",
  "slides": [
    {
      "title": "string",
      "bullets": ["string"]
    }
  ]
}
""".strip()

_TAGS = ("Do:", "Avoid:", "Check:", "Rescue:", "Tip:")

_TRAILING_STOPWORDS = {
    "and", "or", "to", "for", "in", "on", "with", "by", "as", "at", "from",
    "before", "after", "during", "of", "the", "a", "an"
}

_ACTION_VERBS = (
    "select", "choose", "decide",
    "etch", "rinse", "blot", "dry", "air-dry", "air-thin",
    "apply", "scrub", "agitate",
    "cure", "check", "verify",
    "isolate", "maintain", "avoid", "use", "re-etch", "re-prime",
    "extend", "place", "hold", "protect", "repeat", "review"
)

# Title-keyed tips + guaranteed variants
SLIDE_FALLBACK_BANK = {
    "choose": [
        "Check: Match adhesive system to enamel versus dentin margins.",
        "Do: Decide etch mode before isolation and etching steps.",
        "Avoid: Using one protocol across different adhesive systems.",
        "Tip: Prefer selective enamel etch when margins are enamel.",
        "Rescue: If unsure, follow IFU step-by-step without improvising.",
    ],
    "etch": [
        "Do: Respect IFU etch time to avoid dentin damage.",
        "Do: Rinse thoroughly and control moisture before priming.",
        "Avoid: Over-etching dentin and increasing sensitivity risk.",
        "Check: Dentin should look glistening, not flooded or chalky.",
        "Rescue: If over-dried, re-wet dentin lightly before adhesive.",
    ],
    "dentin": [
        "Do: Blot-dry dentin to a glistening surface after rinse.",
        "Avoid: Air-blasting dentin until it looks chalky.",
        "Check: Maintain moist dentin for resin infiltration.",
        "Tip: Use gentle air to evaporate solvent without desiccation.",
        "Rescue: If sensitivity occurs, reassess moisture and seal.",
    ],
    "self-etch": [
        "Do: Scrub self-etch adhesive for the full IFU time.",
        "Tip: Selectively etch enamel margins for stronger enamel bond.",
        "Avoid: Using self-etch on uncut enamel without selective etch.",
        "Check: Do not rinse self-etch unless IFU explicitly says so.",
        "Rescue: If enamel retention is low, switch to selective-etch mode.",
    ],
    "isolation": [
        "Do: Use rubber dam when possible during bonding steps.",
        "Avoid: Saliva or blood contamination after etch/prime.",
        "Check: Re-isolate immediately after any contamination event.",
        "Rescue: Re-etch enamel and re-prime dentin after contamination.",
        "Tip: Use retraction and suction to control gingival fluid.",
    ],
    "solvent": [
        "Do: Air-thin 5–10 seconds as per IFU to evaporate solvent.",
        "Check: Stop air-thinning when surface looks matte, not glossy.",
        "Avoid: Leaving pooled adhesive that increases film thickness.",
        "Tip: Scrub first, then air-thin to a uniform thin layer.",
        "Rescue: If pooling occurs, wick excess and re-air-thin gently.",
    ],
    "curing": [
        "Do: Keep curing tip close and perpendicular to the surface.",
        "Check: Extend cure time if tip distance exceeds 5 mm.",
        "Avoid: Curing at an angle or through contamination.",
        "Tip: Verify light output periodically with a radiometer.",
        "Rescue: If in doubt, add 5–10 seconds curing time safely.",
    ],
    "troubleshooting": [
        "Do: Review isolation, moisture control, and curing before materials.",
        "Check: Sensitivity often reflects desiccation or incomplete seal.",
        "Avoid: Repeating the same steps without root-cause review.",
        "Rescue: Re-etch enamel and re-bond using strict IFU timing.",
        "Tip: Check curing tip cleanliness and light output regularly.",
    ],
}

GENERIC_VARIANTS = [
    "Check: Follow IFU timing, agitation, and curing distance control.",
    "Avoid: Skipping steps or changing order during bonding.",
    "Do: Keep the field dry and uncontaminated throughout bonding.",
    "Tip: Use a checklist for etch, rinse, dry, apply, thin, cure.",
    "Rescue: If contamination occurs, repeat the last critical step.",
]

# Upgrade 1.75: strip these “lecture” inner prefixes
_INNER_PREFIXES = (
    "consider:",
    "understand:",
    "select:",
    "mechanism:",
    "limitations:",
    "indications:",
    "contraindications:",
)

# Upgrade 1.75: verbs that should become Check (not Do)
_DECISION_LEADERS = ("consider", "understand", "select", "know", "recognize")


# -----------------------------
# Robust JSON extraction (with repair)
# -----------------------------
def _extract_first_balanced_object(raw: str) -> str:
    raw = raw or ""
    start = raw.find("{")
    if start == -1:
        raise ValueError("No '{' found.")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i+1]

    return raw[start:]


def _try_parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty model output.")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    candidate = _extract_first_balanced_object(raw).strip()
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Decoded JSON is not an object.")
    return obj


def _build_repair_prompt(bad_output: str) -> str:
    return f"""
Return ONLY a valid STRICT JSON object (double quotes, no trailing commas).
No markdown. No explanations.

Schema:
{SLIDE_DECK_SCHEMA}

Fix/convert this output into STRICT JSON:
{bad_output}
""".strip()


def build_slide_writer_prompt(outline: Dict[str, Any]) -> str:
    outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
    return f"""
You are creating PowerPoint slides for dental professionals.

Use ONLY the provided clinical outline JSON.
Do NOT invent new teaching points.

STRICT OUTPUT REQUIREMENTS:
- One slide per teaching point, same order.
- Each slide MUST have:
  - title: max 8 words
  - bullets: EXACTLY 5 bullets
- Each bullet:
  - max 12 words
  - start with: "Do:", "Avoid:", "Check:", "Rescue:", or "Tip:"
  - chairside actions only
  - minimum 5 words after the tag
- Avoid repetition on the same slide.
- Output MUST be STRICT JSON.

JSON SCHEMA:
{SLIDE_DECK_SCHEMA}

CLINICAL OUTLINE JSON:
{outline_json}
""".strip()


# -----------------------------
# Bullet processing (Upgrade 1.75)
# -----------------------------
def _strip_tag(b: str) -> str:
    return re.sub(r"^(Do:|Avoid:|Check:|Rescue:|Tip:)\s*", "", b).strip()


def _enforce_tag(b: str) -> str:
    if b.startswith(_TAGS):
        return b
    return f"Do: {b}".strip()


def _remove_dont_phrases(text: str) -> str:
    text = re.sub(r"\b(don't|do not)\b[:\s]*", "", text, flags=re.I)
    return " ".join(text.split()).strip(" :-")


def _strip_inner_prefixes(tail: str) -> str:
    t = tail.strip()
    low = t.lower()
    for p in _INNER_PREFIXES:
        if low.startswith(p):
            t = t[len(p):].strip(" :-")
            break
    # also remove occurrences like "Do: Consider: ..." after tag stripping
    for p in _INNER_PREFIXES:
        t = re.sub(rf"\b{re.escape(p)}\b", "", t, flags=re.I).strip(" :-")
    return " ".join(t.split()).strip()


def _truncate_words_smart(tag: str, tail: str, max_words: int = 12) -> str:
    words = tail.split()
    if len(words) <= max_words:
        out = f"{tag} {tail}".strip()
        return out if out.endswith(".") else out + "."
    cut = words[:max_words]
    while cut and cut[-1].lower().strip(".,;:") in _TRAILING_STOPWORDS:
        cut = cut[:-1]
    if not cut:
        cut = words[:max_words]
    out = f"{tag} {' '.join(cut)}".strip()
    return out if out.endswith(".") else out + "."


def _choose_tag_for_tail(tag: str, tail: str) -> str:
    low = tail.lower().strip()
    for v in _DECISION_LEADERS:
        if low.startswith(v + " "):
            return "Check:"
    # If bullet is declarative (no action verb), prefer Check
    if not any(low.startswith(v) for v in _ACTION_VERBS):
        if tag == "Do:":
            return "Check:"
    return tag


def _ensure_action_tone(tag: str, tail: str) -> str:
    low = tail.lower()
    if any(low.startswith(v) for v in _ACTION_VERBS):
        return f"{tag} {tail}".strip()
    # Convert “knowledge” bullets to checks
    if tag in ("Do:", "Tip:"):
        tag = "Check:"
    return f"{tag} {tail}".strip()


def _normalize_one_bullet(b: str) -> str:
    b = " ".join((b or "").split())
    b = _enforce_tag(b)
    tag = next((t for t in _TAGS if b.startswith(t)), "Do:")
    tail = _strip_tag(b)

    tail = _remove_dont_phrases(tail)
    tail = re.sub(r"^(mechanism|benefit|risk|challenge|key step)\s*:\s*", "", tail, flags=re.I).strip()
    tail = _strip_inner_prefixes(tail)

    tag = _choose_tag_for_tail(tag, tail)

    b2 = _ensure_action_tone(tag, tail)
    tag2 = next((t for t in _TAGS if b2.startswith(t)), tag)
    tail2 = _strip_tag(b2)

    return _truncate_words_smart(tag2, tail2, 12)


def _dedup_strong(bullets: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for b in bullets:
        key = re.sub(r"[^a-z0-9]+", " ", _strip_tag(b).lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(b)
    return out


def _bank_for_title(title: str) -> List[str]:
    low = (title or "").lower()
    for k, bank in SLIDE_FALLBACK_BANK.items():
        if k in low:
            return bank
    return GENERIC_VARIANTS


def _fill_to_5(bullets: List[str], title: str) -> List[str]:
    normalized = [_normalize_one_bullet(b) for b in (bullets or []) if str(b).strip()]
    normalized = [b for b in normalized if "strict timing, air-thinning, and adequate curing energy" not in b.lower()]
    normalized = _dedup_strong(normalized)

    bank = _bank_for_title(title)
    bank_norm = [_normalize_one_bullet(b) for b in bank]

    max_iters = 20
    i = 0
    while len(normalized) < 5 and i < max_iters:
        candidate = bank_norm[i % len(bank_norm)]
        tmp = _dedup_strong(normalized + [candidate])
        if len(tmp) > len(normalized):
            normalized = tmp
        i += 1

    j = 1
    while len(normalized) < 5:
        normalized.append(f"Check: Confirm step sequence and timing ({j}).")
        normalized = _dedup_strong(normalized)
        j += 1

    return normalized[:5]


# -----------------------------
# Public API
# -----------------------------
def generate_slide_deck(
    *,
    outline: Dict[str, Any],
    llm_generate_text: Callable[[str, float, int], str],
) -> Dict[str, Any]:
    prompt = build_slide_writer_prompt(outline)
    raw = llm_generate_text(prompt, 0.25, 2200)

    try:
        deck = _try_parse_json(raw)
    except Exception:
        fixed = llm_generate_text(_build_repair_prompt(raw), 0.0, 2600)
        deck = _try_parse_json(fixed)

    tps = outline.get("teaching_points", []) or []
    if not isinstance(tps, list):
        tps = []

    slides_in = deck.get("slides", [])
    if not isinstance(slides_in, list):
        slides_in = []

    cleaned: List[Dict[str, Any]] = []
    for i, tp in enumerate(tps):
        tp = tp if isinstance(tp, dict) else {}
        s = slides_in[i] if i < len(slides_in) and isinstance(slides_in[i], dict) else {}

        title = (s.get("title") or tp.get("title") or "").strip()[:80]
        bullets = s.get("bullets") if isinstance(s.get("bullets"), list) else []
        bullets = _fill_to_5(bullets, title)
        if title:
            cleaned.append({"title": title, "bullets": bullets})

    if not cleaned:
        for tp in tps[:8]:
            title = str((tp or {}).get("title") or "Clinical Point").strip()[:80]
            cleaned.append({"title": title, "bullets": _fill_to_5([], title)})

    return {
        "topic": (outline.get("topic") or "").strip(),
        "audience": (outline.get("audience") or "").strip(),
        "slides": cleaned,
    }
