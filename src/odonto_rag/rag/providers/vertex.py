from __future__ import annotations
from typing import Any, Dict, List


def vertex_extract_text(resp_json: Dict[str, Any]) -> str:
    """
    Extracts text from Vertex AI generateContent response.
    Safe even when parts are missing.
    """
    cands = resp_json.get("candidates") or []
    if not cands:
        return ""

    content = (cands[0].get("content") or {})
    parts: List[Dict[str, Any]] = content.get("parts") or []
    if not parts:
        return ""

    out = []
    for p in parts:
        t = p.get("text")
        if t:
            out.append(t)
    return "\n".join(out).strip()

