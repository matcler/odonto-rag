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


# --- Text generation (Gemini on Vertex AI) -----------------------------------

import os
from typing import Any, Optional


def llm_generate_text(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    *,
    model: str | None = None,
    project: str | None = None,
    location: str | None = None,
    **_: object,
) -> str:
    """
    LLM text generation on Vertex AI (Gemini).

    Compatible with calls like:
      llm_generate_text(prompt, 0.10, 1800)
    """
    import os

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except Exception as e:
        raise RuntimeError(
            "Missing Vertex AI deps. Install with: pip install google-cloud-aiplatform"
        ) from e

    project = project or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = location or os.getenv("GCP_LOCATION") or "europe-west1"
    if not project:
        raise RuntimeError(
            "Vertex init requires a project. Set env GCP_PROJECT (or GOOGLE_CLOUD_PROJECT)."
        )

    vertexai.init(project=project, location=location)

    model_name = model or os.getenv("VERTEX_LLM_MODEL") or "gemini-1.5-flash"
    gm = GenerativeModel(model_name)

    cfg = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    resp = gm.generate_content(prompt, generation_config=cfg)
    return getattr(resp, "text", "") or ""
