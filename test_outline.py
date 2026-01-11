import json
import time
import random
import requests

from rag_engine.clinical_outline import generate_clinical_outline
from rag_engine.outline_refiner import refine_clinical_outline
from rag_engine.slide_writer import generate_slide_deck
from rag_engine.gcloud_auth import gcloud_token
from rag_engine.vertex_utils import vertex_extract_text

PROJECT_ID = "odontology-rag-slides"
MODEL_ID = "gemini-2.5-flash"  # mantieni il tuo
LOCATION = "global"


# ---- Response Schemas (Vertex Structured Output) ----

CLINICAL_OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "audience": {"type": "string"},
        "teaching_points": {
            "type": "array",
            "minItems": 6,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "key_takeaways": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
                    "common_pitfalls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                    "what_to_do": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
                },
                "required": ["title", "key_takeaways", "common_pitfalls", "what_to_do"],
            },
        },
    },
    "required": ["topic", "audience", "teaching_points"],
}

SLIDE_DECK_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "audience": {"type": "string"},
        "slides": {
            "type": "array",
            "minItems": 4,
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
                },
                "required": ["title", "bullets"],
            },
        },
    },
    "required": ["topic", "audience", "slides"],
}


def gemini_generate_text(prompt: str, temperature: float, max_tokens: int, response_schema: dict | None = None) -> str:
    # v1beta1: responseSchema/responseMimeType sono supportati e documentati
    url = (
        f"https://aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/google/models/{MODEL_ID}:generateContent"
    )

    gen_cfg = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_tokens),
    }

    # Structured output (kills JSON truncation/formatting issues)
    if response_schema is not None:
        gen_cfg["responseMimeType"] = "application/json"
        gen_cfg["responseSchema"] = response_schema

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }

    max_attempts = 6
    base_sleep = 2.0
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {gcloud_token()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            if r.status_code == 429:
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.8)
                print(f"WARN 429 rate limit (attempt {attempt}/{max_attempts}) -> sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return vertex_extract_text(r.json())

        except requests.exceptions.RequestException as e:
            last_exc = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.8)
            print(f"WARN request error (attempt {attempt}/{max_attempts}): {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise last_exc if last_exc else RuntimeError("LLM call failed without exception")


if __name__ == "__main__":
    context = """
Adhesive systems can be classified as etch-and-rinse or self-etch.
Etch-and-rinse requires phosphoric acid etching and is technique-sensitive.
Self-etch adhesives reduce post-operative sensitivity and simplify steps.
Selective enamel etching improves enamel bond strength with self-etch.
Common failures include contamination, over-drying dentin, and inadequate curing.
Clinical tips include rubber dam use, scrubbing primer, and strict timing.
Specific protocols may be needed for zirconia, metal, and existing composite.
""".strip()

    outline = generate_clinical_outline(
        topic="Dental Adhesive Systems",
        audience="Dentists",
        context=context,
        llm_generate_text=lambda p, t, m: gemini_generate_text(p, t, min(m, 1800), response_schema=CLINICAL_OUTLINE_SCHEMA),
    )

    refined = refine_clinical_outline(
        outline=outline,
        llm_generate_text=lambda p, t, m: gemini_generate_text(p, t, min(m, 2200), response_schema=CLINICAL_OUTLINE_SCHEMA),
    )

    deck = generate_slide_deck(
        outline=refined,
        llm_generate_text=lambda p, t, m: gemini_generate_text(p, t, min(m, 2200), response_schema=SLIDE_DECK_SCHEMA),
    )

    print("=== REFINED OUTLINE (titles) ===")
    for i, tp in enumerate(refined.get("teaching_points", []), 1):
        print(i, "-", tp.get("title"))

    print("\n=== SLIDES (preview) ===")
    for i, s in enumerate(deck.get("slides", []), 1):
        print("\nSlide", i, ":", s.get("title"))
        for b in s.get("bullets", []):
            print(" -", b)
