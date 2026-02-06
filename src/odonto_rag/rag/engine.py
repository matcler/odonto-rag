from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal, Dict, Callable

from odonto_rag.rag.outline import generate_clinical_outline
from odonto_rag.rag.outline_refiner import refine_clinical_outline
from odonto_rag.deck.builder import get_template_path


DeckMode = Literal["clinical", "practical"]


@dataclass
class EngineRequest:
    topic: str
    mode: DeckMode = "clinical"
    template_path: Optional[str] = None
    refine: bool = True

    # Defaults for your outline signature
    audience: str = "dentist"
    context: str = ""

    # Optional: if you want to inject your own llm function from caller (API/UI)
    llm_generate_text: Optional[Callable[..., Any]] = None

    extra: Optional[Dict[str, Any]] = None


@dataclass
class EngineResult:
    outline: Any
    refined_outline: Any
    pptx_path: Optional[str]
    template_used: str


def _resolve_llm_generate_text() -> Callable[..., Any]:
    """
    Try to resolve a text-generation function from the current codebase.

    We search common names in odonto_rag.rag.providers.vertex.
    If nothing is found, we raise a clear error so you can map it once.
    """
    candidates = (
        "llm_generate_text",
        "vertex_generate_text",
        "generate_text",
        "call_llm",
        "llm_text",
    )
    try:
        import odonto_rag.rag.providers.vertex as v  # type: ignore
        for name in candidates:
            fn = getattr(v, name, None)
            if callable(fn):
                return fn
    except Exception:
        pass

    raise RuntimeError(
        "Could not resolve llm_generate_text. "
        "Pass EngineRequest.llm_generate_text explicitly, or add one of these callables "
        "to odonto_rag.rag.providers.vertex: "
        + ", ".join(candidates)
    )


def _call_generate_outline(req: EngineRequest) -> Any:
    """
    Call generate_clinical_outline adapting to its real signature:
    keyword-only args: topic, audience, context, llm_generate_text
    """
    sig = inspect.signature(generate_clinical_outline)
    params = sig.parameters

    # Build kwargs only for parameters that exist
    kwargs: Dict[str, Any] = {}

    if "topic" in params:
        kwargs["topic"] = req.topic
    if "audience" in params:
        kwargs["audience"] = req.audience
    if "context" in params:
        kwargs["context"] = req.context
    if "llm_generate_text" in params:
        kwargs["llm_generate_text"] = req.llm_generate_text or _resolve_llm_generate_text()

    # If it accepts no params, fall back
    if len(params) == 0:
        return generate_clinical_outline()

    return generate_clinical_outline(**kwargs)


def _resolve_deck_generator():
    """
    Resolve a deck generation function without forcing a big refactor now.

    Priority:
    1) odonto_rag.deck.slide_writer.generate_slide_deck
    2) odonto_rag.deck.builder.generate_slide_deck
    3) common builder names: build_pptx/build_deck/generate_pptx/render_deck
    """
    try:
        from odonto_rag.deck.slide_writer import generate_slide_deck  # type: ignore
        return generate_slide_deck
    except Exception:
        pass

    try:
        from odonto_rag.deck.builder import generate_slide_deck  # type: ignore
        return generate_slide_deck
    except Exception:
        pass

    try:
        import odonto_rag.deck.builder as b  # type: ignore
        for name in ("build_pptx", "build_deck", "generate_pptx", "render_deck"):
            if hasattr(b, name):
                return getattr(b, name)
    except Exception:
        pass

    raise RuntimeError(
        "Could not resolve a deck generator. Expected one of:\n"
        "- odonto_rag.deck.slide_writer.generate_slide_deck\n"
        "- odonto_rag.deck.builder.generate_slide_deck\n"
        "- odonto_rag.deck.builder.build_pptx/build_deck/generate_pptx/render_deck\n"
    )


def generate_deck(req: EngineRequest) -> EngineResult:
    if not req.topic or not req.topic.strip():
        raise ValueError("EngineRequest.topic must be a non-empty string")

    template_used = req.template_path or get_template_path()

    # 1) Outline (adapt to real signature)
    outline = _call_generate_outline(req)

    # 2) Refine (optional)
    refined = refine_clinical_outline(outline) if req.refine else outline

    # 3) Deck render (best-effort during migration)
    pptx_path: Optional[str] = None
    deck_fn = _resolve_deck_generator()

    try:
        try:
            pptx_path = deck_fn(refined, template_path=template_used)  # type: ignore
        except TypeError:
            try:
                pptx_path = deck_fn(refined, template=template_used)  # type: ignore
            except TypeError:
                try:
                    pptx_path = deck_fn(refined, template_used)  # type: ignore
                except TypeError:
                    pptx_path = deck_fn(req.topic, refined, template_path=template_used)  # type: ignore
    except Exception:
        pptx_path = None

    if isinstance(pptx_path, Path):
        pptx_path = str(pptx_path)

    return EngineResult(
        outline=outline,
        refined_outline=refined,
        pptx_path=pptx_path,
        template_used=str(template_used),
    )
