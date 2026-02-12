from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import re
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
try:
    from pydantic import model_validator
except ImportError:  # pydantic v1 fallback
    model_validator = None  # type: ignore[assignment]

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from odonto_rag.deck.pptx_builder import build_pptx_from_slide_plan
from odonto_rag.rag.providers.vertex import llm_generate_text

logger = logging.getLogger(__name__)


def _get_access_token() -> str:
    out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    if not out:
        raise RuntimeError("Empty access token. Run 'gcloud auth login'.")
    return out


def _embed_texts(
    texts: List[str], *, project: str, location: str, model: str
) -> List[List[float]]:
    token = _get_access_token()
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project}/locations/{location}/publishers/google/models/"
        f"{model}:predict"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"instances": [{"content": t} for t in texts]}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return [p["embeddings"]["values"] for p in data.get("predictions", [])]


def _collection_name(version: str, model: str) -> str:
    return f"odonto_items__{version}__{model}"


class RagQuery(BaseModel):
    query: str = Field(..., min_length=1)
    version_id: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    snippet_len: int = Field(160, ge=1, le=2000)
    collection: Optional[str] = None


class RagResult(BaseModel):
    item_id: str
    score: float
    text: str
    locator: Optional[Dict[str, Any]]
    doc_id: Optional[str] = None
    version_id: Optional[str] = None
    item_type: Optional[str] = None


class RagResponse(BaseModel):
    query: str
    version_id: str
    collection: str
    top_k: int
    results: List[RagResult]


class RetrievedItem(BaseModel):
    item_id: str
    score: float
    text: str
    locator: Optional[Dict[str, Any]]
    doc_id: Optional[str] = None
    version_id: Optional[str] = None
    item_type: Optional[str] = None


DEFAULT_ANSWER_SNIPPET_LEN = 800
DEFAULT_OUTLINE_SNIPPET_LEN = 500


class RagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    doc_id: Optional[str] = None
    version: str = Field(..., min_length=1)


class RagCitation(BaseModel):
    ref_id: str
    item_id: str
    page_start: int
    page_end: int
    score: float
    doc_id: Optional[str] = None
    doc_citation: Optional[str] = None
    doc_filename: Optional[str] = None


class RagAnswerResponse(BaseModel):
    answer: str
    citations: List[RagCitation]
    retrieved: Optional[List[RetrievedItem]] = None


class RagOutlineRequest(BaseModel):
    query: Optional[str] = Field(None, min_length=1)
    topic: Optional[str] = Field(None, min_length=1)
    top_k: int = Field(25, ge=1, le=50)
    doc_id: Optional[str] = None
    version: str = Field(..., min_length=1)
    max_sections: int = Field(10, ge=1, le=20)
    include_retrieved: bool = False


class RagOutlineSection(BaseModel):
    id: int
    heading: str
    learning_objectives: List[str]
    key_points: List[str]
    citations: List[str]


class RagOutlineResponse(BaseModel):
    title: str
    sections: List[RagOutlineSection]
    citations: List[RagCitation]
    retrieved: Optional[List[RetrievedItem]] = None


class RagSlidesPlanRequest(BaseModel):
    outline: Optional[Dict[str, Any]] = None
    query: Optional[str] = Field(None, min_length=1)
    top_k: int = Field(25, ge=1, le=50)
    version: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    max_sections: Optional[int] = Field(None, ge=1, le=20)
    slides_per_section: int = Field(2, ge=1, le=5)
    max_slides: int = Field(20, ge=1, le=50)
    include_retrieved: bool = False


class RagSlidePlanItem(BaseModel):
    slide_no: int
    title: str
    bullets: List[str]
    speaker_notes: Optional[str] = None
    citations: List[str]
    sources: List[RagCitation]


class RagSlidesPlanResponse(BaseModel):
    slides: List[RagSlidePlanItem]
    outline_used: Optional[RagOutlineResponse] = None
    retrieved: Optional[List[RetrievedItem]] = None


class RagSlidesPptxRequest(BaseModel):
    slide_plan: Dict[str, Any]
    filename: Optional[str] = None


class RagSlidesPptxResponse(BaseModel):
    path: str
    filename: str
    size_bytes: int


app = FastAPI(title="Odonto RAG API", version="0.1.0")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


def _resolve_collection(version_or_collection: str, model: str) -> str:
    if version_or_collection.startswith("odonto_items__"):
        return version_or_collection
    return _collection_name(version_or_collection, model)


def _page_range(locator: Optional[Dict[str, Any]]) -> tuple[int, int]:
    if not locator:
        return (0, 0)
    try:
        return (int(locator.get("page_start", 0)), int(locator.get("page_end", 0)))
    except Exception:
        return (0, 0)


def _normalize_location(raw: str, *, name: str) -> str:
    loc = raw.strip()
    if not loc:
        return ""
    if "://" in loc or "/" in loc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid {name}: must be a region like 'us-central1', got '{raw}'",
        )
    return loc


def retrieve_items(
    query: str, top_k: int, doc_id: Optional[str], version: Optional[str]
) -> List[RetrievedItem]:
    project = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID") or ""
    embed_location = os.environ.get("VERTEX_EMBED_LOCATION") or os.environ.get("GCP_LOCATION") or os.environ.get("LOCATION") or ""
    embed_location = _normalize_location(embed_location, name="VERTEX_EMBED_LOCATION")
    if not project or not embed_location:
        raise HTTPException(
            status_code=500,
            detail="Missing GCP_PROJECT/PROJECT_ID or VERTEX_EMBED_LOCATION/GCP_LOCATION/LOCATION",
        )

    if not version:
        raise HTTPException(status_code=400, detail="Missing version")

    model = os.environ.get("VERTEX_EMBED_MODEL", "text-embedding-004")
    collection = _resolve_collection(version, model)

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    api_key = os.environ.get("QDRANT_API_KEY") or None

    try:
        vec = _embed_texts([query], project=project, location=embed_location, model=model)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}") from e

    q = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)
    query_filter = None
    if doc_id:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_id", match=qmodels.MatchValue(value=doc_id)
                )
            ]
        )

    points = q.query_points(
        collection_name=collection,
        query=vec,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points

    results: List[RetrievedItem] = []
    for p in points:
        payload = p.payload or {}
        text = (payload.get("text") or "").replace("\n", " ")
        results.append(
            RetrievedItem(
                item_id=str(p.id),
                score=float(p.score),
                text=text,
                locator=payload.get("locator"),
                doc_id=payload.get("doc_id"),
                version_id=payload.get("version"),
                item_type=payload.get("item_type"),
            )
        )

    return results


def _truncate_text(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "…"


def _build_context(items: List[RetrievedItem], snippet_len: int) -> str:
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        page_start, page_end = _page_range(item.locator)
        snippet = _truncate_text(item.text, snippet_len)
        lines.append(f"[S{idx}] (p. {page_start}-{page_end}) {snippet}")
    return "\n".join(lines).strip()


def _dedupe_header_items(items: List[RetrievedItem]) -> List[RetrievedItem]:
    deduped: List[RetrievedItem] = []
    seen_headers: set[str] = set()
    for item in items:
        if (item.item_type or "").lower() == "header":
            norm = " ".join(item.text.strip().lower().split())
            if norm in seen_headers:
                continue
            seen_headers.add(norm)
        deduped.append(item)
    return deduped


def _make_answer_prompt(query: str, context: str) -> str:
    return (
        "You are a didactic assistant for dental materials.\n"
        "Answer the question using ONLY the provided context.\n"
        "Rules:\n"
        "- Use clear sections and bullet points.\n"
        "- When stating facts, cite with [S#] tokens.\n"
        "- Do not invent citations.\n"
        "- If context is insufficient, state what is missing and still cite what you used.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:\n"
    )


def _make_outline_prompt(query: str, context: str, max_sections: int) -> str:
    return (
        "You are a didactic assistant for dental materials.\n"
        "Build a study outline using ONLY the provided context.\n"
        "Return STRICT JSON only (no markdown, no code fences, no extra text).\n"
        "Use this exact schema:\n"
        '{ "title": str, "sections": [ { "id": int, "heading": str, "learning_objectives": [str], '
        '"key_points": [str], "citations": ["S1","S2"] } ] }\n'
        f"Limit sections to at most {max_sections}.\n"
        "Citations must use only S# tokens from context.\n\n"
        f"Query:\n{query}\n\n"
        f"Context:\n{context}\n"
    )


def _extract_citation_tokens(answer: str) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for match in re.finditer(r"\[S(\d+)\]", answer):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _extract_outline_citation_tokens(sections: List[RagOutlineSection]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for section in sections:
        for raw in section.citations:
            match = re.fullmatch(r"\[?S(\d+)\]?", str(raw).strip())
            if not match:
                continue
            idx = int(match.group(1))
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
    return ordered


def _strip_json_fences(s: str) -> str:
    t = s.strip()
    if not t:
        return ""
    if t.startswith("\ufeff"):
        t = t.lstrip("\ufeff").strip()
    if "```" in t:
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE)
        if fence_match:
            t = fence_match.group(1).strip()
    first = t.find("{")
    last = t.rfind("}")
    if first == -1 or last == -1 or last < first:
        return ""
    return t[first : last + 1].strip()


def _map_token_ids_to_citations(
    token_ids: List[int], items: List[RetrievedItem]
) -> List[RagCitation]:
    citations: List[RagCitation] = []
    for token_id in token_ids:
        if 1 <= token_id <= len(items):
            item = items[token_id - 1]
            page_start, page_end = _page_range(item.locator)
            citations.append(
                RagCitation(
                    ref_id=f"S{token_id}",
                    item_id=item.item_id,
                    page_start=page_start,
                    page_end=page_end,
                    score=item.score,
                    doc_id=item.doc_id,
                )
            )
    return citations


def _map_ref_tokens_to_citations(
    refs: List[str], citation_by_ref: Dict[str, RagCitation]
) -> List[RagCitation]:
    out: List[RagCitation] = []
    seen: set[str] = set()
    for raw in refs:
        m = re.fullmatch(r"\[?S(\d+)\]?", str(raw).strip())
        if not m:
            continue
        ref = f"S{int(m.group(1))}"
        if ref in seen:
            continue
        c = citation_by_ref.get(ref)
        if c:
            seen.add(ref)
            out.append(c)
    return out


def _catalog_db_path() -> str:
    env_path = (os.environ.get("SQLITE_PATH") or os.environ.get("CATALOG_DB_PATH") or "").strip()
    if env_path:
        return env_path
    if Path("catalog.sqlite3").exists():
        return "catalog.sqlite3"
    return "data/catalog.db"


def _basename_from_uri_or_path(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    return Path(raw.split("?", 1)[0].rstrip("/")).name


def _load_documents_metadata(doc_ids: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    unique_doc_ids = [d for d in dict.fromkeys(doc_ids) if d]
    if not unique_doc_ids:
        return {}
    db_path = _catalog_db_path()
    if not Path(db_path).exists():
        return {}

    out: Dict[str, Dict[str, Optional[str]]] = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        chunk_size = 500
        for i in range(0, len(unique_doc_ids), chunk_size):
            chunk = unique_doc_ids[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            q = (
                "SELECT doc_id, gcs_raw_path, metadata_json "
                "FROM documents "
                f"WHERE doc_id IN ({placeholders})"
            )
            for row in cur.execute(q, chunk):
                metadata_obj: Dict[str, Any] = {}
                raw_metadata = row["metadata_json"]
                if isinstance(raw_metadata, dict):
                    metadata_obj = raw_metadata
                if isinstance(raw_metadata, str) and raw_metadata.strip():
                    try:
                        parsed = json.loads(raw_metadata)
                        if isinstance(parsed, dict):
                            metadata_obj = parsed
                    except Exception:
                        metadata_obj = {}
                doc_id = str(row["doc_id"]) if row["doc_id"] is not None else ""
                if not doc_id:
                    continue
                out[doc_id] = {
                    "doc_citation": str(metadata_obj.get("citation")).strip() if metadata_obj.get("citation") else None,
                    "doc_filename": _basename_from_uri_or_path(str(row["gcs_raw_path"] or "")) or None,
                }
    except Exception as e:
        logger.warning("Doc metadata enrichment skipped: %s", e)
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def _enrich_citation_map_with_doc_metadata(
    citation_by_ref: Dict[str, RagCitation],
    item_doc_map: Dict[str, str],
) -> Dict[str, RagCitation]:
    doc_ids = []
    for c in citation_by_ref.values():
        resolved_doc_id = c.doc_id or item_doc_map.get(c.item_id, "")
        if resolved_doc_id:
            doc_ids.append(resolved_doc_id)
    docs_meta = _load_documents_metadata(doc_ids)
    enriched: Dict[str, RagCitation] = {}
    for ref, citation in citation_by_ref.items():
        resolved_doc_id = citation.doc_id or item_doc_map.get(citation.item_id, "")
        doc_meta = docs_meta.get(resolved_doc_id, {})
        update = {
            "doc_id": resolved_doc_id or None,
            "doc_citation": doc_meta.get("doc_citation"),
            "doc_filename": doc_meta.get("doc_filename"),
        }
        if hasattr(citation, "model_copy"):
            enriched[ref] = citation.model_copy(update=update)  # type: ignore[attr-defined]
        else:
            enriched[ref] = citation.copy(update=update)
    return enriched


def _dedupe_outline_sections(sections: List[RagOutlineSection]) -> List[RagOutlineSection]:
    deduped: List[RagOutlineSection] = []
    seen: set[str] = set()
    for sec in sections:
        norm = " ".join(sec.heading.strip().lower().split())
        if norm and norm in seen:
            continue
        if norm:
            seen.add(norm)
        deduped.append(sec)
    return deduped


def _make_slides_plan_prompt(
    outline_json: str, target_slides: int, max_slides: int
) -> str:
    return (
        "You are planning didactic dental slides from an outline.\n"
        "Return STRICT JSON only (no markdown, no code fences, no extra text).\n"
        "Use this schema exactly:\n"
        '{ "slides": [ { "slide_no": int, "title": str, "bullets": [str], "speaker_notes": str, "citations": ["S1","S2"] } ] }\n'
        f"Generate around {target_slides} slides and never exceed {max_slides}.\n"
        "Avoid duplicate slide titles/headings.\n"
        "Use only citation tokens already present in the outline.\n\n"
        f"Outline JSON:\n{outline_json}\n"
    )


@app.post("/rag/query", response_model=RagResponse)
def rag_query(req: RagQuery) -> RagResponse:
    model = os.environ.get("VERTEX_EMBED_MODEL", "text-embedding-004")
    collection = req.collection or _collection_name(req.version_id, model)
    items = retrieve_items(
        req.query,
        req.top_k,
        doc_id=None,
        version=collection,
    )

    results: List[RagResult] = []
    for item in items:
        snippet = _truncate_text(item.text, req.snippet_len)
        results.append(
            RagResult(
                item_id=item.item_id,
                score=item.score,
                text=snippet,
                locator=item.locator,
                doc_id=item.doc_id,
                version_id=item.version_id,
                item_type=item.item_type,
            )
        )

    return RagResponse(
        query=req.query,
        version_id=req.version_id,
        collection=collection,
        top_k=req.top_k,
        results=results,
    )


# Smoke test (local):
# export QDRANT_URL=http://localhost:6333
# export GCP_PROJECT=odontology-rag-slides
# export VERTEX_EMBED_MODEL=text-embedding-004
# export VERTEX_EMBED_LOCATION=europe-west1
# export VERTEX_LLM_LOCATION=us-central1
# export VERTEX_LLM_MODEL=gemini-2.0-flash-001
# uvicorn odonto_rag.api.rag_app:app --reload --port 8000
# curl -s -X POST http://localhost:8000/rag/query \
#   -H "Content-Type: application/json" \
#   -d '{"query":"What is the role of adhesive systems?","version_id":"v1-docai","top_k":3}' | jq
# curl -s -X POST http://localhost:8000/rag/answer \
#   -H "Content-Type: application/json" \
#   -d '{"query":"What is the role of adhesive systems?","top_k":3,"version":"v1-docai"}' | jq


@app.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(req: RagAnswerRequest) -> RagAnswerResponse:
    project = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID") or ""
    llm_location = os.environ.get("VERTEX_LLM_LOCATION") or os.environ.get("GCP_LOCATION") or os.environ.get("LOCATION") or ""
    llm_location = _normalize_location(llm_location, name="VERTEX_LLM_LOCATION")
    model = os.environ.get("VERTEX_LLM_MODEL") or ""
    if not project or not llm_location or not model:
        raise HTTPException(
            status_code=500,
            detail="Missing GCP_PROJECT/PROJECT_ID or VERTEX_LLM_LOCATION/GCP_LOCATION/LOCATION or VERTEX_LLM_MODEL",
        )

    items = retrieve_items(req.query, req.top_k, req.doc_id, req.version)
    context = _build_context(items, snippet_len=DEFAULT_ANSWER_SNIPPET_LEN)
    prompt = _make_answer_prompt(req.query, context)

    try:
        answer = llm_generate_text(
            prompt,
            temperature=0.2,
            max_output_tokens=1400,
            model=model,
            project=project,
            location=llm_location,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}") from e

    token_ids = _extract_citation_tokens(answer)
    citations = _map_token_ids_to_citations(token_ids, items)

    return RagAnswerResponse(answer=answer, citations=citations, retrieved=items)


@app.post("/rag/outline", response_model=RagOutlineResponse)
def rag_outline(req: RagOutlineRequest) -> RagOutlineResponse:
    project = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID") or ""
    llm_location = os.environ.get("VERTEX_LLM_LOCATION") or os.environ.get("GCP_LOCATION") or os.environ.get("LOCATION") or ""
    llm_location = _normalize_location(llm_location, name="VERTEX_LLM_LOCATION")
    model = os.environ.get("VERTEX_LLM_MODEL") or ""
    if not project or not llm_location or not model:
        raise HTTPException(
            status_code=500,
            detail="Missing GCP_PROJECT/PROJECT_ID or VERTEX_LLM_LOCATION/GCP_LOCATION/LOCATION or VERTEX_LLM_MODEL",
        )

    query_text = (req.query or req.topic or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Missing query/topic")

    items = retrieve_items(query_text, req.top_k, req.doc_id, req.version)
    context_items = _dedupe_header_items(items)
    context = _build_context(context_items, snippet_len=DEFAULT_OUTLINE_SNIPPET_LEN)
    prompt = _make_outline_prompt(query_text, context, req.max_sections)

    try:
        raw_output = llm_generate_text(
            prompt,
            temperature=0.2,
            max_output_tokens=2200,
            model=model,
            project=project,
            location=llm_location,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}") from e

    try:
        cleaned = _strip_json_fences(raw_output)
        if not cleaned:
            raise ValueError("missing JSON object braces")
        parsed = json.loads(cleaned)
    except Exception as e:
        logger.error("Outline JSON parse failed. raw_output_head=%s", raw_output[:500])
        raise HTTPException(
            status_code=500,
            detail=f"Outline JSON parse failed: {e}; output_head={raw_output[:200]}",
        ) from e

    title = str(parsed.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=500, detail="Outline JSON invalid: missing title")

    sections_raw = parsed.get("sections")
    if not isinstance(sections_raw, list):
        raise HTTPException(status_code=500, detail="Outline JSON invalid: sections must be a list")

    sections: List[RagOutlineSection] = []
    for sec in sections_raw[: req.max_sections]:
        if not isinstance(sec, dict):
            continue
        sections.append(
            RagOutlineSection(
                id=int(sec.get("id", 0)),
                heading=str(sec.get("heading") or "").strip(),
                learning_objectives=[str(x).strip() for x in sec.get("learning_objectives", []) if str(x).strip()],
                key_points=[str(x).strip() for x in sec.get("key_points", []) if str(x).strip()],
                citations=[str(x).strip() for x in sec.get("citations", []) if str(x).strip()],
            )
        )

    token_ids = _extract_outline_citation_tokens(sections)
    citations = _map_token_ids_to_citations(token_ids, context_items)

    return RagOutlineResponse(
        title=title,
        sections=sections,
        citations=citations,
        retrieved=context_items if req.include_retrieved else None,
    )


@app.post("/rag/slides/plan", response_model=RagSlidesPlanResponse)
def rag_slides_plan(req: RagSlidesPlanRequest) -> RagSlidesPlanResponse:
    query_text = (req.query or "").strip()

    if req.outline is not None and query_text:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of outline or query, not both",
        )

    if req.outline is None and not query_text:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'outline' or 'query'",
        )

    project = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID") or ""
    llm_location = os.environ.get("VERTEX_LLM_LOCATION") or os.environ.get("GCP_LOCATION") or os.environ.get("LOCATION") or ""
    llm_location = _normalize_location(llm_location, name="VERTEX_LLM_LOCATION")
    model = os.environ.get("VERTEX_LLM_MODEL") or ""
    if not project or not llm_location or not model:
        raise HTTPException(
            status_code=500,
            detail="Missing GCP_PROJECT/PROJECT_ID or VERTEX_LLM_LOCATION/GCP_LOCATION/LOCATION or VERTEX_LLM_MODEL",
        )

    outline_used: Optional[RagOutlineResponse] = None
    retrieved: Optional[List[RetrievedItem]] = None

    if req.outline is not None:
        raw = req.outline
        sections_raw = raw.get("sections")
        if not isinstance(sections_raw, list):
            raise HTTPException(status_code=400, detail="outline.sections must be a list")
        sections: List[RagOutlineSection] = []
        for sec in sections_raw:
            if not isinstance(sec, dict):
                continue
            sections.append(
                RagOutlineSection(
                    id=int(sec.get("id", 0)),
                    heading=str(sec.get("heading") or "").strip(),
                    learning_objectives=[str(x).strip() for x in sec.get("learning_objectives", []) if str(x).strip()],
                    key_points=[str(x).strip() for x in sec.get("key_points", []) if str(x).strip()],
                    citations=[str(x).strip() for x in sec.get("citations", []) if str(x).strip()],
                )
            )
        citations_raw = raw.get("citations") if isinstance(raw.get("citations"), list) else []
        citations: List[RagCitation] = []
        for c in citations_raw:
            if not isinstance(c, dict):
                continue
            citations.append(
                RagCitation(
                    ref_id=str(c.get("ref_id") or "").strip(),
                    item_id=str(c.get("item_id") or "").strip(),
                    page_start=int(c.get("page_start", 0)),
                    page_end=int(c.get("page_end", 0)),
                    score=float(c.get("score", 0.0)),
                    doc_id=(str(c.get("doc_id")).strip() if c.get("doc_id") is not None else None),
                    doc_citation=(str(c.get("doc_citation")).strip() if c.get("doc_citation") is not None else None),
                    doc_filename=(str(c.get("doc_filename")).strip() if c.get("doc_filename") is not None else None),
                )
            )
        if req.include_retrieved and isinstance(raw.get("retrieved"), list):
            retrieved = []
            for item in raw["retrieved"]:
                if not isinstance(item, dict):
                    continue
                retrieved.append(
                    RetrievedItem(
                        item_id=str(item.get("item_id") or "").strip(),
                        score=float(item.get("score", 0.0)),
                        text=str(item.get("text") or ""),
                        locator=item.get("locator"),
                        doc_id=item.get("doc_id"),
                        version_id=item.get("version_id"),
                        item_type=item.get("item_type"),
                    )
                )
        outline_used = RagOutlineResponse(
            title=str(raw.get("title") or "").strip(),
            sections=sections,
            citations=citations,
            retrieved=retrieved if req.include_retrieved else None,
        )
    else:
        outline_used = rag_outline(
            RagOutlineRequest(
                query=query_text,
                top_k=req.top_k,
                doc_id=req.doc_id,
                version=req.version,
                max_sections=req.max_sections or 10,
                include_retrieved=req.include_retrieved,
            )
        )
        retrieved = outline_used.retrieved if req.include_retrieved else None

    if not outline_used.title:
        raise HTTPException(status_code=400, detail="outline.title is required")

    dedup_sections = _dedupe_outline_sections(outline_used.sections)
    if not dedup_sections:
        raise HTTPException(status_code=400, detail="outline.sections is empty")
    target_slides = min(req.max_slides, max(1, len(dedup_sections) * req.slides_per_section))
    outline_for_prompt = {
        "title": outline_used.title,
        "sections": [s.model_dump() for s in dedup_sections],
    }
    prompt = _make_slides_plan_prompt(
        json.dumps(outline_for_prompt, ensure_ascii=False),
        target_slides=target_slides,
        max_slides=req.max_slides,
    )

    try:
        raw_output = llm_generate_text(
            prompt,
            temperature=0.2,
            max_output_tokens=2600,
            model=model,
            project=project,
            location=llm_location,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}") from e

    try:
        cleaned = _strip_json_fences(raw_output)
        if not cleaned:
            raise ValueError("missing JSON object braces")
        parsed = json.loads(cleaned)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Slides plan JSON parse failed: {e}; output_head={raw_output[:200]}",
        ) from e

    slides_raw = parsed.get("slides")
    if not isinstance(slides_raw, list):
        raise HTTPException(status_code=500, detail="Slides JSON invalid: slides must be a list")

    item_doc_map: Dict[str, str] = {}
    if retrieved:
        for item in retrieved:
            if item.item_id and item.doc_id:
                item_doc_map[item.item_id] = item.doc_id
    for c in outline_used.citations:
        if c.item_id and c.doc_id and c.item_id not in item_doc_map:
            item_doc_map[c.item_id] = c.doc_id

    citation_by_ref = {c.ref_id: c for c in outline_used.citations}
    citation_by_ref = _enrich_citation_map_with_doc_metadata(citation_by_ref, item_doc_map)
    slides: List[RagSlidePlanItem] = []
    seen_titles: set[str] = set()
    for idx, slide in enumerate(slides_raw, start=1):
        if len(slides) >= req.max_slides:
            break
        if not isinstance(slide, dict):
            continue
        title = str(slide.get("title") or "").strip()
        if not title:
            continue
        norm_title = " ".join(title.lower().split())
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        citations = [str(x).strip() for x in slide.get("citations", []) if str(x).strip()]
        slides.append(
            RagSlidePlanItem(
                slide_no=int(slide.get("slide_no", idx)),
                title=title,
                bullets=[str(x).strip() for x in slide.get("bullets", []) if str(x).strip()],
                speaker_notes=(str(slide.get("speaker_notes")).strip() if slide.get("speaker_notes") is not None else None),
                citations=citations,
                sources=_map_ref_tokens_to_citations(citations, citation_by_ref),
            )
        )

    for idx, slide in enumerate(slides, start=1):
        slide.slide_no = idx

    if not slides:
        raise HTTPException(status_code=500, detail="Slides plan JSON invalid: produced 0 slides")

    return RagSlidesPlanResponse(
        slides=slides,
        outline_used=outline_used if req.outline is None else None,
        retrieved=retrieved if req.include_retrieved else None,
    )


@app.post("/rag/slides/pptx", response_model=RagSlidesPptxResponse)
def rag_slides_pptx(req: RagSlidesPptxRequest, download: int = 0) -> RagSlidesPptxResponse | FileResponse:
    slides = req.slide_plan.get("slides") if isinstance(req.slide_plan, dict) else None
    if not isinstance(slides, list) or not slides:
        raise HTTPException(status_code=400, detail="slide_plan['slides'] must be a non-empty list")

    try:
        pptx_output_dir = os.environ.get("PPTX_OUTPUT_DIR", "out/decks")
        out_path = build_pptx_from_slide_plan(
            req.slide_plan,
            out_dir=pptx_output_dir,
            filename=req.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPTX build failed: {e}") from e

    if download == 1:
        return FileResponse(
            path=str(out_path),
            filename=out_path.name,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

    return RagSlidesPptxResponse(
        path=str(out_path),
        filename=out_path.name,
        size_bytes=out_path.stat().st_size,
    )
