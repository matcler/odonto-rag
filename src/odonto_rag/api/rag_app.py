from __future__ import annotations

import json
import os
import subprocess
import re
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from odonto_rag.rag.providers.vertex import llm_generate_text


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
    if t.startswith("```"):
        if t.startswith("```json"):
            t = t[len("```json") :].strip()
        else:
            t = t[len("```") :].strip()
        if t.endswith("```"):
            t = t[: -len("```")].strip()
    first = t.find("{")
    last = t.rfind("}")
    if first == -1 or last == -1 or last < first:
        return ""
    return t[first : last + 1].strip()


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
                )
            )

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
    citations: List[RagCitation] = []
    for token_id in token_ids:
        if 1 <= token_id <= len(context_items):
            item = context_items[token_id - 1]
            page_start, page_end = _page_range(item.locator)
            citations.append(
                RagCitation(
                    ref_id=f"S{token_id}",
                    item_id=item.item_id,
                    page_start=page_start,
                    page_end=page_end,
                    score=item.score,
                )
            )

    return RagOutlineResponse(
        title=title,
        sections=sections,
        citations=citations,
        retrieved=context_items if req.include_retrieved else None,
    )
