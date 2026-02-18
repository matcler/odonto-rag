from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import re
from datetime import datetime, timezone
import zipfile
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
    specialty: Optional[str] = None


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
    specialty: Optional[str] = None


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
    specialty: Optional[str] = None
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
    specialty: Optional[str] = None
    max_sections: Optional[int] = Field(None, ge=1, le=20)
    slides_per_section: int = Field(2, ge=1, le=5)
    max_slides: int = Field(20, ge=1, le=50)
    include_retrieved: bool = False


class RagSlidePlanItem(BaseModel):
    slide_no: int
    title: str
    bullets: List[str]
    bullet_source_item_ids: List[List[str]] = Field(default_factory=list)
    bullet_sources_structured: List[List[Dict[str, Any]]] = Field(default_factory=list)
    bullet_visual_asset_ids: List[List[str]] = Field(default_factory=list)
    visual_role: str = "illustrative"
    visuals: List[Dict[str, Any]] = Field(default_factory=list)
    speaker_notes: Optional[str] = None
    citations: List[str]
    sources: List[RagCitation]


class RagSlidesPlanResponse(BaseModel):
    slides: List[RagSlidePlanItem]
    request: Optional[Dict[str, Any]] = None
    outline_used: Optional[RagOutlineResponse] = None
    retrieved: Optional[List[RetrievedItem]] = None


class RagSlidesPptxRequest(BaseModel):
    slide_plan: Dict[str, Any]
    filename: Optional[str] = None
    evidence_bundle: bool = False


class RagSlidesPptxResponse(BaseModel):
    path: str
    filename: str
    size_bytes: int
    audit_path: Optional[str] = None
    evidence_bundle_path: Optional[str] = None


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
    query: str,
    top_k: int,
    doc_id: Optional[str],
    version: Optional[str],
    specialty: Optional[str] = None,
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
    must_conditions: List[qmodels.FieldCondition] = []
    if doc_id:
        must_conditions.append(
            qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=doc_id)
            )
        )
    if specialty:
        normalized_specialty = specialty.strip()
        if normalized_specialty:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="specialty", match=qmodels.MatchValue(value=normalized_specialty)
                )
            )
    query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

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


def _gs_read_bytes(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    try:
        from google.cloud import storage

        _, _, rest = uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid gs:// URI: {uri}")
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        return blob.download_as_bytes()
    except Exception:
        return subprocess.check_output(["gsutil", "cat", uri])


def _load_assets_for_docs(
    *,
    doc_ids: List[str],
    version: str,
    specialty: Optional[str],
    max_assets: int = 40,
) -> List[Dict[str, Any]]:
    unique_doc_ids = [d for d in dict.fromkeys(doc_ids) if d]
    if not unique_doc_ids:
        return []

    db_path = _catalog_db_path()
    if not Path(db_path).exists():
        return []

    docs_meta = _load_documents_metadata(unique_doc_ids)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in unique_doc_ids)
        rows = conn.execute(
            "SELECT doc_id, gcs_assets_path FROM document_versions "
            f"WHERE version = ? AND doc_id IN ({placeholders})",
            [version, *unique_doc_ids],
        ).fetchall()
    finally:
        conn.close()

    specialty_norm = (specialty or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for row in rows:
        doc_id = str(row["doc_id"] or "")
        assets_uri = str(row["gcs_assets_path"] or "")
        raw_lines: List[str] = []
        local_enriched = Path("out/assets") / doc_id / version / "assets.enriched.jsonl"
        if local_enriched.exists():
            try:
                raw_lines = local_enriched.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                raw_lines = []
        if not raw_lines and assets_uri:
            try:
                raw_lines = _gs_read_bytes(assets_uri).decode("utf-8", errors="replace").splitlines()
            except Exception:
                raw_lines = []
        if not raw_lines:
            continue
        for line in raw_lines:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            render_path = str(obj.get("render_path") or "").strip()
            if not render_path or not Path(render_path).exists():
                continue
            obj_doc_id = str(obj.get("doc_id") or doc_id).strip()
            obj_specialty = str(obj.get("specialty") or "").strip().lower()
            if not obj_specialty:
                obj_specialty = str((docs_meta.get(obj_doc_id, {}) or {}).get("doc_specialty") or "").strip().lower()
            if specialty_norm and obj_specialty != specialty_norm:
                continue
            obj_type = str(obj.get("asset_type") or "").strip().lower()
            if obj_type not in {"table", "figure", "image", "chart"}:
                continue
            asset_id = str(obj.get("asset_id") or "").strip()
            if not asset_id:
                continue
            caption = _normalize_asset_caption(obj)
            table_rows = _load_table_rows_for_asset(obj)
            out.append(
                {
                    "asset_id": asset_id,
                    "asset_type": obj_type,
                    "doc_id": obj_doc_id,
                    "specialty": obj_specialty or None,
                    "caption": caption,
                    "page": obj.get("page"),
                    "locator": obj.get("locator"),
                    "bbox": obj.get("bbox"),
                    "render_path": render_path,
                    "files": obj.get("files") if isinstance(obj.get("files"), dict) else {},
                    "table_rows": table_rows,
                }
            )
            if len(out) >= max_assets:
                return out
    return out


def _load_table_rows_for_asset(asset: Dict[str, Any]) -> Optional[List[List[str]]]:
    if str(asset.get("asset_type") or "").strip().lower() != "table":
        return None
    files = asset.get("files") if isinstance(asset.get("files"), dict) else {}
    table_uri = str(files.get("table_uri") or "").strip()
    if not table_uri:
        return None
    try:
        payload = _gs_read_bytes(table_uri).decode("utf-8", errors="replace")
        obj = json.loads(payload)
    except Exception:
        return None
    rows = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    out: List[List[str]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        normalized = [" ".join(str(cell or "").replace("\n", " ").split()) for cell in row]
        if any(normalized):
            out.append(normalized)
    return out or None


def _score_asset_for_query(asset: Dict[str, Any], query: str) -> float:
    text = f"{asset.get('caption','')} {asset.get('asset_type','')}".lower()
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
    if not q_tokens:
        return 0.0
    hits = sum(1 for t in q_tokens if t in text)
    return float(hits)


_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_TABLE_INTENT_TOKENS = {
    "table",
    "tabella",
    "comparison",
    "compare",
    "confronto",
    "data",
    "dati",
    "values",
    "valori",
    "percent",
    "percentuale",
    "metric",
    "metrics",
    "outcome",
    "outcomes",
    "risultati",
    "result",
    "results",
}
_QUANT_INTENT_TOKENS = {
    "rate",
    "rates",
    "survival",
    "followup",
    "follow-up",
    "years",
    "months",
    "incidence",
    "prevalence",
}
_FIGURE_INTENT_TOKENS = {
    "figure",
    "figures",
    "image",
    "images",
    "chart",
    "charts",
    "diagram",
    "workflow",
    "radiograph",
    "radiographic",
    "xray",
    "x-ray",
    "cbct",
    "scan",
    "photo",
    "microscopy",
}
_VISUAL_EVIDENCE_CAPTION_TOKENS = {
    "result",
    "results",
    "outcome",
    "outcomes",
    "shows",
    "demonstrates",
    "comparison",
    "rate",
    "survival",
    "efficacy",
    "performance",
}
_NUMERIC_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|mm|cm|mg|g|kg|ml|l|months?|years?|yrs?)\b",
    re.IGNORECASE,
)


def _caption_is_unhelpful(caption: str) -> bool:
    text = (caption or "").strip()
    if not text:
        return True
    normalized = " ".join(text.lower().split())
    if _UUID_RE.search(normalized):
        return True
    return bool(re.fullmatch(r"(table|tabella|figure|image|chart|asset)\s*[\w-]*", normalized))


def _normalize_asset_caption(asset: Dict[str, Any]) -> str:
    raw_caption = str(asset.get("caption") or "").strip()
    if raw_caption and not _caption_is_unhelpful(raw_caption):
        return raw_caption
    asset_type = str(asset.get("asset_type") or "asset").strip().lower() or "asset"
    page = asset.get("page")
    try:
        page_no = int(page)
    except Exception:
        page_no = 0
    if page_no > 0:
        return f"{asset_type.title()} (source page {page_no})"
    return f"{asset_type.title()} from source"


def _tokenize_for_match(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _visual_relevant_for_slide(visual: Dict[str, Any], title: str, bullets: List[str]) -> bool:
    slide_tokens = _tokenize_for_match(" ".join([title] + bullets))
    if not slide_tokens:
        return False
    visual_caption = str(visual.get("caption") or "").strip()
    # Use only semantic surface (caption/type), not doc_id tokens, to avoid false-positive matches.
    visual_text = " ".join([visual_caption, str(visual.get("asset_type") or "")])
    visual_tokens = _tokenize_for_match(visual_text)
    overlap = slide_tokens.intersection(visual_tokens)
    if overlap:
        return True
    asset_type = str(visual.get("asset_type") or "").strip().lower()
    if asset_type == "table":
        has_table_intent = bool(slide_tokens.intersection(_TABLE_INTENT_TOKENS))
        return has_table_intent and not _caption_is_unhelpful(visual_caption)
    if asset_type in {"figure", "image", "chart"}:
        return bool(slide_tokens.intersection(_FIGURE_INTENT_TOKENS))
    return False


def _slide_has_table_fallback_intent(title: str, bullets: List[str]) -> bool:
    text = " ".join([title] + bullets).lower()
    slide_tokens = _tokenize_for_match(text)
    if slide_tokens.intersection(_TABLE_INTENT_TOKENS):
        return True
    if slide_tokens.intersection(_QUANT_INTENT_TOKENS):
        return True
    if re.search(r"\b\d+(?:[.,]\d+)?\s*%", text):
        return True
    numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    return len(numeric_tokens) >= 2


def _slide_has_figure_fallback_intent(title: str, bullets: List[str]) -> bool:
    slide_tokens = _tokenize_for_match(" ".join([title] + bullets))
    return bool(slide_tokens.intersection(_FIGURE_INTENT_TOKENS))


def _bullet_has_numeric_intent(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(r"\b\d+(?:[.,]\d+)?\s*%", lowered):
        return True
    if _NUMERIC_UNIT_RE.search(text or ""):
        return True
    tokens = _tokenize_for_match(lowered)
    if tokens.intersection(_QUANT_INTENT_TOKENS):
        return True
    return bool(re.search(r"\b\d+(?:[.,]\d+)?\b", lowered))


def _table_header_tokens(visual: Dict[str, Any]) -> set[str]:
    rows = visual.get("table_rows")
    if not isinstance(rows, list) or not rows:
        return set()
    first = rows[0]
    if not isinstance(first, list):
        return set()
    return _tokenize_for_match(" ".join(str(c or "") for c in first))


def _visual_match_tokens(visual: Dict[str, Any]) -> set[str]:
    caption = str(visual.get("caption") or "").strip()
    locator = visual.get("locator") if isinstance(visual.get("locator"), dict) else {}
    locator_bits = [
        str(locator.get("section_title") or ""),
        str(locator.get("section") or ""),
        str(locator.get("label") or ""),
    ]
    tokens = _tokenize_for_match(" ".join([caption, str(visual.get("asset_type") or "")] + locator_bits))
    return tokens.union(_table_header_tokens(visual))


def _caption_suggests_evidence(caption: str) -> bool:
    tokens = _tokenize_for_match(caption or "")
    if tokens.intersection(_VISUAL_EVIDENCE_CAPTION_TOKENS):
        return True
    if re.search(r"\bfigure\s*\d+\b", (caption or "").lower()):
        return True
    return False


def _infer_visual_role(
    title: str,
    bullets: List[str],
    visuals: List[Dict[str, Any]],
) -> str:
    if not visuals:
        return "illustrative"
    for v in visuals:
        if str(v.get("asset_type") or "").strip().lower() == "table":
            return "evidence"
    for v in visuals:
        if _caption_suggests_evidence(str(v.get("caption") or "")):
            return "evidence"
    # Last deterministic check: if slide text itself has strong quant/result intent, treat visuals as evidence.
    if _slide_has_table_fallback_intent(title, bullets):
        return "evidence"
    return "illustrative"


def _compute_bullet_visual_asset_ids(
    bullets: List[str],
    visuals: List[Dict[str, Any]],
) -> List[List[str]]:
    if not bullets:
        return []
    if not visuals:
        return [[] for _ in bullets]

    visuals_sorted = sorted(
        [v for v in visuals if str(v.get("asset_id") or "").strip()],
        key=lambda v: str(v.get("asset_id") or ""),
    )
    table_ids = [
        str(v.get("asset_id") or "")
        for v in visuals_sorted
        if str(v.get("asset_type") or "").strip().lower() == "table"
    ]
    out: List[List[str]] = []
    for bullet in bullets:
        links: List[str] = []
        bullet_tokens = _tokenize_for_match(bullet)
        if _bullet_has_numeric_intent(bullet) and table_ids:
            links.extend(table_ids[:1])
        for visual in visuals_sorted:
            aid = str(visual.get("asset_id") or "").strip()
            if not aid:
                continue
            overlap = bullet_tokens.intersection(_visual_match_tokens(visual))
            if overlap and aid not in links:
                links.append(aid)
        out.append(links)
    return out


def _select_table_fallback_for_slide(
    *,
    visual_candidates: List[Dict[str, Any]],
    title: str,
    bullets: List[str],
    sources: List[RagCitation],
    used_asset_ids: set[str],
) -> List[Dict[str, Any]]:
    source_doc_ids = {str(s.doc_id or "").strip() for s in sources if str(s.doc_id or "").strip()}
    scoped: List[Dict[str, Any]] = []
    for cand in visual_candidates:
        if str(cand.get("asset_type") or "").strip().lower() != "table":
            continue
        aid = str(cand.get("asset_id") or "").strip()
        if not aid or aid in used_asset_ids:
            continue
        scoped.append(cand)
    if not scoped:
        return []

    query = " ".join([title] + bullets).strip()
    ranked = sorted(
        scoped,
        key=lambda a: (
            0 if str(a.get("doc_id") or "").strip() in source_doc_ids else 1,
            -_score_asset_for_query(a, query),
            str(a.get("doc_id") or ""),
            str(a.get("asset_id") or ""),
        ),
    )
    return ranked[:1]


def _select_figure_fallback_for_slide(
    *,
    visual_candidates: List[Dict[str, Any]],
    title: str,
    bullets: List[str],
    sources: List[RagCitation],
    used_asset_ids: set[str],
) -> List[Dict[str, Any]]:
    source_doc_ids = {str(s.doc_id or "").strip() for s in sources if str(s.doc_id or "").strip()}
    scoped: List[Dict[str, Any]] = []
    for cand in visual_candidates:
        if str(cand.get("asset_type") or "").strip().lower() not in {"figure", "image", "chart"}:
            continue
        aid = str(cand.get("asset_id") or "").strip()
        if not aid or aid in used_asset_ids:
            continue
        scoped.append(cand)
    if not scoped:
        return []

    query = " ".join([title] + bullets).strip()
    ranked = sorted(
        scoped,
        key=lambda a: (
            0 if str(a.get("doc_id") or "").strip() in source_doc_ids else 1,
            -_score_asset_for_query(a, query),
            str(a.get("doc_id") or ""),
            str(a.get("asset_id") or ""),
        ),
    )
    return ranked[:1]


def _rank_visual_candidates_for_query(
    assets: List[Dict[str, Any]],
    query: str,
    *,
    max_assets: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        assets,
        key=lambda a: (
            -_score_asset_for_query(a, query),
            str(a.get("doc_id") or ""),
            str(a.get("asset_id") or ""),
        ),
    )
    if not ranked:
        return []
    figure_like = [a for a in ranked if str(a.get("asset_type") or "").strip().lower() in {"figure", "image", "chart"}]
    min_figures = min(max_assets, len(figure_like), 6)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for a in figure_like[:min_figures]:
        aid = str(a.get("asset_id") or "").strip()
        if not aid or aid in selected_ids:
            continue
        selected.append(a)
        selected_ids.add(aid)
    for a in ranked:
        if len(selected) >= max_assets:
            break
        aid = str(a.get("asset_id") or "").strip()
        if not aid or aid in selected_ids:
            continue
        selected.append(a)
        selected_ids.add(aid)
    return selected[:max_assets]


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
                    "doc_specialty": str(metadata_obj.get("specialty")).strip() if metadata_obj.get("specialty") else None,
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


def _missing_doc_citation_doc_ids(citations: List[RagCitation]) -> List[str]:
    missing: set[str] = set()
    for c in citations:
        if not c.doc_id:
            continue
        if str(c.doc_citation or "").strip():
            continue
        missing.add(c.doc_id)
    return sorted(missing)


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dumps_deterministic(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _source_locator_from_citation(src: Dict[str, Any]) -> Dict[str, int]:
    try:
        page_start = int(src.get("page_start", 0))
    except Exception:
        page_start = 0
    try:
        page_end = int(src.get("page_end", 0))
    except Exception:
        page_end = 0
    return {"page_start": page_start, "page_end": page_end}


def _build_bullet_sources_structured(
    bullet_source_item_ids: List[List[str]],
    sources: List[RagCitation],
) -> List[List[Dict[str, Any]]]:
    source_by_item: Dict[str, Dict[str, Any]] = {}
    for src in sources:
        item_id = str(src.item_id or "").strip()
        if not item_id or item_id in source_by_item:
            continue
        src_payload = src.model_dump() if hasattr(src, "model_dump") else src.dict()
        source_by_item[item_id] = {
            "item_id": item_id,
            "doc_id": str(src.doc_id or "").strip() or None,
            "locator": _source_locator_from_citation(src_payload),
            "score": float(src.score),
            "ref_id": str(src.ref_id or "").strip() or None,
        }

    out: List[List[Dict[str, Any]]] = []
    for item_ids in bullet_source_item_ids:
        bullet_items: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in item_ids:
            item_id = str(raw or "").strip()
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            source = source_by_item.get(item_id)
            if source:
                bullet_items.append(dict(source))
            else:
                bullet_items.append(
                    {
                        "item_id": item_id,
                        "doc_id": None,
                        "locator": {"page_start": 0, "page_end": 0},
                        "score": None,
                        "ref_id": None,
                    }
                )
        out.append(bullet_items)
    return out


def _render_path_exists(visual: Dict[str, Any]) -> bool:
    render_path = str(visual.get("render_path") or "").strip()
    return bool(render_path and Path(render_path).exists())


def _extract_request_for_audit(slide_plan: Dict[str, Any]) -> Dict[str, Any]:
    raw = slide_plan.get("request") if isinstance(slide_plan.get("request"), dict) else {}
    mode = str(raw.get("mode") or "").strip()
    if not mode:
        mode = "query" if str(raw.get("query") or "").strip() else "outline"
    return {
        "mode": mode,
        "query": str(raw.get("query") or "").strip() or None,
        "outline_title": str(raw.get("outline_title") or "").strip() or None,
        "version": str(raw.get("version") or "").strip() or None,
        "specialty": str(raw.get("specialty") or "").strip() or None,
        "timestamp": _utc_timestamp(),
        "env_profile": {
            "PPTX_KEYNOTE_SAFE": _env_flag("PPTX_KEYNOTE_SAFE"),
            "PPTX_KEYNOTE_KEEP_TABLES": _env_flag("PPTX_KEYNOTE_KEEP_TABLES"),
            "SLIDES_ENFORCE_NUMERIC_VISUAL_LINK": _env_flag("SLIDES_ENFORCE_NUMERIC_VISUAL_LINK"),
            "SLIDES_ALLOW_MISSING_ASSET": _env_flag("SLIDES_ALLOW_MISSING_ASSET"),
        },
    }


def _build_deck_audit_manifest(
    slide_plan: Dict[str, Any],
    deck_id: str,
    deck_path: Path,
) -> Dict[str, Any]:
    slides_raw = slide_plan.get("slides")
    if not isinstance(slides_raw, list):
        slides_raw = []

    slides_out: List[Dict[str, Any]] = []
    unique_docs: set[str] = set()
    unique_items: set[str] = set()
    unique_assets: set[str] = set()
    missing_assets: List[str] = []

    for idx, raw_slide in enumerate(slides_raw, start=1):
        slide = raw_slide if isinstance(raw_slide, dict) else {}
        title = str(slide.get("title") or "").strip()
        bullets = [str(x).strip() for x in (slide.get("bullets") or []) if str(x).strip()]
        bullet_item_ids = slide.get("bullet_source_item_ids")
        bullet_structured = slide.get("bullet_sources_structured")
        bullet_visual_ids = slide.get("bullet_visual_asset_ids")
        sources_raw = [s for s in (slide.get("sources") or []) if isinstance(s, dict)]
        visuals_raw = [v for v in (slide.get("visuals") or []) if isinstance(v, dict)]

        if not isinstance(bullet_item_ids, list):
            bullet_item_ids = [[] for _ in bullets]
        if not isinstance(bullet_structured, list):
            bullet_structured = []
        if not isinstance(bullet_visual_ids, list):
            bullet_visual_ids = [[] for _ in bullets]

        sources_by_item: Dict[str, Dict[str, Any]] = {}
        for src in sources_raw:
            item_id = str(src.get("item_id") or "").strip()
            if item_id and item_id not in sources_by_item:
                sources_by_item[item_id] = src

        bullets_out: List[Dict[str, Any]] = []
        for bullet_idx, bullet_text in enumerate(bullets):
            structured_for_bullet = (
                bullet_structured[bullet_idx]
                if bullet_idx < len(bullet_structured) and isinstance(bullet_structured[bullet_idx], list)
                else []
            )
            evidence_items: List[Dict[str, Any]] = []
            if structured_for_bullet:
                for item in structured_for_bullet:
                    if not isinstance(item, dict):
                        continue
                    item_id = str(item.get("item_id") or "").strip()
                    if not item_id:
                        continue
                    doc_id = str(item.get("doc_id") or "").strip()
                    locator = item.get("locator") if isinstance(item.get("locator"), dict) else {"page_start": 0, "page_end": 0}
                    evidence_items.append(
                        {
                            "item_id": item_id,
                            "doc_id": doc_id or None,
                            "locator": locator,
                            "score": item.get("score"),
                        }
                    )
            else:
                item_ids = (
                    bullet_item_ids[bullet_idx]
                    if bullet_idx < len(bullet_item_ids) and isinstance(bullet_item_ids[bullet_idx], list)
                    else []
                )
                for raw_item in item_ids:
                    item_id = str(raw_item or "").strip()
                    if not item_id:
                        continue
                    src = sources_by_item.get(item_id, {})
                    doc_id = str(src.get("doc_id") or "").strip()
                    evidence_items.append(
                        {
                            "item_id": item_id,
                            "doc_id": doc_id or None,
                            "locator": _source_locator_from_citation(src),
                            "score": src.get("score"),
                        }
                    )

            for ev in evidence_items:
                item_id = str(ev.get("item_id") or "").strip()
                doc_id = str(ev.get("doc_id") or "").strip()
                if item_id:
                    unique_items.add(item_id)
                if doc_id:
                    unique_docs.add(doc_id)

            visual_ids = (
                [str(x).strip() for x in bullet_visual_ids[bullet_idx] if str(x).strip()]
                if bullet_idx < len(bullet_visual_ids) and isinstance(bullet_visual_ids[bullet_idx], list)
                else []
            )
            bullets_out.append(
                {
                    "text": bullet_text,
                    "evidence_items": evidence_items,
                    "visual_asset_ids": visual_ids,
                }
            )

        visuals_out: List[Dict[str, Any]] = []
        for visual in visuals_raw:
            asset_id = str(visual.get("asset_id") or "").strip()
            doc_id = str(visual.get("doc_id") or "").strip()
            render_path = str(visual.get("render_path") or "").strip()
            exists = _render_path_exists(visual)
            if asset_id:
                unique_assets.add(asset_id)
            if doc_id:
                unique_docs.add(doc_id)
            if not exists and asset_id:
                missing_assets.append(asset_id)
            visuals_out.append(
                {
                    "asset_id": asset_id or None,
                    "type": str(visual.get("asset_type") or "").strip() or None,
                    "doc_id": doc_id or None,
                    "locator": visual.get("locator") if isinstance(visual.get("locator"), dict) else {},
                    "render_path": render_path or None,
                    "specialty": str(visual.get("specialty") or "").strip() or None,
                    "exists": exists,
                }
            )

        slides_out.append(
            {
                "slide_index": idx,
                "title": title,
                "bullets": bullets_out,
                "visual_role": str(slide.get("visual_role") or "").strip() or "illustrative",
                "visuals": visuals_out,
            }
        )

    unique_missing_assets = sorted(set(missing_assets))
    return {
        "deck_id": deck_id,
        "deck_path": str(deck_path),
        "request": _extract_request_for_audit(slide_plan),
        "slides": slides_out,
        "summary": {
            "unique_docs": sorted(unique_docs),
            "unique_items": sorted(unique_items),
            "unique_assets": sorted(unique_assets),
            "missing_asset_count": len(unique_missing_assets),
            "missing_assets": unique_missing_assets,
            "gates_applied": {
                "s2_bullet_source_required": True,
                "s2_structured_sources_embedded": True,
                "s5_visual_grounding": True,
                "s5_numeric_visual_link_enforced": _env_flag("SLIDES_ENFORCE_NUMERIC_VISUAL_LINK"),
            },
        },
    }


def _resolve_table_json_bytes(visual: Dict[str, Any]) -> Optional[bytes]:
    files = visual.get("files") if isinstance(visual.get("files"), dict) else {}
    local_table_path = str(files.get("table_local_path") or "").strip()
    if local_table_path and Path(local_table_path).exists():
        try:
            return Path(local_table_path).read_bytes()
        except Exception:
            pass
    table_uri = str(files.get("table_uri") or "").strip()
    if table_uri:
        try:
            return _gs_read_bytes(table_uri)
        except Exception:
            return None
    return None


def _write_evidence_bundle(
    *,
    bundle_path: Path,
    audit_payload: Dict[str, Any],
    slide_plan: Dict[str, Any],
) -> Path:
    slides_raw = slide_plan.get("slides")
    if not isinstance(slides_raw, list):
        slides_raw = []
    visuals_by_asset: Dict[str, Dict[str, Any]] = {}
    for raw_slide in slides_raw:
        slide = raw_slide if isinstance(raw_slide, dict) else {}
        visuals_raw = slide.get("visuals")
        if not isinstance(visuals_raw, list):
            continue
        for visual in visuals_raw:
            if not isinstance(visual, dict):
                continue
            asset_id = str(visual.get("asset_id") or "").strip()
            if not asset_id or asset_id in visuals_by_asset:
                continue
            visuals_by_asset[asset_id] = visual

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        def _writestr(name: str, payload: bytes) -> None:
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)

        _writestr("audit.json", _json_dumps_deterministic(audit_payload).encode("utf-8"))

        for asset_id in sorted(visuals_by_asset.keys()):
            visual = visuals_by_asset[asset_id]
            render_path = str(visual.get("render_path") or "").strip()
            if render_path and Path(render_path).exists():
                ext = Path(render_path).suffix.lower() or ".png"
                _writestr(f"assets/{asset_id}{ext}", Path(render_path).read_bytes())
            if str(visual.get("asset_type") or "").strip().lower() == "table":
                table_bytes = _resolve_table_json_bytes(visual)
                if table_bytes:
                    _writestr(f"tables/{asset_id}.json", table_bytes)
    return bundle_path


def _extract_slide_bullets(raw_slide: Dict[str, Any]) -> List[str]:
    bullets_raw = raw_slide.get("bullets")
    out: List[str] = []
    if not isinstance(bullets_raw, list):
        return out
    for b in bullets_raw:
        if isinstance(b, str):
            t = b.strip()
            if t:
                out.append(t)
            continue
        if isinstance(b, dict):
            t = str(b.get("text") or b.get("bullet") or "").strip()
            if t:
                out.append(t)
    return out


def _unique_item_ids(sources: List[RagCitation]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for s in sources:
        item_id = str(s.item_id or "").strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        out.append(item_id)
    return out


def _validate_slides_grounding(
    slides: List[RagSlidePlanItem], requested_specialty: Optional[str]
) -> None:
    errors: List[str] = []
    source_doc_ids: set[str] = set()
    enforce_numeric_visual_link = str(os.environ.get("SLIDES_ENFORCE_NUMERIC_VISUAL_LINK", "")).strip() == "1"

    for slide in slides:
        if len(slide.visuals) > 2:
            errors.append(f"slide {slide.slide_no}: more than 2 visuals")
        visual_by_id: Dict[str, Dict[str, Any]] = {}
        for v in slide.visuals:
            aid = str(v.get("asset_id") or "").strip()
            if not aid:
                errors.append(f"slide {slide.slide_no}: visual without asset_id")
                continue
            visual_by_id[aid] = v
            rpath = str(v.get("render_path") or "").strip()
            if not rpath:
                errors.append(f"slide {slide.slide_no}: visual {aid or '<missing>'} missing render_path")
            elif not Path(rpath).exists():
                errors.append(f"slide {slide.slide_no}: visual {aid or '<missing>'} render_path not found")
            v_specialty = str(v.get("specialty") or "").strip().lower()
            req_specialty = (requested_specialty or "").strip().lower()
            if req_specialty and v_specialty and v_specialty != req_specialty:
                errors.append(
                    f"slide {slide.slide_no}: visual {aid or '<missing>'} specialty mismatch "
                    f"({v_specialty} != {req_specialty})"
                )
            v_doc_id = str(v.get("doc_id") or "").strip()
            if v_doc_id:
                source_doc_ids.add(v_doc_id)

        if not slide.sources:
            errors.append(f"slide {slide.slide_no}: missing sources")
        if not slide.bullets:
            errors.append(f"slide {slide.slide_no}: missing bullets")
            continue
        if len(slide.bullet_source_item_ids) != len(slide.bullets):
            errors.append(
                f"slide {slide.slide_no}: bullet/source cardinality mismatch "
                f"({len(slide.bullet_source_item_ids)} vs {len(slide.bullets)})"
            )
            continue
        if len(slide.bullet_sources_structured) != len(slide.bullets):
            errors.append(
                f"slide {slide.slide_no}: bullet/structured-source cardinality mismatch "
                f"({len(slide.bullet_sources_structured)} vs {len(slide.bullets)})"
            )
            continue
        if len(slide.bullet_visual_asset_ids) != len(slide.bullets):
            errors.append(
                f"slide {slide.slide_no}: bullet/visual cardinality mismatch "
                f"({len(slide.bullet_visual_asset_ids)} vs {len(slide.bullets)})"
            )
            continue
        for i, item_ids in enumerate(slide.bullet_source_item_ids, start=1):
            if not item_ids:
                errors.append(f"slide {slide.slide_no} bullet {i}: empty source item_ids")
        for i, structured_items in enumerate(slide.bullet_sources_structured, start=1):
            if not structured_items:
                errors.append(f"slide {slide.slide_no} bullet {i}: empty structured evidence")
                continue
            for evidence in structured_items:
                if not isinstance(evidence, dict):
                    errors.append(f"slide {slide.slide_no} bullet {i}: structured evidence must be object")
                    continue
                if not str(evidence.get("item_id") or "").strip():
                    errors.append(f"slide {slide.slide_no} bullet {i}: structured evidence missing item_id")
                if not str(evidence.get("doc_id") or "").strip():
                    errors.append(f"slide {slide.slide_no} bullet {i}: structured evidence missing doc_id")
                locator = evidence.get("locator")
                if not isinstance(locator, dict):
                    errors.append(f"slide {slide.slide_no} bullet {i}: structured evidence missing locator")
        any_bullet_visual_link = False
        has_table_visual = any(
            str(v.get("asset_type") or "").strip().lower() == "table" for v in slide.visuals
        )
        for i, visual_ids in enumerate(slide.bullet_visual_asset_ids, start=1):
            numeric_bullet = _bullet_has_numeric_intent(slide.bullets[i - 1])
            if not isinstance(visual_ids, list):
                errors.append(f"slide {slide.slide_no} bullet {i}: visual links must be a list")
                continue
            clean_ids: List[str] = []
            for raw_id in visual_ids:
                aid = str(raw_id or "").strip()
                if not aid:
                    continue
                if aid not in visual_by_id:
                    errors.append(
                        f"slide {slide.slide_no} bullet {i}: linked visual {aid} not present on slide"
                    )
                    continue
                v = visual_by_id.get(aid, {})
                v_specialty = str(v.get("specialty") or "").strip().lower()
                req_specialty = (requested_specialty or "").strip().lower()
                if req_specialty and v_specialty and v_specialty != req_specialty:
                    errors.append(
                        f"slide {slide.slide_no} bullet {i}: linked visual {aid} specialty mismatch "
                        f"({v_specialty} != {req_specialty})"
                    )
                    continue
                if aid not in clean_ids:
                    clean_ids.append(aid)
            if clean_ids:
                any_bullet_visual_link = True
            if enforce_numeric_visual_link and numeric_bullet and not clean_ids and has_table_visual:
                errors.append(
                    f"slide {slide.slide_no} bullet {i}: numeric claim without linked visual evidence"
                )

        role = str(slide.visual_role or "").strip().lower()
        if role not in {"evidence", "illustrative"}:
            errors.append(f"slide {slide.slide_no}: invalid visual_role '{slide.visual_role}'")
        if slide.visuals and role != "illustrative" and not any_bullet_visual_link:
            errors.append(
                f"slide {slide.slide_no}: visuals present but no bullet visual links (set visual_role=illustrative or link evidence)"
            )
        for src in slide.sources:
            if src.doc_id:
                source_doc_ids.add(src.doc_id)

    normalized_specialty = (requested_specialty or "").strip().lower()
    if normalized_specialty and source_doc_ids:
        docs_meta = _load_documents_metadata(sorted(source_doc_ids))
        mismatches: List[str] = []
        for doc_id in sorted(source_doc_ids):
            found = str((docs_meta.get(doc_id, {}) or {}).get("doc_specialty") or "").strip().lower()
            if found != normalized_specialty:
                mismatches.append(f"{doc_id}:{found or '<missing>'}")
        if mismatches:
            errors.append(
                "specialty/source mismatch for docs: " + ", ".join(mismatches)
            )

    if errors:
        raise HTTPException(
            status_code=400,
            detail="Slide grounding validation failed: " + " | ".join(errors),
        )


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
    outline_json: str, target_slides: int, max_slides: int, visual_candidates_json: str
) -> str:
    return (
        "You are planning didactic dental slides from an outline.\n"
        "Return STRICT JSON only (no markdown, no code fences, no extra text).\n"
        "Use this schema exactly:\n"
        '{ "slides": [ { "slide_no": int, "title": str, "bullets": [str], "speaker_notes": str, "citations": ["S1","S2"], "visuals": ["asset_id_1","asset_id_2"] } ] }\n'
        f"Generate around {target_slides} slides and never exceed {max_slides}.\n"
        "Avoid duplicate slide titles/headings.\n"
        "Use only citation tokens already present in the outline.\n\n"
        "Grounding policy:\n"
        "- Every bullet must be supportable by the slide citations.\n"
        "- Visuals must be chosen ONLY from provided visual candidates.\n"
        "- Visuals are optional: use an empty list when no visual is clearly relevant.\n"
        "- Do NOT force one visual per slide; add tables only for data/comparison content.\n"
        "- Select visuals by semantic match with slide title/bullets and candidate caption.\n"
        "- Use at most 2 visuals per slide.\n"
        "- If evidence is thin, reduce ambition: fewer claims, more definitions/overview.\n\n"
        f"Outline JSON:\n{outline_json}\n"
        f"Visual candidates JSON:\n{visual_candidates_json}\n"
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
        specialty=req.specialty,
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
# curl -s -X POST http://localhost:8000/rag/query \
#   -H "Content-Type: application/json" \
#   -d '{"query":"What is the role of adhesive systems?","version_id":"v1-docai","top_k":3,"specialty":"endodontics"}' | jq
# curl -s -X POST http://localhost:8000/rag/answer \
#   -H "Content-Type: application/json" \
#   -d '{"query":"What is the role of adhesive systems?","top_k":3,"version":"v1-docai"}' | jq
# curl -s -X POST http://localhost:8000/rag/slides/plan \
#   -H "Content-Type: application/json" \
#   -d '{"query":"Adhesive systems overview","top_k":5,"version":"v1-docai","specialty":"endodontics","include_retrieved":true}' | jq


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

    items = retrieve_items(req.query, req.top_k, req.doc_id, req.version, req.specialty)
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

    items = retrieve_items(query_text, req.top_k, req.doc_id, req.version, req.specialty)
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
                specialty=req.specialty,
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
    citation_doc_ids = [c.doc_id for c in outline_used.citations if c.doc_id]
    visual_candidates = _load_assets_for_docs(
        doc_ids=[str(x) for x in citation_doc_ids if x],
        version=req.version,
        specialty=req.specialty,
        max_assets=40,
    )
    visual_candidates = _rank_visual_candidates_for_query(
        visual_candidates,
        query_text or outline_used.title,
        max_assets=20,
    )
    visual_candidate_ids = {str(a.get("asset_id") or "") for a in visual_candidates}
    visual_candidates_json = json.dumps(
        [
            {
                "asset_id": a.get("asset_id"),
                "asset_type": a.get("asset_type"),
                "doc_id": a.get("doc_id"),
                "caption": a.get("caption"),
                "page": a.get("page"),
            }
            for a in visual_candidates
        ],
        ensure_ascii=False,
    )
    outline_for_prompt = {
        "title": outline_used.title,
        "sections": [s.model_dump() for s in dedup_sections],
    }
    prompt = _make_slides_plan_prompt(
        json.dumps(outline_for_prompt, ensure_ascii=False),
        target_slides=target_slides,
        max_slides=req.max_slides,
        visual_candidates_json=visual_candidates_json,
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
    fallback_refs = sorted(citation_by_ref.keys(), key=lambda ref: int(ref[1:]) if ref[1:].isdigit() else 10**9)
    slides: List[RagSlidePlanItem] = []
    seen_titles: set[str] = set()
    used_visual_asset_ids: set[str] = set()
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
        bullets = _extract_slide_bullets(slide)
        sources = _map_ref_tokens_to_citations(citations, citation_by_ref)

        # Fallback for unsupported claims: keep slide conservative and tie it to available evidence.
        if not sources and fallback_refs:
            citations = [fallback_refs[0]]
            sources = _map_ref_tokens_to_citations(citations, citation_by_ref)
            if bullets:
                bullets = bullets[:2]
            else:
                bullets = [
                    f"Overview: {title}",
                    "Key definitions and high-level concepts from available evidence.",
                ]

        source_item_ids = _unique_item_ids(sources)
        bullet_source_item_ids = [list(source_item_ids) for _ in bullets]
        bullet_sources_structured = _build_bullet_sources_structured(
            bullet_source_item_ids,
            sources,
        )

        visuals_raw = slide.get("visuals", [])
        chosen_visual_ids: List[str] = []
        if isinstance(visuals_raw, list):
            for v in visuals_raw:
                aid = str(v or "").strip()
                if not aid or aid in chosen_visual_ids:
                    continue
                if aid not in visual_candidate_ids:
                    continue
                chosen_visual_ids.append(aid)
                if len(chosen_visual_ids) >= 2:
                    break
        visuals: List[Dict[str, Any]] = []
        for aid in chosen_visual_ids:
            for cand in visual_candidates:
                if str(cand.get("asset_id") or "") == aid:
                    if _visual_relevant_for_slide(cand, title, bullets):
                        visuals.append(cand)
                        used_visual_asset_ids.add(aid)
                    break

        if not visuals and _slide_has_table_fallback_intent(title, bullets):
            fallback_tables = _select_table_fallback_for_slide(
                visual_candidates=visual_candidates,
                title=title,
                bullets=bullets,
                sources=sources,
                used_asset_ids=used_visual_asset_ids,
            )
            for v in fallback_tables:
                aid = str(v.get("asset_id") or "").strip()
                if aid:
                    used_visual_asset_ids.add(aid)
                visuals.append(v)

        if not visuals and _slide_has_figure_fallback_intent(title, bullets):
            fallback_figures = _select_figure_fallback_for_slide(
                visual_candidates=visual_candidates,
                title=title,
                bullets=bullets,
                sources=sources,
                used_asset_ids=used_visual_asset_ids,
            )
            for v in fallback_figures:
                aid = str(v.get("asset_id") or "").strip()
                if aid:
                    used_visual_asset_ids.add(aid)
                visuals.append(v)

        slides.append(
            RagSlidePlanItem(
                slide_no=int(slide.get("slide_no", idx)),
                title=title,
                bullets=bullets,
                bullet_source_item_ids=bullet_source_item_ids,
                bullet_sources_structured=bullet_sources_structured,
                bullet_visual_asset_ids=_compute_bullet_visual_asset_ids(bullets, visuals),
                visual_role=_infer_visual_role(title, bullets, visuals),
                visuals=visuals,
                speaker_notes=(str(slide.get("speaker_notes")).strip() if slide.get("speaker_notes") is not None else None),
                citations=citations,
                sources=sources,
            )
        )

    missing_doc_ids: set[str] = set()
    for slide in slides:
        missing_doc_ids.update(_missing_doc_citation_doc_ids(slide.sources))
    if missing_doc_ids:
        missing_sorted = ", ".join(sorted(missing_doc_ids))
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing document citation metadata for one or more sources. "
                f"Set documents.metadata_json['citation'] for: {missing_sorted}"
            ),
        )

    _validate_slides_grounding(slides, req.specialty)

    for idx, slide in enumerate(slides, start=1):
        slide.slide_no = idx

    if not slides:
        raise HTTPException(status_code=500, detail="Slides plan JSON invalid: produced 0 slides")

    request_payload = {
        "mode": "outline" if req.outline is not None else "query",
        "query": query_text or None,
        "outline_title": outline_used.title if outline_used else None,
        "version": req.version,
        "specialty": req.specialty,
        "top_k": req.top_k,
        "max_sections": req.max_sections,
        "slides_per_section": req.slides_per_section,
        "max_slides": req.max_slides,
    }

    return RagSlidesPlanResponse(
        slides=slides,
        request=request_payload,
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

    deck_id = out_path.stem
    audit_payload = _build_deck_audit_manifest(req.slide_plan, deck_id=deck_id, deck_path=out_path)
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(_json_dumps_deterministic(audit_payload), encoding="utf-8")

    export_bundle = bool(req.evidence_bundle or _env_flag("PPTX_EXPORT_EVIDENCE_BUNDLE"))
    bundle_path: Optional[Path] = None
    if export_bundle:
        bundle_path = _write_evidence_bundle(
            bundle_path=out_path.with_suffix(".evidence.zip"),
            audit_payload=audit_payload,
            slide_plan=req.slide_plan,
        )

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
        audit_path=str(audit_path),
        evidence_bundle_path=str(bundle_path) if bundle_path else None,
    )
