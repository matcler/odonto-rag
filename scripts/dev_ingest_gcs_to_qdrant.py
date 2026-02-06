import os
import json
import uuid
import subprocess
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from google.cloud import storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# --- Catalogo DB (SQLite) ---
from odonto_rag.catalog.init_db import init_db
from odonto_rag.catalog.db import make_session_factory
from odonto_rag.catalog.models import Document, DocumentVersion


PROJECT_ID = os.environ.get("PROJECT_ID", "odontology-rag-slides")
LOCATION = os.environ.get("LOCATION", "europe-west1")

PARSED_BUCKET = os.environ["PARSED_BUCKET"]
PREFIX = os.environ.get("PARSED_PREFIX", "parsed/")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "odonto_chunks_gemini")

# Embeddings
MODEL = os.environ.get("VERTEX_EMBED_MODEL", "gemini-embedding-001")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

# Catalog DB path
CATALOG_DB_PATH = Path(os.getenv("CATALOG_DB_PATH", "data/catalog.db"))


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def get_access_token() -> str:
    out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    if not out:
        raise RuntimeError("Access token vuoto. Hai fatto 'gcloud auth login'?")
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    token = get_access_token()
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/"
        f"{MODEL}:predict"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"instances": [{"content": t} for t in texts]}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return [p["embeddings"]["values"] for p in data.get("predictions", [])]


def list_chunks_json_blobs() -> List[str]:
    client = storage.Client()
    blobs = client.list_blobs(PARSED_BUCKET, prefix=PREFIX)
    return [b.name for b in blobs if b.name.endswith("/chunks.json")]


def load_chunks_json_bytes(blob_name: str) -> bytes:
    client = storage.Client()
    b = client.bucket(PARSED_BUCKET).blob(blob_name)
    return b.download_as_bytes()


def parse_chunks_json(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def point_id_for(document_id: str, doc_version: str, chunk_index: int) -> str:
    # deterministic per (doc_id, version, chunk_index)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{doc_version}:{chunk_index}"))


def upsert_catalog_pending(
    session,
    *,
    doc_id: str,
    version: str,
    chunks_gcs_path: str,
    n_chunks: int,
    gcs_raw_path: str | None,
    gcs_parsed_prefix: str | None,
):
    doc = session.query(Document).filter_by(doc_id=doc_id).one_or_none()
    if doc is None:
        doc = Document(doc_id=doc_id, status="new")
        session.add(doc)

    if gcs_raw_path:
        doc.gcs_raw_path = gcs_raw_path
    if gcs_parsed_prefix:
        doc.gcs_parsed_prefix = gcs_parsed_prefix

    ver = session.query(DocumentVersion).filter_by(doc_id=doc_id, version=version).one_or_none()
    if ver is None:
        ver = DocumentVersion(
            doc_id=doc_id,
            version=version,
            gcs_chunks_path=chunks_gcs_path,
            n_chunks=n_chunks,
            ingest_status="pending",
        )
        session.add(ver)
    else:
        ver.gcs_chunks_path = chunks_gcs_path
        ver.n_chunks = n_chunks
        if ver.ingest_status != "ingested":
            ver.ingest_status = "pending"

    # non settiamo active_version finché non finisce l'upsert su Qdrant
    session.commit()
    return doc, ver


def mark_catalog_ingested(session, *, doc: Document, ver: DocumentVersion):
    ver.ingest_status = "ingested"
    ver.last_error = None
    doc.active_version = ver.version
    doc.status = "ingested"
    doc.last_error = None
    session.commit()


def mark_catalog_failed(session, *, doc: Document, ver: DocumentVersion, err: Exception):
    msg = str(err)
    ver.ingest_status = "failed"
    ver.last_error = msg[:2000]
    doc.status = "failed"
    doc.last_error = msg[:2000]
    session.commit()


def main():
    # init catalog DB
    engine = init_db(CATALOG_DB_PATH)
    Session = make_session_factory(engine)

    blobs = list_chunks_json_blobs()
    if not blobs:
        raise RuntimeError("Nessun chunks.json trovato")

    print(f"Found {len(blobs)} chunks.json files in gs://{PARSED_BUCKET}/{PREFIX}")

    q = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    total = 0

    for blob_name in blobs:
        chunks_gcs_path = f"gs://{PARSED_BUCKET}/{blob_name}"
        raw_bytes = load_chunks_json_bytes(blob_name)
        version = _sha16(raw_bytes)

        data = parse_chunks_json(raw_bytes)
        document_id = data["document_id"]
        chunks = data.get("chunks", [])
        source = data.get("source", {}) or {}

        # Optional: raw source pointer if present
        src_bucket = source.get("bucket")
        src_object_path = source.get("object_path")
        gcs_raw_path = f"gs://{src_bucket}/{src_object_path}" if (src_bucket and src_object_path) else None

        # parsed prefix is basically where chunks live
        parsed_prefix = os.path.dirname(blob_name)

        print(f"\nProcessing {chunks_gcs_path} | doc_id={document_id} | ver={version} | chunks={len(chunks)}")

        with Session() as session:
            doc, ver = upsert_catalog_pending(
                session,
                doc_id=document_id,
                version=version,
                chunks_gcs_path=chunks_gcs_path,
                n_chunks=len(chunks),
                gcs_raw_path=gcs_raw_path,
                gcs_parsed_prefix=parsed_prefix,
            )

        try:
            # Upsert points in batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                texts = [c.get("text", " ") or " " for c in batch]
                vectors = embed_texts(texts)

                points = []
                for j, (c, v) in enumerate(zip(batch, vectors)):
                    idx = i + j
                    payload = {
                        "doc_id": document_id,
                        "doc_version": version,
                        "chunk_index": idx,
                        "text": c.get("text"),
                        "section": c.get("section") or c.get("section_title"),
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "parsed_gcs_blob": blob_name,
                        "chunks_gcs_path": chunks_gcs_path,
                        "source_bucket": src_bucket,
                        "source_object_path": src_object_path,
                    }
                    points.append(
                        qm.PointStruct(
                            id=point_id_for(document_id, version, idx),
                            vector=v,
                            payload=payload,
                        )
                    )

                if points:
                    q.upsert(collection_name=QDRANT_COLLECTION, points=points)
                    total += len(points)
                    print(f"  upserted {total} points")
                    time.sleep(0.2)

            # Mark DB ingested
            with Session() as session:
                doc = session.query(Document).filter_by(doc_id=document_id).one()
                ver = session.query(DocumentVersion).filter_by(doc_id=document_id, version=version).one()
                mark_catalog_ingested(session, doc=doc, ver=ver)

        except Exception as e:
            with Session() as session:
                doc = session.query(Document).filter_by(doc_id=document_id).one()
                ver = session.query(DocumentVersion).filter_by(doc_id=document_id, version=version).one()
                mark_catalog_failed(session, doc=doc, ver=ver, err=e)
            raise

    print("\n✅ DONE")
    print("Total upserted points:", total)
    print(f"Catalog DB: {CATALOG_DB_PATH}")


if __name__ == "__main__":
    main()
