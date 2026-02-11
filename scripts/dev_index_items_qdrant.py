#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from typing import List, Dict, Any, Callable, TypeVar

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from odonto_rag.catalog.db import make_engine, make_session_factory
from odonto_rag.catalog.models import DocumentVersion


T = TypeVar("T")


def _with_retries(fn: Callable[[], T], *, attempts: int = 3, base_sleep: float = 0.5) -> T:
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (i + 1))
    raise last_err  # type: ignore[misc]


def _get_access_token() -> str:
    try:
        out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    except FileNotFoundError as e:
        raise RuntimeError("gcloud not found. Install the Google Cloud SDK and run 'gcloud auth login'.") from e
    if not out:
        raise RuntimeError("Empty access token. Run 'gcloud auth login'.")
    return out


def _embed_texts(texts: List[str], *, project: str, location: str, model: str) -> List[List[float]]:
    token = _get_access_token()
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project}/locations/{location}/publishers/google/models/"
        f"{model}:predict"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"instances": [{"content": t} for t in texts]}
    def _do_request() -> Dict[str, Any]:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()

    data = _with_retries(_do_request)
    preds = data.get("predictions", [])
    if len(preds) != len(texts):
        raise SystemExit(
            f"Embedding count mismatch: expected {len(texts)} predictions, got {len(preds)}"
        )
    return [p["embeddings"]["values"] for p in preds]


def _gsutil_cat(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    # Prefer GCS API to avoid gsutil multiprocessing issues on macOS.
    try:
        from google.cloud import storage

        _, _, rest = uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid gs:// URI: {uri}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception:
        # Fallback to gsutil for environments without the GCS client.
        return subprocess.check_output(
            [
                "gsutil",
                "-o",
                "GSUtil:parallel_process_count=1",
                "-o",
                "GSUtil:parallel_thread_count=1",
                "cat",
                uri,
            ]
        )


def _load_items(items_uri: str) -> List[Dict[str, Any]]:
    raw = _gsutil_cat(items_uri)
    items: List[Dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _collection_name(version: str, model: str) -> str:
    return f"odonto_items__{version}__{model}"


def _ensure_collection(q: QdrantClient, name: str, vector_size: int) -> None:
    try:
        info = _with_retries(lambda: q.get_collection(name))
        existing = info.config.params.vectors
        if isinstance(existing, qm.VectorParams):
            size = existing.size
        else:
            size = list(existing.values())[0].size
        if size != vector_size:
            raise SystemExit(f"Collection {name} vector size {size} != {vector_size}")
        return
    except Exception:
        _with_retries(
            lambda: q.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            )
        )


def _point_id(item: Dict[str, Any], idx: int) -> str:
    doc_id = item.get("doc_id", "unknown")
    version_id = item.get("version_id", "unknown")
    raw = item.get("item_id")
    if raw:
        return f"{doc_id}:{version_id}:{raw}"
    locator = item.get("locator") or {}
    return (
        f"{doc_id}:{version_id}:{item.get('item_type','item')}:"
        f"{locator.get('page_start','')}:{locator.get('page_end','')}:{idx}"
    )


def _resolve_items_uri(doc_id: str, version: str, sqlite_path: str) -> str:
    engine = make_engine(sqlite_path)
    Session = make_session_factory(engine)
    with Session() as sess:
        dv = (
            sess.query(DocumentVersion)
            .filter(DocumentVersion.doc_id == doc_id, DocumentVersion.version == version)
            .one()
        )
        if not dv.gcs_items_path:
            raise SystemExit("document_versions.gcs_items_path is empty")
        return dv.gcs_items_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Index items.jsonl into Qdrant (items only)")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--version-id", required=True)
    ap.add_argument("--items-uri", default="")
    ap.add_argument("--sqlite", default=os.environ.get("SQLITE_PATH", "catalog.sqlite3"))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    args = ap.parse_args()

    project = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID") or ""
    location = os.environ.get("GCP_LOCATION") or os.environ.get("LOCATION") or ""
    if not project or not location:
        raise SystemExit("Missing GCP_PROJECT/PROJECT_ID or GCP_LOCATION/LOCATION")

    model = os.environ.get("VERTEX_EMBED_MODEL", "text-embedding-004")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None

    items_uri = args.items_uri or _resolve_items_uri(args.doc_id, args.version_id, args.sqlite)

    print("items_uri:", items_uri)
    items = _load_items(items_uri)
    if not items:
        raise SystemExit("No items loaded from items.jsonl")

    q = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    collection = _collection_name(args.version_id, model)

    total = 0
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        texts = [b.get("text", " ") or " " for b in batch]
        vectors = _embed_texts(texts, project=project, location=location, model=model)

        if i == 0:
            _ensure_collection(q, collection, vector_size=len(vectors[0]))

        points = []
        for idx, (item, vec) in enumerate(zip(batch, vectors)):
            locator = item.get("locator") or {}
            payload = {
                "doc_id": item.get("doc_id"),
                "version": item.get("version_id"),
                "item_id": item.get("item_id"),
                "item_type": item.get("item_type"),
                "locator": locator,
                "page_start": locator.get("page_start"),
                "page_end": locator.get("page_end"),
                "meta": item.get("meta"),
                "text": item.get("text"),
            }
            points.append(qm.PointStruct(id=_point_id(item, i + idx), vector=vec, payload=payload))

        if points:
            _with_retries(lambda: q.upsert(collection_name=collection, points=points))
            total += len(points)
            print(f"upserted {total}/{len(items)}")
            time.sleep(0.1)

    count = _with_retries(lambda: q.count(collection_name=collection, exact=True)).count
    print("collection:", collection)
    print("expected_n_items:", len(items))
    print("qdrant_count:", count)
    if count != len(items):
        raise SystemExit("Sanity check failed: count != n_items")


if __name__ == "__main__":
    main()
