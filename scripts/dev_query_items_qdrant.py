#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from typing import List, Callable, TypeVar, Dict, Any

import requests
from qdrant_client import QdrantClient


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


def _collection_name(version: str, model: str) -> str:
    return f"odonto_items__{version}__{model}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Query items collection in Qdrant")
    ap.add_argument("--query", required=True)
    ap.add_argument("--version-id", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--collection", default="")
    args = ap.parse_args()

    project = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID") or ""
    location = os.environ.get("GCP_LOCATION") or os.environ.get("LOCATION") or ""
    if not project or not location:
        raise SystemExit("Missing GCP_PROJECT/PROJECT_ID or GCP_LOCATION/LOCATION")

    model = os.environ.get("VERTEX_EMBED_MODEL", "text-embedding-004")
    collection = args.collection or _collection_name(args.version_id, model)

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None

    vec = _embed_texts([args.query], project=project, location=location, model=model)[0]

    q = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    results = _with_retries(
        lambda: q.query_points(
            collection_name=collection,
            query=vec,
            limit=args.top_k,
            with_payload=True,
        )
    ).points

    for r in results:
        payload = r.payload or {}
        text = (payload.get("text") or "").replace("\n", " ")
        snippet = text[:160]
        locator = payload.get("locator")
        print(f"{r.id} | {r.score:.4f} | {snippet} | locator={locator}")


if __name__ == "__main__":
    main()
