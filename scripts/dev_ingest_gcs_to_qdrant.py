import os, json, uuid, subprocess, time
from typing import List, Dict, Any

import requests
from google.cloud import storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


PROJECT_ID = os.environ.get("PROJECT_ID", "odontology-rag-slides")
LOCATION = os.environ.get("LOCATION", "europe-west1")
PARSED_BUCKET = os.environ["PARSED_BUCKET"]
PREFIX = os.environ.get("PARSED_PREFIX", "parsed/")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "odonto_chunks_gemini")

MODEL = "gemini-embedding-001"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))


def get_access_token() -> str:
    out = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
    if not out:
        raise RuntimeError("Access token vuoto. Hai fatto gcloud auth login?")
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    token = get_access_token()
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/"
        f"{MODEL}:predict"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"instances": [{"content": t} for t in texts]}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return [p["embeddings"]["values"] for p in data.get("predictions", [])]


def list_chunks_json_blobs() -> List[str]:
    client = storage.Client()
    blobs = client.list_blobs(PARSED_BUCKET, prefix=PREFIX)
    return [b.name for b in blobs if b.name.endswith("/chunks.json")]


def load_chunks_json(blob_name: str) -> Dict[str, Any]:
    client = storage.Client()
    b = client.bucket(PARSED_BUCKET).blob(blob_name)
    return json.loads(b.download_as_text())


def point_id_for(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))


def main():
    blobs = list_chunks_json_blobs()
    if not blobs:
        raise RuntimeError("Nessun chunks.json trovato")

    print(f"Found {len(blobs)} chunks.json files")

    q = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    total = 0

    for blob_name in blobs:
        data = load_chunks_json(blob_name)
        document_id = data["document_id"]
        chunks = data["chunks"]
        source = data.get("source", {})

        print(f"\nProcessing {blob_name} | chunks={len(chunks)}")

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.get("text", " ") or " " for c in batch]
            vectors = embed_texts(texts)

            points = []
            for c, v in zip(batch, vectors):
                idx = int(c["chunk_index"])
                payload = {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "section": c.get("section"),
                    "text": c.get("text"),
                    "parsed_gcs_blob": blob_name,
                    "source_bucket": source.get("bucket"),
                    "source_object_path": source.get("object_path"),
                }
                points.append(
                    qm.PointStruct(
                        id=point_id_for(document_id, idx),
                        vector=v,
                        payload=payload,
                    )
                )

            q.upsert(collection_name=QDRANT_COLLECTION, points=points)
            total += len(points)
            print(f"  upserted {total} points")
            time.sleep(0.2)

    print("\n✅ DONE")
    print("Total upserted points:", total)


if __name__ == "__main__":
    main()

