# S1 Hardening Checklist

## Goal
Validate specialty segmentation end-to-end with at least 2 specialties indexed in the same collection.

## Preconditions
- Local Qdrant reachable on `http://localhost:6333`.
- `catalog.sqlite3` contains docs for `v1-docai`.
- API env is configured:
  - `QDRANT_URL`
  - `GCP_PROJECT`
  - `GCP_LOCATION`
  - `VERTEX_EMBED_MODEL`
  - `VERTEX_EMBED_LOCATION`
  - `VERTEX_LLM_MODEL` (required for `/rag/slides/plan`)
  - `VERTEX_LLM_LOCATION` (required for `/rag/slides/plan`)
  - `SQLITE_PATH`
  - `PYTHONPATH=src`

## 1) Ensure Specialty Metadata Exists
Use registration `--specialty`, or patch existing docs:

```bash
python3 scripts/catalog_set_specialty.py --doc-id article-adhesive-systems --specialty endodontics
python3 scripts/catalog_set_specialty.py --doc-id implant-macrodesign-10y --specialty implantology
```

Quick check:

```bash
sqlite3 catalog.sqlite3 "select doc_id, json_extract(metadata_json,'$.specialty') from documents order by doc_id;"
```

## 2) Reindex Docs to Backfill Qdrant Payload Specialty

```bash
python3 scripts/dev_index_items_qdrant.py --doc-id article-adhesive-systems --version-id v1-docai
python3 scripts/dev_index_items_qdrant.py --doc-id implant-macrodesign-10y --version-id v1-docai
python3 scripts/dev_index_items_qdrant.py --doc-id implant-macrodesign-osseo --version-id v1-docai
python3 scripts/dev_index_items_qdrant.py --doc-id implant-lovatto-2018 --version-id v1-docai
```

## 3) Query Smoke (Specialty Filter)

Expected:
- `specialty=implantology` -> only implantology docs
- `specialty=endodontics` -> only endodontics docs
- `specialty` omitted -> mixed corpus allowed

```bash
curl -s -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"implant macrodesign features","version_id":"v1-docai","top_k":5,"specialty":"implantology"}' | jq

curl -s -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"adhesive systems dentin","version_id":"v1-docai","top_k":5,"specialty":"endodontics"}' | jq

curl -s -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"dental materials and implants","version_id":"v1-docai","top_k":5}' | jq
```

## 4) Slides Plan Scoped Test
Expected:
- `/rag/slides/plan` with `specialty=implantology` returns sources/citations only from implantology docs.

```bash
curl -s -X POST http://127.0.0.1:8000/rag/slides/plan \
  -H "Content-Type: application/json" \
  -d '{"query":"implant macrodesign overview","version":"v1-docai","top_k":5,"specialty":"implantology","include_retrieved":true}' | jq
```

Validation tip:
- In response, inspect:
  - `retrieved[].doc_id`
  - `outline_used.citations[].doc_id`
  - `slides[].sources[].doc_id`
- They should all belong to docs with `metadata_json.specialty = "implantology"`.
