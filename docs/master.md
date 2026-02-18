# ODONTO-RAG WORKFLOW MEMORY (DEV HANDOFF) — v0.9

## Index
- [PURPOSE](#purpose)
- [PROJECT GOAL](#project-goal)
- [HIGH-LEVEL STATUS (as of v0.5)](#highlevel-status-as-of-v05)
- [CORE ARCHITECTURE (UPDATED — v0.5)](#core-architecture-updated-v05)
- [A. STORAGE (CANONICAL — v0.5)](#a-storage-canonical-v05)
- [B. CANONICAL DATA MODEL (NEW — v0.5)](#b-canonical-data-model-new-v05)
- [C. CATALOG DATABASE (STABLE — v0.5)](#c-catalog-database-stable-v05)
- [D. IMPLEMENTED DEV SCRIPTS (NEW — v0.5)](#d-implemented-dev-scripts-new-v05)
- [E. ENVIRONMENT (REFERENCE)](#e-environment-reference)
- [F. VERSIONING STRATEGY](#f-versioning-strategy)
- [G. NEXT STEP](#g-next-step)
- [UPDATE LOG — v0.6 (APPEND-ONLY)](#update-log-v06-appendonly)
- [What we did since v0.5](#what-we-did-since-v05)
- [Files created/updated in this step](#files-createdupdated-in-this-step)
- [Notes / Constraints](#notes-constraints)
- [NEXT STEP (v0.6 → v0.7 plan)](#next-step-v06-v07-plan)
- [UPDATE LOG — v0.7 (APPEND-ONLY)](#update-log-v07-appendonly)
- [What we did since v0.6](#what-we-did-since-v06)
- [Learnings / anomalies](#learnings-anomalies)
- [NEXT STEP](#next-step)
- [Follow-up fixes + rerun](#followup-fixes-rerun)
- [UPDATE LOG — v0.9 (APPEND-ONLY)](#update-log-v09-appendonly)
- [Architecture recap (short)](#architecture-recap-short)
- [STEP 7 implemented](#step-7-implemented)
- [Env vars needed](#env-vars-needed)
- [Next step](#next-step)
- [Addendum — 2026-02-11](#addendum-20260211)
- [Addendum — 2026-02-11 (STEP 8 MVP)](#addendum-20260211-step-8-mvp)
- [UPDATE LOG — v0.10 (APPEND-ONLY)](#update-log-v010-appendonly)
- [STEP 9 hardening completed (XOR + robustness)](#step-9-hardening-completed-xor-robustness)
- [Smoke tests (local)](#smoke-tests-local)
- [Files touched](#files-touched)
- [Next step](#next-step)
- [UPDATE LOG — v0.12 (APPEND-ONLY)](#update-log-v012-appendonly)
- [Scope note (chat scoping)](#scope-note-chat-scoping)
- [STEP 10 — PPTX generation from slide plan (DONE)](#step-10-pptx-generation-from-slide-plan-done)
- [STEP 11 — Figures & tables in PPTX (NEXT)](#step-11-figures-tables-in-pptx-next)
- [UPDATE LOG — v0.13 (APPEND-ONLY)](#update-log-v013-appendonly)
- [STEP S1 — Macro-specialty segmentation (DESIGN + IMPLEMENTATION PLAN)](#step-s1-macrospecialty-segmentation-design-implementation-plan)
- [UPDATE LOG — v0.14 (APPEND-ONLY)](#update-log-v014-appendonly)
- [GCS RAW ARCHITECTURE — Specialties folder convention (EN)](#gcs-raw-architecture-specialties-folder-convention-en)
- [NEXT IMPLEMENTATION STEPS (S1 execution checklist)](#next-implementation-steps-s1-execution-checklist)
- [UPDATE LOG — v0.15 (APPEND-ONLY)](#update-log-v015-appendonly)
- [STEP S1 — Macro-specialty segmentation (SMOKE TEST END-TO-END)](#step-s1-macrospecialty-segmentation-smoke-test-endtoend)
- [NEXT STEPS (S1 hardening)](#next-steps-s1-hardening)


## PURPOSE
This file is used at the beginning of each new development chat to restore
full context, decisions, and workflow for the odontoiatric RAG system.
Update this file at the end of every chat before switching to a new one.

## PROJECT GOAL
Create a RAG-based system focused on DIDACTIC PRESENTATIONS that:
- Ingests odontoiatric documents (PDF, PPTX, DOCX; later VIDEO)
- Understands them semantically (RAG)
- Generates structured teaching material (slides, courses, decks)
- Target audience: students, clinicians, colleagues
- Output: teaching-oriented, source-backed slides (ECM / academic)

# HIGH-LEVEL STATUS (as of v0.5)
✔ Canonical storage layout defined (GCS)
✔ Catalog DB schema redesigned and working (SQLite)
✔ Document/version registration pipeline working
✔ GCS writer for items/assets working (tested)
⏳ Document AI real extraction (NEXT STEP)
⏳ Qdrant indexing of new model (items/assets)
⏳ Runtime RAG + FastAPI + UI

# CORE ARCHITECTURE (UPDATED — v0.5)
RAW CONTENT (PDF / PPTX / DOCX / VIDEO)
  → GCS RAW bucket
  → extractor (Document AI / others)
  → GCS PARSED bucket (canonical layout)
      - items.jsonl
      - assets.jsonl
      - assets/images/
      - assets/tables/
      - raw/extractor_output.json
  → SQLite catalog (metadata only)
  → Qdrant (vector search)
  → RAG runtime
  → Slide / course generator
  → PPTX builder
  → UI

# A. STORAGE (CANONICAL — v0.5)
Google Cloud Storage (GCS) is the SOURCE OF TRUTH.

RAW bucket:
- odonto-raw-docs-matcler/
- Original source files only (immutable)

PARSED bucket:
- odonto-parsed-matcler/

Canonical layout (per doc + version):
parsed/<doc_id>/<version_id>/
  items.jsonl
  assets.jsonl
  assets/images/
  assets/tables/
  raw/extractor_output.json

Design rules:
- items/assets are NEVER stored in SQLite
- GCS paths are deterministic and versioned
- Safe to re-ingest without data loss

# B. CANONICAL DATA MODEL (NEW — v0.5)

1) items.jsonl (NDJSON)
- One record = one semantic teaching unit
- Used for embeddings + retrieval

Core fields:
- item_id
- doc_id
- version_id
- doc_type
- item_type
- text
- section / title
- locator (page, slide, time, bbox)
- source (uri, publisher, year)
- tags
- meta

2) assets.jsonl (NDJSON)
- One record = one non-text asset

Core fields:
- asset_id
- doc_id
- version_id
- asset_type
- caption
- locator (page, slide, time, bbox)
- files (image_uri, table_uri)
- table (optional)
- tags
- meta

# C. CATALOG DATABASE (STABLE — v0.5)
Type:
- SQLite (local-first, admin/query only)

Purpose:
- Track documents and versions
- Drive ingest, query, UI
- NEVER store full text

Tables:
1) documents
2) document_versions

# D. IMPLEMENTED DEV SCRIPTS (NEW — v0.5)

1) scripts/dev_register_doc_version.py
- Registers doc + version
- Computes canonical GCS paths

2) scripts/dev_write_sample_extract.py
- Writes items.jsonl / assets.jsonl to GCS
- Updates DB counters
- Uses gsutil

# E. ENVIRONMENT (REFERENCE)
.env.local variables include:
- GCP_PROJECT=odontology-rag-slides
- GCP_LOCATION=europe-west1
- DOCAI_LOCATION=eu
- RAW_BUCKET=odonto-raw-docs-matcler
- PARSED_BUCKET=odonto-parsed-matcler
- RAW_PREFIX=test/articles/
- SQLITE_PATH=catalog.sqlite3
- QDRANT_URL=http://localhost:6333
- VERTEX_EMBED_MODEL=text-embedding-004

# F. VERSIONING STRATEGY
- New version for every ingest logic change
- documents.active_version defines production view
- Old versions kept for audit

# G. NEXT STEP
STEP 4 — REAL PDF INGEST WITH DOCUMENT AI LAYOUT PARSER
- Use Layout Parser processor
- Generate real items.jsonl / assets.jsonl
- Update DB counts and status
- Preserve raw extractor output

# UPDATE LOG — v0.6 (APPEND-ONLY)
Date: 2026-02-06 (Europe/Rome)

## What we did since v0.5
1) Git/GitHub versioning
- Initialized / prepared the project for GitHub versioning.
- Confirmed .gitignore strategy: keep repo source-only; ignore runtime/state:
  - .env.local, *.sqlite3, qdrant/, qdrant.tar.gz, data/, storage/, out/, outputs/
- Confirmed that Qdrant (local index/state) must NOT be committed; only code + schemas + scripts are versioned.
- Confirmed that templates PPTX used for early tests were removed from the repo (tracked as deletions):
  - templates/chairside_test_template.pptx
  - templates/dark_master.pptx
  Rationale: templates are not part of the ingest/RAG foundation; reintroduce later with a dedicated commit and stable template strategy.

2) Document AI Layout ingest — moved from “sample extract” to real extractor script
- Added a real Document AI Layout Parser ingest script (new file):
  - scripts/dev_docai_layout_ingest.py
- Converted the script from gcsfs-based I/O to gsutil-based I/O immediately (robustness + consistency with prior work):
  - Read PDF bytes: gsutil cat gs://...
  - Write JSON/NDJSON outputs: gsutil cp - gs://...
- The script:
  - Reads documents.gcs_raw_path (or --raw-uri override)
  - Calls Document AI Layout Parser (ProcessDocument, raw_document)
  - Writes raw extractor dump to:
      parsed/<doc_id>/<version_id>/raw/extractor_output.json
    (path generated via ParsedLayout + gcs_uri)
  - Writes:
      items.jsonl  → one item per page (item_type="page")
      assets.jsonl → table assets (asset_type="table") + per-table JSON payload saved under assets/tables/<asset_id>.json
  - Updates SQLite document_versions:
      n_items, n_assets, ingest_status="ingested"

3) Requirements updated for STEP 4
- requirements.txt (root) now includes only what is needed for this step:
  - google-cloud-documentai
  - python-pptx
- Removed gcsfs dependency after switching to gsutil.

## Files created/updated in this step
- requirements.txt (new/updated)
- scripts/dev_docai_layout_ingest.py (new) — gsutil-based Document AI Layout ingest
- (Repo hygiene) .gitignore updated/validated to keep runtime out of git

## Notes / Constraints
- The gsutil approach assumes Google Cloud SDK is installed and authenticated on the dev machine.
- Document AI credentials and target processor are provided via:
  - --project / --location / --processor-id
  and/or env:
  - GCP_PROJECT, DOCAI_LOCATION, DOCAI_PROCESSOR_ID
- Current extraction scope:
  - Items: page-level text units
  - Assets: tables only
  - Figures/images extraction is planned next (from Document AI page image / rendered pages / figure detection strategy).

# NEXT STEP (v0.6 → v0.7 plan)
STEP 5 — Run real DocAI ingest end-to-end on the test PDFs + validate outputs
1) Ensure DB registration exists for doc_id + version_id (document_versions row has items_uri/assets_uri)
2) Run:
   python scripts/dev_docai_layout_ingest.py --doc-id <...> --version-id v2-docai --processor-id <...>
3) Validate in GCS:
   - items.jsonl exists and has expected number of lines (≈ number of pages)
   - assets.jsonl exists and contains table assets if tables are detected
   - raw/extractor_output.json exists
   - assets/tables/<asset_id>.json exists for each detected table
4) Decide chunking granularity for items:
   - page-level vs paragraph/section-level items for better retrieval
   - update script accordingly (still NDJSON, but item_type changes: paragraph/section)
5) After content is stable:
   - Add Qdrant indexing for items (and optionally assets captions/metadata)
   - Establish collection naming convention including version + embed model
   - Implement runtime query path (RAG): embed query → Qdrant search → assemble context → LLM answer

# UPDATE LOG — v0.7 (APPEND-ONLY)
Date: 2026-02-09

## What we did since v0.6
1) Script fixes for Document AI ingest
- Updated _text_from_anchor() to treat missing end_index as end=len(full_text).
- Added debug prints after Document AI returns the document:
  - pdf_bytes_len
  - pages_count
  - text_len

2) Dependencies installed (local)
- Installed runtime deps needed to run the ingest:
  - google-cloud-documentai, python-pptx, gcsfs (from requirements.txt)
  - sqlalchemy (missing from requirements.txt but required by catalog DB)

3) Ran real DocAI ingest (article-adhesive-systems, v1-docai)
- Command used (with env):
  GCP_PROJECT=odontology-rag-slides DOCAI_LOCATION=eu PYTHONPATH=src \
  python3 scripts/dev_docai_layout_ingest.py \
    --doc-id article-adhesive-systems \
    --version-id v1-docai \
    --processor-id cefc5e7bf97d1e2c
- Debug output:
  - pdf_bytes_len: 1627148
  - pages_count: 0
  - text_len: 0
- Script completed and wrote GCS outputs, but no items/assets were produced.

4) GCS validation
- Parsed outputs present:
  - items.jsonl (size 1 byte; empty)
  - assets.jsonl (size 1 byte; empty)
  - raw/extractor_output.json (size ~450 KB)
- No assets/tables/*.json files were created (no tables detected).

## Learnings / anomalies
- Document AI returned zero pages and empty text for this PDF, so items.jsonl is empty.
- This fails the “non-trivial items.jsonl with JSON lines” requirement.

## NEXT STEP
1) Inspect raw/extractor_output.json to see why pages/text are missing.
2) Verify processor_id and DOCAI_LOCATION are correct for the Layout processor.
3) Re-run ingest after confirmation, then re-validate items.jsonl/ assets.jsonl.

--- Addendum (2026-02-09) ---
## Follow-up fixes + rerun
1) Ingest script update (layout parsing)
- Added recursive traversal of document_layout blocks to capture nested text and table blocks.
- Added table extraction for layout table_block rows.

2) Re-ran ingest (article-adhesive-systems, v1-docai)
- Debug output:
  - pdf_bytes_len: 1627148
  - pages_count: 0
  - text_len: 0
- Output counts:
  - items: 395
  - assets: 7

3) GCS validation (non-trivial outputs now present)
- items.jsonl size: 181,057 bytes (contains JSON lines)
- assets.jsonl size: 2,304 bytes
- assets/tables/*.json: 7 files

# UPDATE LOG — v0.9 (APPEND-ONLY)
Date: 2026-02-11 (Europe/Rome)

## Architecture recap (short)
- FastAPI app in src/odonto_rag/api/rag_app.py with /healthz, /rag/query, /rag/answer.
- Qdrant collections named odonto_items__{version}__{embed_model}.
- Vertex AI embeddings via REST (text-embedding-004) and LLM via vertexai GenerativeModel.
- Catalog metadata in SQLite (catalog.sqlite3) with GCS-backed items.jsonl.

## STEP 7 implemented
- Added retrieval helper used by /rag/query and new /rag/answer endpoint.
- /rag/answer builds context with [S#] tokens + page ranges and generates didactic answers with citations.
- Structured citations extracted from [S#] tokens and mapped to page_start/page_end.
- Qdrant dev index/query scripts updated for consistent collection naming, safe upsert IDs,
  payload page_start/page_end, default QDRANT_URL, and small retry/backoff.

## Env vars needed
- QDRANT_URL (default http://localhost:6333)
- QDRANT_API_KEY (optional)
- GCP_PROJECT or PROJECT_ID
- GCP_LOCATION or LOCATION
- VERTEX_EMBED_MODEL (default text-embedding-004)
- VERTEX_LLM_MODEL
- SQLITE_PATH (for dev_index_items_qdrant.py; default catalog.sqlite3)

## Next step
- Run a local /rag/answer smoke test and confirm citations map to correct pages; then add lightweight
  evaluation prompts for coverage and grounding.

## Addendum — 2026-02-11
- LLM model availability may differ by region; embeddings can stay in europe-west1 while LLM may require us-central1.
- New env vars for split locations:
  - VERTEX_EMBED_LOCATION (defaults to GCP_LOCATION/LOCATION)
  - VERTEX_LLM_LOCATION (defaults to GCP_LOCATION/LOCATION)
- Recommended dev smoke-test LLM model: gemini-2.0-flash-001 in us-central1 (or any available Gemini model with -001/-002 suffix).

## Addendum — 2026-02-11 (STEP 8 MVP)
- Added new POST endpoint `/rag/outline` in `src/odonto_rag/api/rag_app.py`.
- Request model: `RagOutlineRequest` with `query/topic`, `top_k` (default 25), optional `doc_id`, required `version`, `max_sections` (default 10), and `include_retrieved` (default false).
- Retrieval path reuses existing `retrieve_items(query, top_k, doc_id, version)` and builds `[S#]` context snippets (about 500 chars each).
- Added lightweight dedup before context assembly: repeated `header` items are collapsed by case-insensitive identical text.
- LLM prompt requires strict JSON-only outline output; parse failures return HTTP 500 including the first 200 chars of model output.
- Response model: `RagOutlineResponse` with `title`, structured `sections`, resolved `citations` (`RagCitation` mapped from `S#` tokens to item/page ranges), and optional `retrieved` payload.
- Env vars reused (no new env required): `GCP_PROJECT`/`PROJECT_ID`, `VERTEX_LLM_LOCATION` (or `GCP_LOCATION`/`LOCATION` fallback), `VERTEX_LLM_MODEL`, plus retrieval-side vars (`VERTEX_EMBED_LOCATION`, `VERTEX_EMBED_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY`).
- outline parser now strips ```json fences

# UPDATE LOG — v0.10 (APPEND-ONLY)
Date: 2026-02-12

## STEP 9 hardening completed (XOR + robustness)
- Endpoint `/rag/slides/plan` hardened:
  - Enforced XOR between `outline` and `query` at route level:
    - If both provided -> HTTP 400 with detail: "Provide only one of outline or query, not both"
    - If neither provided -> HTTP 400 with detail: "Provide either 'outline' or 'query'"
  - Keeps successful paths unchanged (query-only and outline-only return HTTP 200 with slide plan JSON).
- Added more robust JSON cleanup for LLM outputs:
  - `_strip_json_fences` now handles BOM and extracts fenced ```json blocks via regex before parsing.
- Added logging for outline parse failures:
  - Logs first ~500 chars of the raw model output when JSON parse fails to aid debugging.
- Note: `/rag/slides/plan` (query path) depends on retrieval; if Qdrant is not reachable (e.g., Docker daemon stopped),
  the endpoint can error. Ensure Qdrant is running and `QDRANT_URL=http://localhost:6333`.

## Smoke tests (local)
- Health:
  - `curl -s http://localhost:8000/healthz | jq`
- Slides plan (query-only) -> 200:
  - `curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST http://localhost:8000/rag/slides/plan -H "Content-Type: application/json" -d '{"query":"Adhesive systems","version":"v1-docai"}'`
- Slides plan (missing both) -> 400:
  - `curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST http://localhost:8000/rag/slides/plan -H "Content-Type: application/json" -d '{"version":"v1-docai"}'`
- Slides plan (outline + query) -> 400 with detail:
  - `curl -s -X POST http://localhost:8000/rag/slides/plan -H "Content-Type: application/json" -d '{"outline":{...},"query":"x","version":"v1-docai"}' | jq`

## Files touched
- src/odonto_rag/api/rag_app.py
- Master/ODONTO_RAG_WORKFLOW_MEMORY_v0.9.txt

## Next step
- STEP 10: PPTX generation from slide plan output.

# UPDATE LOG — v0.12 (APPEND-ONLY)
Date: 2026-02-12

## Scope note (chat scoping)
- This chat is considered STEP 10 only.
- Any auxiliary improvements made while delivering STEP 10 (e.g., citation formatting refinements) are treated as STEP 10 implementation details, not a separate milestone in this workflow log.

## STEP 10 — PPTX generation from slide plan (DONE)
Goal
- Generate a .pptx deck from the strict JSON slide plan output of POST /rag/slides/plan.

Delivered
- POST /rag/slides/pptx added to FastAPI runtime (local-first).
- PPTX generation implemented with python-pptx under src/odonto_rag/deck/.
- Deterministic output:
  - stable slide order and rendering (title + bullets),
  - stable filename generation (same slide_plan -> same output filename/path),
  - output directory configurable via env (PPTX_OUTPUT_DIR).
- Citations preserved:
  - slide footer contains citation tokens (S#) and optional document-level citation string,
  - speaker notes include a “Sources” section mapping S# to page ranges (and other available identifiers).

Smoke tests (validated)
- Generate slide plan -> generate PPTX -> verify file exists and is non-empty.
- Optional: unzip deck.pptx to confirm ppt/slides/*.xml and ppt/notesSlides/*.xml presence.

## STEP 11 — Figures & tables in PPTX (NEXT)
Goal
- Include figures/tables from source PDFs into generated PPTX slides (initially minimal and deterministic).

First check
- Verify ingest/layout pipeline already captures figure/table blocks and bounding boxes (Document AI layout -> canonical layout).
- Confirm where figure/table metadata is stored (layout JSON, catalog, Qdrant payload) and how to map it to doc_id + page.

Planned approach (minimal)
- During PPTX generation, optionally add one visual per slide:
  - Use slide.sources[].doc_id + page_start/page_end to search for candidate FIGURE/TABLE blocks in the canonical layout.
  - Select deterministically (largest area or first in reading order).
  - Render PDF page to image and crop by bbox; embed image on slide.
- Keep citations and speaker notes behavior unchanged.

Test set expansion (for STEP 11)
- Prepare 2–3 articles (PDFs) that include both tables and figures.
- Ensure ingest produces layout artifacts for these PDFs and that RAG retrieval can cite pages containing visuals.

# UPDATE LOG — v0.13 (APPEND-ONLY)
Date: 2026-02-12 (Europe/Rome)

## STEP S1 — Macro-specialty segmentation (DESIGN + IMPLEMENTATION PLAN)
Goal
- Support multi-specialty corpora by grouping documents by macro-specialty (e.g., endodonzia, implantologia, conservativa).
- Enable retrieval filtering by specialty.
- Ensure slide plans + PPTX generation can target a single specialty without cross-specialty bleed.

Design decision (storage model)
- Canonical source: documents.metadata_json["specialty"].
  - Current representation: a single string (e.g., "endodonzia").
  - Forward-compatible: allow list[str] ("specialties") later; retrieval filter will accept a single specialty value for now.
- Retrieval/index surface: each Qdrant chunk payload MUST include:
  - payload["specialty"] = "<specialty-string>"
  so filtering happens at vector search time.

Derivation rules (minimal + explicit)
- Primary: explicit CLI parameter during registration/ingest (recommended).
- Fallback: derive from GCS raw path convention:
    gs://<RAW_BUCKET>/<RAW_PREFIX>/<specialty>/...
  Example:
    raw/endodonzia/<file>.pdf  -> specialty="endodonzia"
- If neither available, specialty is omitted (treated as “global corpus” / untagged).

Implementation plan (surgical changes)
1) Catalog metadata persistence
   - Extend scripts/dev_register_doc_version.py:
     - add optional --specialty (string)
     - if missing, derive from --raw-uri (first folder after RAW_PREFIX if present)
     - upsert documents.metadata_json["specialty"] (preserve existing metadata keys such as "citation")
   - (Optional) add a small helper script scripts/catalog_set_specialty.py mirroring catalog_set_citation.py:
     - set specialty for an existing doc_id without re-registration.

2) Qdrant indexing payload extension
   - Update scripts/dev_index_items_qdrant.py:
     - read documents.metadata_json["specialty"] for the target doc_id once (SQLite)
     - add payload["specialty"] = specialty if present
   - Note: filtering requires re-index (upsert) for existing points to carry the new payload field.

3) Runtime retrieval filter
   - Extend retrieve_items(query, top_k, doc_id, version, specialty):
     - if specialty provided, add must condition on payload field "specialty" == value
     - if doc_id also provided, both must conditions apply
   - Extend request models/endpoints:
     - POST /rag/query: accept optional specialty (string)
     - POST /rag/answer: accept optional specialty (string)
     - POST /rag/outline: accept optional specialty (string)
     - POST /rag/slides/plan: accept optional specialty (string) and propagate to outline/retrieval
   - Default behavior unchanged when specialty is omitted.

4) Slide plan propagation
   - /rag/slides/plan:
     - query path: pass specialty into outline generation, which passes it into retrieval
     - outline-provided path: no retrieval is done unless include_retrieved is true; in that case ensure retrieved items came from the same specialty (enforced by retrieval filter)

5) Test dataset expansion (GCS layout)
- Prepare at least 2 PDFs per specialty:
  raw/
    endodonzia/
    implantologia/
    conservativa/
- For each doc:
  - Register with --specialty (or rely on path derivation)
  - Run DocAI ingest for a version (e.g. v1-docai)
  - Index into Qdrant for that version

Smoke tests (local)
- Index (repeat per doc_id):
  GCP_PROJECT=... GCP_LOCATION=... VERTEX_EMBED_MODEL=text-embedding-004 QDRANT_URL=http://localhost:6333 SQLITE_PATH=catalog.sqlite3 PYTHONPATH=src \
  python scripts/dev_index_items_qdrant.py --doc-id <doc_id> --version-id v1-docai

- Retrieval without filter:
  curl -s -X POST http://localhost:8000/rag/query -H "Content-Type: application/json" \
    -d '{"query":"...", "version_id":"v1-docai", "top_k":5}' | jq

- Retrieval with specialty filter:
  curl -s -X POST http://localhost:8000/rag/query -H "Content-Type: application/json" \
    -d '{"query":"...", "version_id":"v1-docai", "top_k":5, "specialty":"endodonzia"}' | jq

- Slides plan with specialty filter (query path):
  curl -s -X POST http://localhost:8000/rag/slides/plan -H "Content-Type: application/json" \
    -d '{"query":"...", "version":"v1-docai", "specialty":"implantologia"}' | jq

Operational notes
- Keep env vars explicit; no implicit defaults beyond existing behavior.
- No git push without explicit confirmation.
- Specialty filtering requires payload presence; re-index older docs after adding the payload field.

# UPDATE LOG — v0.14 (APPEND-ONLY)
Date: 2026-02-13 (Europe/Rome)

## GCS RAW ARCHITECTURE — Specialties folder convention (EN)
Decision
- Macro-specialty folder names will be ENGLISH to match: (a) majority of raw docs (80–90% EN) and (b) slide output language (EN).
- Specialty values are stable identifiers: lowercase + underscores only.

Bucket layout
- Raw bucket: gs://odonto-raw-docs-matcler/raw/<specialty>/<file>.pdf
- Parsed bucket remains separate: gs://odonto-parsed-matcler/<doc_id>/<version_id>/...

MVP specialties (initial folders)
- endodontics
- restorative
- implantology
- zygomatic_implants
- periodontology
- oral_surgery
- prosthodontics
- orthodontics
- pediatric_dentistry

Operational note
- Adding new specialties later is supported (no schema changes).
- Renaming an existing specialty requires re-indexing points in Qdrant for affected docs.

## NEXT IMPLEMENTATION STEPS (S1 execution checklist)
1) Update dev_register_doc_version.py
   - add --specialty
   - derive specialty from raw URI (raw/<specialty>/...) if flag missing
   - persist to documents.metadata_json["specialty"] without overwriting existing keys (e.g., "citation")

2) Update dev_index_items_qdrant.py
   - read documents.metadata_json["specialty"] once per doc_id
   - add payload["specialty"] to each point if present
   - re-index existing docs to backfill payload specialty field

3) Update API retrieval + slide plan propagation
   - add optional specialty to request models (/rag/query, /rag/answer, /rag/outline, /rag/slides/plan)
   - apply Qdrant filter key="specialty" when provided, combined with doc_id filter if present
   - ensure slides plan passes specialty through to retrieval/outline

4) Smoke tests
   - register + ingest + index 3 implantology docs
   - confirm: /rag/query with specialty="implantology" returns only implantology docs
   - confirm: /rag/query with specialty="endodontics" returns none (until docs added)
   - confirm: /rag/slides/plan specialty-scoped does not mix sources

# UPDATE LOG — v0.15 (APPEND-ONLY)
Date: 2026-02-16 (Europe/Rome)

## STEP S1 — Macro-specialty segmentation (SMOKE TEST END-TO-END)
Status: ✅ PASSED (implantology-only corpus)

Key runtime fixes (DocAI EU)
- Confirmed processor:
  - displayName: odonto-layout-parser
  - type: Layout Parser
  - region: eu
  - processor_id: cefc5e7bf97d1e2c
  - prediction endpoint: https://eu-documentai.googleapis.com/.../locations/eu/processors/<id>:process
- Root cause of prior failures:
  - Using global endpoint documentai.googleapis.com caused deployment mismatch and INVALID_ARGUMENT for EU processors.
- Fix implemented:
  - Use EU regional endpoint for both:
    (a) REST “discovery” checks and (b) gRPC client api_endpoint.
  - Verified via log:
    [docai] api_endpoint=eu-documentai.googleapis.com location=eu ...

Parsed bucket path convention (confirmed)
- Parsed layout outputs are stored under an explicit "parsed/" prefix:
  gs://odonto-parsed-matcler/parsed/<doc_id>/<version_id>/
  Files include: items.jsonl, assets.jsonl, plus assets/ and raw/ subfolders.

Ingest validation (EU)
- implant-macrodesign-osseo:
  - OK ingested via Document AI (gsutil)
  - items: 268, assets: 6
  - GCS: gs://odonto-parsed-matcler/parsed/implant-macrodesign-osseo/v1-docai/...
- implant-lovatto-2018:
  - OK ingested via Document AI (gsutil)
  - items: 249, assets: 4
  - GCS: gs://odonto-parsed-matcler/parsed/implant-lovatto-2018/v1-docai/...

Qdrant indexing validation
- Indexed and counted per doc_id matched expected items:
  - implant-macrodesign-10y: expected_n_items=231, qdrant_count=231
  - implant-macrodesign-osseo: expected_n_items=268, qdrant_count=268
  - implant-lovatto-2018: expected_n_items=249, qdrant_count=249

Retrieval filter smoke test
- POST /rag/query with specialty="implantology": ✅ non-empty results (top_k=5)
- POST /rag/query with specialty="endodontics": ✅ results=[] (expected because no endodontics docs ingested yet)
Interpretation:
- Specialty filter is functioning and prevents cross-specialty bleed.

Local API runtime note
- To complete smoke, FastAPI/Uvicorn were installed and API run locally.

Implementation note (index script)
- scripts/dev_index_items_qdrant.py was updated during smoke to:
  - ensure Qdrant point IDs are valid UUIDs
  - add sanity checks for doc_id/version_id

## NEXT STEPS (S1 hardening)
1) Commit-level review
- Capture the exact diffs for:
  - scripts/dev_docai_layout_ingest.py (EU endpoint selection + discovery)
  - scripts/dev_index_items_qdrant.py (UUID ids + sanity checks + specialty payload)
  - src/odonto_rag/api/rag_app.py (specialty propagation + filters)
  - requirements / dependency notes (fastapi/uvicorn if now required)

2) Add a minimal regression test checklist
- One doc in endodontics + rerun the same smoke to confirm:
  - specialty=endodontics returns only endodontics
  - specialty=implantology returns only implantology
  - specialty omitted searches across both

3) Slides plan scoped test
- Run /rag/slides/plan with specialty="implantology" and verify:
  - citations come only from implantology docs
  - no bleed once additional specialties are added

Workflow rules reminder
- No git push without explicit confirmation.
- Master remains append-only; updates delivered as downloadable .txt.

# UPDATE LOG — v0.16 (APPEND-ONLY)
Date: 2026-02-16 (Europe/Rome)

## STEP S1 — FINAL VALIDATION (PASSED)
Status: ✅ PASSED (multi-specialty scoped retrieval + slides)

What was validated
- Endodontics docs ingested and indexed:
  - endo-clinical-radiographic-failure
  - endo-fransson-2022
- Scoped retrieval (`/rag/query`) returned only docs of the requested specialty.
- Scoped slides plan (`/rag/slides/plan`) returned coherent doc_ids per specialty:
  - implantology plan doc_ids:
    - implant-macrodesign-10y
    - implant-macrodesign-osseo
  - endodontics plan doc_ids:
    - endo-clinical-radiographic-failure
    - endo-fransson-2022
- PPTX generation completed for both specialties without cross-specialty contamination.

Hardening completed in this step
- Citation metadata now treated as mandatory for slide-quality output:
  - existing docs backfilled with `metadata_json.citation`
  - audit script added: `scripts/catalog_audit_citation.py`
  - current audit status: `missing_citation=0`
- Registration hardening:
  - `scripts/dev_register_doc_version.py` now supports `--citation`
  - new docs require citation at registration time
  - existing docs without citation now fail registration update
- Runtime hardening:
  - `/rag/slides/plan` now fails fast if any source lacks `doc_citation`
    (explicit error listing offending `doc_id`s)
- Deck layout hardening:
  - `src/odonto_rag/deck/pptx_builder.py` enforces 16:9 output (13.333 x 7.5)
  - uniform fonts and bounded text to reduce overflow:
    - title/body/footer standardized
    - bullet/title truncation + bullet count cap

## NEXT STEP — S2 Citation-grounded slide claims
Goal
- Ensure each slide claim is explicitly grounded in retrieved evidence.

Scope
1) Enforce per-bullet grounding
- Every bullet must carry explicit sources as item references:
  - `sources: [item_id, ...]` (or equivalent structured mapping).

2) Enforce specialty-consistent sources
- Validate that all source item_ids map to docs with the same requested specialty.
- Reject or repair any mixed-specialty source assignment.

3) Fallback policy for weak evidence
- If support is insufficient for strong claims:
  - reduce ambition (fewer claims, more definitions/overview content)
  - avoid unsupported specific statements.

4) Automated smoke gate
- Add an automatic check asserting:
  - every generated slide has coherent sources
  - every bullet has at least one supporting source item
  - all source docs match requested specialty.

Acceptance criteria
- S2 is PASS only when the automatic check succeeds for both:
  - specialty=implantology
  - specialty=endodontics

# UPDATE LOG — v0.17 (APPEND-ONLY)
Date: 2026-02-16 (Europe/Rome)

## STEP S2 — Citation-grounded slide claims (MVP IMPLEMENTED)
Status: ✅ PASSED (automatic smoke on implantology + endodontics)

Implemented
- API schema update:
  - `RagSlidePlanItem` now includes:
    - `bullet_source_item_ids: List[List[str]]`
- Grounding validation in `/rag/slides/plan`:
  - every slide must include `sources`
  - every bullet must include at least one supporting `item_id` in `bullet_source_item_ids`
  - bullet/source cardinality is validated
- Specialty consistency validation:
  - source docs are checked against requested `specialty`
  - mismatch returns `HTTP 400` with explicit details
- Fallback behavior (low evidence):
  - if a generated slide has no mappable source from citations, generation falls back to lower-ambition content
  - slide is grounded to available evidence (no unsupported claim-only output)

Files added/updated
- Updated: `src/odonto_rag/api/rag_app.py`
- Added: `scripts/check_slides_plan_grounding.py`

Automatic smoke gate
1) Generate scoped plans:
- implantology -> `/tmp/plan_implantology_s2.json`
- endodontics -> `/tmp/plan_endodontics_s2.json`

2) Validate grounding + specialty coherence:
- `python3 scripts/check_slides_plan_grounding.py --plan-json /tmp/plan_implantology_s2.json --specialty implantology --db catalog.sqlite3`
- `python3 scripts/check_slides_plan_grounding.py --plan-json /tmp/plan_endodontics_s2.json --specialty endodontics --db catalog.sqlite3`

Observed result
- Implantology check: `PASS` (`doc_ids=implant-macrodesign-10y,implant-macrodesign-osseo`)
- Endodontics check: `PASS` (`doc_ids=endo-clinical-radiographic-failure,endo-fransson-2022`)

Operational note
- Citation metadata remains mandatory; `/rag/slides/plan` already fails if `doc_citation` is missing for used sources.

# UPDATE LOG — v0.18 (APPEND-ONLY)
Date: 2026-02-16 (Europe/Rome)

## STEP S3 — Visual assets in slides (tables-first) status
Status: ✅ IMPLEMENTED (MVP)

What is now working
- `slides/plan` can include `visuals` selected from deterministic asset candidates (specialty-safe).
- `slides/pptx` renders visuals with fixed deterministic placement (1-up / 2-up policy).
- Table visuals are now rendered as REAL PPTX tables (not snapshots) when `table_uri -> rows` is available.
- Fallback policy remains deterministic (`MISSING_ASSET` notes / controlled degrade path).

Validation snapshot
- Implantology generated deck with real tables:
  - `TABLE_SHAPES=10`, `PICTURE_SHAPES=0`
- Endodontics generated deck with real tables:
  - `TABLE_SHAPES=6`, `PICTURE_SHAPES=0`

## NEXT STEP (planned) — Table hardening for coherence
Primary objective
- Improve consistency, readability, and semantic fidelity of rendered tables across decks.

Hardening scope
1) Table normalization
- Stable row/column limits by layout profile.
- Better header detection and header styling consistency.
- Predictable truncation/wrapping rules per cell.

2) Semantic coherence
- Preserve meaningful column labels and units when available.
- Reduce noisy rows and boilerplate artifacts from extraction.
- Add deterministic row-priority rules (e.g., keep top informative rows).

3) Visual coherence
- Unified typography, padding, border thickness, and alignment.
- Deterministic width allocation by content class (text vs numeric).
- Safer behavior for oversized tables (split/compact policy).

4) Auditability
- Per-table render metadata in notes/logs (asset_id, doc_id, page, applied normalization policy).
- Deterministic fallback reason codes when degradation occurs.

# UPDATE LOG — v0.19 (APPEND-ONLY)
Date: 2026-02-17 (Europe/Rome)

## STEP S3.1 — Table hardening + planner fallback (IMPLEMENTED)
Status: ✅ IMPLEMENTED (with regression fix)

What was implemented
- PPTX table rendering hardening in `src/odonto_rag/deck/pptx_builder.py`:
  - Replaced placeholder captions (`Table from source`) with deterministic caption+locator format.
  - Added per-table audit notes in speaker notes:
    - `TABLE_1 asset_id=... doc_id=... locator=...`
    - `TABLE_2 ...`
  - Added deterministic table normalization:
    - drop fully empty rows/columns
    - header hole fill (`prev header` or `—`)
    - optional spans-to-merge translation when span metadata exists
  - Added consistent table style (header shading/bold, first-column emphasis, padding).

- Slides planner deterministic fallback in `src/odonto_rag/api/rag_app.py`:
  - If a slide has quantitative/table intent and no visual selected by the LLM,
    auto-attach one top-ranked table candidate (deterministic ranking).
  - Intent gate includes table/data tokens + numeric/percent patterns.
  - Preference order for fallback table:
    1) same source doc(s) as slide citations
    2) semantic score to slide text
    3) stable doc_id/asset_id tie-break.

Regression discovered and fixed
- During hardening, overflow policy in PPTX builder became too aggressive and forced table-as-image fallbacks.
- Fix applied:
  - keep table-as-shape as default,
  - use compact mode (smaller font) instead of immediate image fallback,
  - raise overflow threshold to reduce false-positive degrade.
- Post-fix validation confirms real table shapes restored.

Validation snapshot (implantology)
- Plan v2 visual assignment: 7/10 slides with table visuals.
- Final deck (post-fix):
  - `out/decks/implantology_10slides_hardening_v3_true_tables.pptx`
  - `SLIDES=10`
  - `TABLE_SHAPES=7`
  - `PICTURE_SHAPES=0`
  - no `TABLE_RENDER_FALLBACK` notes.

Files updated in this step
- Updated: `src/odonto_rag/deck/pptx_builder.py`
- Updated: `src/odonto_rag/api/rag_app.py`

Operational note
- The latest implantology deck now includes deterministic table grounding and keeps tables as native PPTX tables (not snapshots), except where explicit missing-asset conditions would require fallback.

# UPDATE LOG — v0.20 (APPEND-ONLY)
Date: 2026-02-17 (Europe/Rome)

## STEP S3.2 — Keynote compatibility + output hygiene
Status: ✅ IMPLEMENTED

What was implemented
- Added Keynote compatibility mode in PPTX builder:
  - Env flag: `PPTX_KEYNOTE_SAFE=1`
  - Compatibility behavior:
    - conservative font/truncation handling,
    - avoids note slides output,
    - avoids fragile table merge behavior.

- Added hybrid mode for Keynote + native tables:
  - Env flag combination:
    - `PPTX_KEYNOTE_SAFE=1`
    - `PPTX_KEYNOTE_KEEP_TABLES=1`
  - Result:
    - Keynote-friendly package profile,
    - real PPTX tables preserved (no snapshot fallback by policy).

Validation artifacts
- Keynote-safe (max compatibility):
  - `out/decks/implantology_10slides_keynote_safe_v2.pptx`
- Keynote-hybrid with native tables:
  - `out/decks/implantology_10slides_keynote_hybrid_true_tables.pptx`
  - XML/package check confirms 10 slides and 7 table markers, no note slides.

Output folder cleanup
- Per user request, `out/` was cleaned from obsolete test/temp artifacts:
  - removed `test_*`, `tmp_*`, legacy generic outputs (`RAG_Slides_*`, `engine_test.pptx`, `final.pptx`, etc.), and `.DS_Store`.
- Preserved:
  - `out/assets/` (source visual cache)
  - `out/decks/` (current implantology deliverables)

Files updated in this step
- Updated: `src/odonto_rag/deck/pptx_builder.py`
- Updated: `src/odonto_rag/api/rag_app.py` (table fallback planner from previous sub-step)
- Updated: `docs/master.md`

# UPDATE LOG — v0.21 (APPEND-ONLY)
Date: 2026-02-17 (Europe/Rome)

## STEP S4 — Figure assets end-to-end (S4.1→S4.4)
Status: ✅ IMPLEMENTED

What was implemented
- S4.1:
  - Extended DocAI ingest asset extraction to include figure/image-like visual elements with `bbox + page` in `assets.jsonl`.
  - Kept extraction-only behavior (no render in ingest).
- S4.2:
  - Hardened deterministic PDF→PNG crop renderer with cache-first behavior under `out/assets/<doc_id>/<version>/<asset_id>.png`.
  - Added `--force` to bypass cache when needed.
- S4.3:
  - Added deterministic visual-candidate ranking that guarantees figure/image/chart candidates are available to planner.
  - Added figure fallback intent path in planner.
  - Strengthened slide grounding validation for visuals:
    - render path must exist
    - visual specialty must match requested specialty (when present).
- S4.4:
  - Kept deterministic 1-up/2-up visual rendering for figures in PPTX builder.
  - Standardized figure notes to `FIGURE_n asset_id=... doc_id=... locator=...`.
  - Preserved Keynote-safe behavior.

Files updated in this step
- Updated: `scripts/dev_docai_layout_ingest.py`
- Updated: `scripts/dev_render_layout_assets.py`
- Updated: `src/odonto_rag/api/rag_app.py`
- Updated: `src/odonto_rag/deck/pptx_builder.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.22 (APPEND-ONLY)
Date: 2026-02-17 (Europe/Rome)

## STEP S4.5 — Figure fallback + anti-stretch render (IMPLEMENTED)
Status: ✅ IMPLEMENTED

What was implemented
- Figure extraction hardening in ingest:
  - `scripts/dev_docai_layout_ingest.py` now includes a deterministic PDF fallback detector
    when DocAI layout does not emit figure/image blocks.
  - Fallback writes figure assets with `asset_type=figure`, `bbox + page + locator`,
    source tag: `pdf_fallback_connected_components`.
  - Added opt-out flag: `--disable-figure-fallback`.

- PPTX figure rendering quality fix:
  - `src/odonto_rag/deck/pptx_builder.py` now renders pictures with preserved aspect ratio
    (contain + centered in slot), avoiding stretched/distorted visuals.

Validation run (implantology, v1-docai)
- Re-ingested docs:
  - `implant-lovatto-2018`
  - `implant-macrodesign-10y`
  - `implant-macrodesign-osseo`
- Rendered/enriched assets regenerated in `out/assets/.../assets.enriched.jsonl`.
- Observed figure assets:
  - `implant-macrodesign-10y`: 5 figure assets detected
  - other two docs remained table-only with current source structure.

Deck generation + cleanup
- Generated:
  - `out/decks/implantology_main_10slides_fixed_aspect.pptx`
- Cleaned output folder as requested:
  - `out/decks/` now retains only the final deliverable above.

Operational note
- Planner behavior remains non-forced:
  - visuals are inserted only when semantically relevant;
  - no mandatory figure injection policy is applied.

Files updated in this step
- Updated: `scripts/dev_docai_layout_ingest.py`
- Updated: `src/odonto_rag/deck/pptx_builder.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.23 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S5 — Visual Claim Grounding (bullet → visual linkage) + audit gate
Status: ✅ IMPLEMENTED

To be implemented
- [x] Schema (slide plan):
  - add `bullet_visual_asset_ids: List[List[str]]` with bullet-level cardinality.
  - add slide-level `visual_role: evidence | illustrative`.
- [x] Planner deterministic assignment:
  - bullet→visual links only from already selected visual candidates.
  - numeric/quant bullets prefer table linking.
  - lexical match on bullet text vs visual caption/locator/table headers.
  - stable tie-break via `asset_id` ordering.
- [x] Rendering audit trace:
  - speaker notes include `BULLET_n evidence_items=[...] visuals=[...]`.
- [x] Smoke gate S5:
  - validate bullet/visual cardinality.
  - validate linked visual ids exist on slide.
  - validate specialty-safe linked visuals.
  - if slide has visuals: require at least one linked bullet unless `visual_role=illustrative`.

Implemented
- [x] Added new slide-plan fields in API model:
  - `bullet_visual_asset_ids` (default deterministic list per bullet)
  - `visual_role` (defaulted by deterministic inference).
- [x] Implemented deterministic visual-claim matcher in planner:
  - no creative linking outside chosen slide visuals.
  - numeric claim heuristic links to tables first.
  - token overlap matcher uses captions + locator hints + table headers.
  - deterministic ordering by `asset_id`.
- [x] Implemented `visual_role` deterministic policy:
  - any table on slide ⇒ `evidence`.
  - figure-only slides default `illustrative`, upgraded to `evidence` with result/outcome caption cues.
- [x] Extended slide grounding validation:
  - bullet/source and bullet/visual cardinality checks.
  - linked visual existence + specialty checks.
  - visual-role gate (`evidence` requires at least one bullet→visual link).
  - optional strict gate via env flag `SLIDES_ENFORCE_NUMERIC_VISUAL_LINK=1`.
- [x] Added speaker-notes audit lines in PPTX builder:
  - `Bullet grounding:`
  - `BULLET_n evidence_items=[...] visuals=[...]`.

Files updated in this step
- Updated: `src/odonto_rag/api/rag_app.py`
- Updated: `src/odonto_rag/deck/pptx_builder.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.24 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## S5 Operational Decision — Keynote profile
Status: ✅ IMPLEMENTED

Decision
- Default delivery profile for S5 is now **Keynote hybrid**:
  - `PPTX_KEYNOTE_SAFE=1`
  - `PPTX_KEYNOTE_KEEP_TABLES=1`

Why this profile
- Keeps Keynote-safe package behavior (no notes slides emitted).
- Preserves native PPTX tables for better editability/review workflow.
- Avoids visual-evidence degradation from forced picture-only table fallback.

Validated outputs
- Preferred final deck:
  - `out/decks/implantology_main_10slides_s5_keynote_hybrid_regen.pptx`
- Alternative (safe non-hybrid, picture-based visuals):
  - `out/decks/implantology_main_10slides_s5_keynote_safe_regen.pptx`

Files updated in this step
- Updated: `docs/master.md`

# UPDATE LOG — v0.25 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S6 — Audit Pack & Deterministic Evidence Export (deck → JSON → assets)
Status: ✅ IMPLEMENTED

What was implemented
- Deterministic audit manifest export on PPTX build:
  - Endpoint `POST /rag/slides/pptx` now writes:
    - `out/decks/<deck_id>.audit.json`
  - Manifest includes:
    - request metadata (`mode`, `query/outline`, `version`, `specialty`, timestamp, env profile),
    - per-slide bullet evidence mapping (`item_id + doc_id + locator + score`),
    - per-slide visuals (`asset_id`, `type`, `doc_id`, `locator`, `render_path`, `exists`, `visual_role`),
    - summary (`unique_docs/items/assets`, `missing_asset_count`, applied S2/S5 gates).

- S2.1 closure (structured sources embedded per bullet):
  - Added `bullet_sources_structured` to slide plan schema.
  - Planner now expands every bullet evidence item directly with `doc_id + locator (+score)`.
  - Grounding validator now enforces cardinality and presence of structured evidence fields.

- Optional evidence bundle export:
  - `POST /rag/slides/pptx` accepts `evidence_bundle: true`
    (or env `PPTX_EXPORT_EVIDENCE_BUNDLE=1`).
  - Writes deterministic zip:
    - `out/decks/<deck_id>.evidence.zip`
  - Bundle content:
    - `audit.json`
    - `assets/<asset_id>.<ext>` for used visuals with local render path
    - `tables/<asset_id>.json` when table payload is available.

- S6 smoke gate script:
  - New script: `scripts/check_slides_audit_pack.py`
  - Validates:
    - audit file exists and is valid JSON,
    - every bullet has at least one evidence item with `item_id/doc_id/locator`,
    - linked visuals have valid object structure and specialty coherence (when provided),
    - `summary.missing_asset_count` consistency and policy gate (`--allow-missing-assets`).

Files updated in this step
- Updated: `src/odonto_rag/api/rag_app.py`
- Added: `scripts/check_slides_audit_pack.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.26 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S6.1 — One-shot smoke runner (plan → pptx → audit)
Status: ✅ IMPLEMENTED

What was implemented
- Added a single local smoke runner script to execute the full verification chain:
  1) optional S5 plan grounding check
  2) PPTX generation via `POST /rag/slides/pptx`
  3) S6 audit-pack validation on generated `*.audit.json`
- Script auto-infers specialty from `plan.request.specialty` when available.
- Supports evidence bundle requests and missing-asset policy override.

New script
- Added: `scripts/run_slides_audit_smoke.py`

Usage example
- `python3 scripts/run_slides_audit_smoke.py --plan-json /tmp/plan.json --api-base http://127.0.0.1:8000`
- Optional:
  - `--evidence-bundle`
  - `--allow-missing-assets`
  - `--specialty implantology`
  - `--skip-plan-check`

Files updated in this step
- Added: `scripts/run_slides_audit_smoke.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.27 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S7 — Deterministic Regression Suite (golden plans + golden audit diffs)
Status: ✅ IMPLEMENTED

To be implemented
- [x] S7.1 Golden fixtures (2 specialties):
  - create `tests/fixtures/slides/implantology.plan.json`
  - create `tests/fixtures/slides/endodontics.plan.json`
  - fixture plans must be stable and used as renderer input (no Qdrant/LLM in tests)
- [x] S7.2 Offline smoke runner mode (plan → pptx → audit):
  - extend `scripts/run_slides_audit_smoke.py` with `--offline`
  - in offline mode call local builder directly from plan JSON
  - deterministic outputs:
    - `out/tests/<name>/deck.pptx`
    - `out/tests/<name>/deck.audit.json`
- [x] S7.3 Golden audit diff (canonical + semantic):
  - add `scripts/diff_audit_json.py old.audit.json new.audit.json`
  - normalization rules:
    - ignore volatile fields (`timestamp`, `run_id`)
    - sort lists deterministically (slides, bullets, evidence items, visual assets)
  - semantic diff signal:
    - `bullet text changed`
    - `evidence items changed`
    - `visual links changed`
    - `missing_asset_count changed`
- [x] S7.4 Determinism gate (hash):
  - normalized `SHA256` per fixture audit JSON
  - fail when hash/diff changes without explicit golden update

Implemented
- [x] Added deterministic fixture plans:
  - `tests/fixtures/slides/implantology.plan.json`
  - `tests/fixtures/slides/endodontics.plan.json`
- [x] Added offline build mode to smoke runner:
  - updated `scripts/run_slides_audit_smoke.py`
  - new args: `--offline`, `--out-root`, `--test-name`
  - offline flow reuses local PPTX builder and local audit manifest generation.
- [x] Added canonical/semantic audit diff tool:
  - new `scripts/diff_audit_json.py`
  - supports:
    - semantic diff between old/new audits
    - normalized SHA256 generation (`--sha256`)
    - CI-friendly fail gate (`--fail-on-diff`)
- [x] Added deterministic suite orchestrator:
  - new `scripts/run_slides_regression_suite.py`
  - runs fixtures offline, validates against golden audits, enforces hash gate
  - supports `--update-golden` for intentional baseline refresh.
- [x] Committed golden baselines:
  - `tests/fixtures/slides/implantology.golden.audit.json`
  - `tests/fixtures/slides/endodontics.golden.audit.json`
  - `tests/fixtures/slides/golden_hashes.json`

Validation snapshot
- `python3 scripts/run_slides_regression_suite.py --update-golden` ✅
- `python3 scripts/run_slides_regression_suite.py` ✅
- `python3 scripts/diff_audit_json.py tests/fixtures/slides/implantology.golden.audit.json out/tests/implantology/deck.audit.json --fail-on-diff` ✅

Files updated in this step
- Added: `tests/fixtures/slides/implantology.plan.json`
- Added: `tests/fixtures/slides/endodontics.plan.json`
- Added: `tests/fixtures/slides/implantology.golden.audit.json`
- Added: `tests/fixtures/slides/endodontics.golden.audit.json`
- Added: `tests/fixtures/slides/golden_hashes.json`
- Updated: `scripts/run_slides_audit_smoke.py`
- Added: `scripts/diff_audit_json.py`
- Added: `scripts/run_slides_regression_suite.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.28 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S8 — Clinical Quality Eval Suite (rubric + metrics su audit.json)
Status: ✅ IMPLEMENTED

Implemented
- [x] Added deterministic quality evaluator:
  - new `scripts/eval_audit_quality.py`
  - input: `*.audit.json`
  - outputs:
    - `out/tests/<name>/quality.json`
    - `out/tests/<name>/quality.txt`
- [x] Implemented deterministic metric groups:
  - evidence density (`mean/min/max`, `% bullets with single evidence`)
  - source diversity (`unique_docs` deck + per-slide, slide diversity warnings)
  - locator quality (`complete locator %`, `page-only %`)
  - visual coherence (`visual_role=evidence` link consistency, numeric bullet→table link rate)
  - readability (bullets/slide distribution, long bullet risk, table density when rows/cols available)
  - fallback rate (`pdf_fallback_connected_components` detection if present in visual payload/meta + missing assets)
  - coverage (`query/outline token coverage`, low-overlap slide warnings)
- [x] Added optional quality budget gates (CLI-tunable):
  - hard:
    - `--hard-min-evidence-per-bullet` (default `1`)
    - missing assets hard gate via `--strict-missing-assets` or `--hard-max-missing-assets`
  - warn:
    - `--warn-max-single-evidence-ratio` (default `0.30`)
    - `--warn-max-fallback-asset-ratio` (default `0.20`)
    - readability/overload thresholds (`bullets cap`, `long bullets`, etc.)
- [x] Added deterministic quality diff/hash utility:
  - new `scripts/diff_quality_json.py`
  - supports:
    - semantic diff `old/new`
    - normalized SHA256 (`--sha256`)
    - CI fail mode (`--fail-on-diff`)
- [x] Integrated quality eval in regression suite:
  - updated `scripts/run_slides_regression_suite.py`
  - now generates `quality.json` per fixture
  - optional golden check via `--quality-golden`:
    - compares against `tests/fixtures/slides/<name>.golden.quality.json`
    - enforces `tests/fixtures/slides/golden_quality_hashes.json`
  - `--update-golden --quality-golden` refreshes both quality golden JSON and hash baselines.
- [x] Added quality golden baselines:
  - `tests/fixtures/slides/implantology.golden.quality.json`
  - `tests/fixtures/slides/endodontics.golden.quality.json`
  - `tests/fixtures/slides/golden_quality_hashes.json`

Validation snapshot
- `python3 -m py_compile scripts/eval_audit_quality.py scripts/diff_quality_json.py scripts/run_slides_regression_suite.py` ✅
- `python3 scripts/run_slides_regression_suite.py --update-golden --quality-golden` ✅
- `python3 scripts/run_slides_regression_suite.py --quality-golden` ✅

Files updated in this step
- Added: `scripts/eval_audit_quality.py`
- Added: `scripts/diff_quality_json.py`
- Updated: `scripts/run_slides_regression_suite.py`
- Added: `tests/fixtures/slides/implantology.golden.quality.json`
- Added: `tests/fixtures/slides/endodontics.golden.quality.json`
- Added: `tests/fixtures/slides/golden_quality_hashes.json`
- Updated: `tests/fixtures/slides/implantology.golden.audit.json`
- Updated: `tests/fixtures/slides/endodontics.golden.audit.json`
- Updated: `tests/fixtures/slides/golden_hashes.json`
- Updated: `docs/master.md`

# UPDATE LOG — v0.29 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S8 follow-up — CI profile strict defaults (dev vs ci)
Status: ✅ IMPLEMENTED

Implemented
- [x] Added execution profile in regression suite:
  - updated `scripts/run_slides_regression_suite.py`
  - new flag: `--profile {dev,ci}` (default: `dev`)
- [x] `--profile ci` now auto-enables strict quality gate behavior:
  - `quality_golden=True`
  - `quality_fail_on_hard=True`
  - `quality_strict_missing_assets=True`
- [x] Kept `dev` behavior permissive by default (current local workflow unchanged unless flags are passed).
- [x] Hardened quality golden diff/hash across profiles:
  - updated `scripts/diff_quality_json.py`
  - `thresholds` treated as non-semantic for normalization/hash
  - rationale: CI/dev may use different gate thresholds but should diff on outcomes/metrics, not config knobs.
- [x] Refreshed quality hash baselines after normalization change:
  - updated `tests/fixtures/slides/golden_quality_hashes.json`

Validation snapshot
- `python3 -m py_compile scripts/diff_quality_json.py scripts/run_slides_regression_suite.py` ✅
- `python3 scripts/run_slides_regression_suite.py --profile dev --quality-golden --update-golden` ✅
- `python3 scripts/run_slides_regression_suite.py --profile dev --quality-golden` ✅
- `python3 scripts/run_slides_regression_suite.py --profile ci` ✅

Files updated in this step
- Updated: `scripts/run_slides_regression_suite.py`
- Updated: `scripts/diff_quality_json.py`
- Updated: `tests/fixtures/slides/golden_quality_hashes.json`
- Updated: `docs/master.md`

# UPDATE LOG — v0.30 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S9 — Deterministic Auto-Repair Loop (plan repair) + CI-friendly deltas
Status: ✅ IMPLEMENTED

Implemented
- [x] S9.1 Deterministic plan normalizer/repair:
  - added `scripts/repair_slide_plan.py`
  - input: `plan.json`
  - outputs:
    - `plan.repaired.json`
    - `repairs.applied.json` (machine-readable list)
  - deterministic rules implemented:
    - readability:
      - cap bullets per slide (`max_bullets_per_slide`, default `6`)
      - split/truncate long bullets with deterministic policy + ellipsis
    - visual coherence:
      - for `visual_role=evidence` + no linked bullets:
        - deterministic relinking (`numeric -> first table`, else keyword overlap, stable tie-break)
        - optional downgrade to `illustrative` when no deterministic link exists
    - hard-mode additions:
      - evidence overload trim to top-N per score (stable tie-break)
      - table density compact policy (`compact_mode`) when rows/cols/cells exceed thresholds
- [x] S9.2 Hooked deterministic auto-repair in regression runner:
  - updated `scripts/run_slides_regression_suite.py`
  - new flag: `--auto-repair {off,soft,hard}` (default: `off`)
  - flow with auto-repair enabled:
    - run `before`: build + audit + quality eval
    - if hard-fail or warning_count above threshold:
      - repair plan deterministically
      - run `after`: rebuild + re-eval
      - persist artifacts in:
        - `out/tests/<name>/before/`
        - `out/tests/<name>/after/`
      - produce CI-friendly deltas:
        - `out/tests/<name>/after/audit.diff.txt`
        - `out/tests/<name>/after/quality.diff.txt`
        - `out/tests/<name>/after/repairs.applied.json`
      - write consolidated report:
        - `out/tests/<name>/auto_repair_report.json` (before/after gates + diff + repairs)
- [x] S9.3 Explicit policy controls via env/flags:
  - env-supported in `repair_slide_plan.py`:
    - `SLIDES_REPAIR_MAX_BULLETS_PER_SLIDE`
    - `SLIDES_REPAIR_MAX_EVIDENCE_PER_BULLET`
    - `SLIDES_REPAIR_LONG_BULLET_CHARS`
    - `SLIDES_REPAIR_DOWNGRADE_VISUAL_ROLE_ON_NO_LINK`
  - CLI overrides:
    - in `repair_slide_plan.py`:
      - `--max-bullets-per-slide`
      - `--max-evidence-per-bullet`
      - `--long-bullet-chars`
      - `--downgrade-visual-role-on-no-link`
    - pass-through in regression suite:
      - `--repair-max-bullets-per-slide`
      - `--repair-max-evidence-per-bullet`
      - `--repair-long-bullet-chars`
      - `--repair-downgrade-visual-role-on-no-link`

Validation snapshot
- `python3 -m py_compile scripts/repair_slide_plan.py scripts/run_slides_regression_suite.py` ✅
- `python3 scripts/run_slides_regression_suite.py --fixtures implantology` ✅
- `python3 scripts/run_slides_regression_suite.py --fixtures implantology --auto-repair soft --auto-repair-warning-threshold 0 --quality-warn-long-bullet-chars 20 --quality-warn-max-long-bullets 0 --repair-long-bullet-chars 20` ✅ (auto-repair branch executed; expected strict-threshold failure due residual non-readability warnings, with full before/after deltas generated)

Files updated in this step
- Added: `scripts/repair_slide_plan.py`
- Updated: `scripts/run_slides_regression_suite.py`
- Updated: `docs/master.md`

# UPDATE LOG — v0.31 (APPEND-ONLY)
Date: 2026-02-18 (Europe/Rome)

## STEP S10 — Audit Review UI (local, static) + click-to-evidence
Status: ✅ IMPLEMENTED

Implemented
- [x] S10.1 Static HTML viewer (zero backend):
  - Added `scripts/build_audit_viewer.py` to generate:
    - `out/decks/<deck_id>.audit.html` (or custom output path)
  - Viewer is built from normalized/stable audit ordering:
    - slides sorted deterministically,
    - evidence items sorted deterministically,
    - visuals sorted deterministically.
  - Viewer UI includes:
    - slide list with warning badges,
    - per-slide bullets + evidence list (`doc_id`, `locator`, `score`),
    - visuals with thumbnail when local/bundled render is available,
    - table JSON link (`tables/<asset_id>.json`) when evidence bundle is provided/extracted.

- [x] S10.2 Evidence preview minimale:
  - For evidence items with page locator, viewer exposes `Preview page` links.
  - `build_audit_viewer.py` pre-generates/caches deterministic page PNGs under:
    - `out/assets/pages/<doc_id>/<version>/pXXXX.png`
  - Cache-first policy:
    - reuse existing preview if present,
    - otherwise render once from source PDF (via catalog `documents.gcs_raw_path`) and reuse.
  - Visual crops are shown directly from `render_path` when available, with bundle fallback from extracted `assets/`.

- [x] S10.3 Review outcomes (human-in-the-loop):
  - Viewer supports per-bullet review fields:
    - `status: ok | needs_edit | reject`
    - `note`
    - optional `suggested_rewrite`
  - Added save/load flow in static UI:
    - download JSON,
    - file picker save (when browser supports File System Access API),
    - load existing review JSON.
  - `build_audit_viewer.py` also emits default review template:
    - `out/decks/<deck_id>.review.json` (or custom path)
    - with deterministic `slide_id` / `bullet_id`.

- [x] S10.4 Optional gate “no unresolved review”:
  - Added `scripts/check_review_complete.py`.
  - Fails when:
    - any review row has `status=reject`,
    - high-risk bullets (numeric-like text + only 1 evidence item) have no review status.

Usage
- Build viewer:
  - `python3 scripts/build_audit_viewer.py --audit-json out/decks/<deck_id>.audit.json --evidence-zip out/decks/<deck_id>.evidence.zip`
- Run review-completion gate:
  - `python3 scripts/check_review_complete.py --audit-json out/decks/<deck_id>.audit.json --review-json out/decks/<deck_id>.review.json`

Validation snapshot
- `python3 -m py_compile scripts/build_audit_viewer.py scripts/check_review_complete.py` ✅
- Viewer generation on fixture audit ✅
- Review gate failure path validated (missing high-risk review) ✅
- Review gate pass path validated after status assignment ✅

Files updated in this step
- Added: `scripts/build_audit_viewer.py`
- Added: `scripts/check_review_complete.py`
- Updated: `docs/master.md`
