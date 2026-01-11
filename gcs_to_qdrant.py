"""
DEPRECATED: compatibility forwarder.

Use instead:
  python scripts/dev_ingest_gcs_to_qdrant.py
"""
from __future__ import annotations

import warnings
warnings.warn(
    "gcs_to_qdrant.py is deprecated; use scripts/dev_ingest_gcs_to_qdrant.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

from scripts.dev_ingest_gcs_to_qdrant import main

if __name__ == "__main__":
    main()
