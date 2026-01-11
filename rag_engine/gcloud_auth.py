"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.ingest.gcloud_auth
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine/gcloud_auth.py is deprecated; import from odonto_rag.ingest.gcloud_auth instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.ingest.gcloud_auth import *  # noqa: F401,F403
