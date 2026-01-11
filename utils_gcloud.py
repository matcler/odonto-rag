"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.ingest.utils_gcloud
"""
from __future__ import annotations

import warnings
warnings.warn(
    "utils_gcloud.py is deprecated; import from odonto_rag.ingest.utils_gcloud instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.ingest.utils_gcloud import *  # noqa: F401,F403
