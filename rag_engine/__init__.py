"""
DEPRECATED package.

Use the new package under `odonto_rag` instead.
This module will be removed once all imports are migrated.
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine is deprecated; import from odonto_rag.* instead.",
    DeprecationWarning,
    stacklevel=2,
)
