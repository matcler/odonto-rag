"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.rag.outline
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine/clinical_outline.py is deprecated; import from odonto_rag.rag.outline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.rag.outline import *  # noqa: F401,F403
