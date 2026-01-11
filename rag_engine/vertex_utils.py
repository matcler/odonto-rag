"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.rag.providers.vertex
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine/vertex_utils.py is deprecated; import from odonto_rag.rag.providers.vertex instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.rag.providers.vertex import *  # noqa: F401,F403
