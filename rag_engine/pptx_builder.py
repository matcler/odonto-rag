"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.deck.builder
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine/pptx_builder.py is deprecated; import from odonto_rag.deck.builder instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.deck.builder import *  # noqa: F401,F403
