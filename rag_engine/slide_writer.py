"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.deck.slide_writer
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine/slide_writer.py is deprecated; import from odonto_rag.deck.slide_writer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.deck.slide_writer import *  # noqa: F401,F403
