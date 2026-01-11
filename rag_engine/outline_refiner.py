"""
DEPRECATED: compatibility shim.

This module will be removed soon.
Update imports to: odonto_rag.rag.outline_refiner
"""
from __future__ import annotations

import warnings
warnings.warn(
    "rag_engine/outline_refiner.py is deprecated; import from odonto_rag.rag.outline_refiner instead.",
    DeprecationWarning,
    stacklevel=2,
)

from odonto_rag.rag.outline_refiner import *  # noqa: F401,F403
