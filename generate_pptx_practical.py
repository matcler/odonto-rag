"""
DEPRECATED: compatibility forwarder.

Use instead:
  python scripts/dev_generate_pptx_practical.py
"""
from __future__ import annotations

import warnings
warnings.warn(
    "generate_pptx_practical.py is deprecated; use scripts/dev_generate_pptx_practical.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

from scripts.dev_generate_pptx_practical import main

if __name__ == "__main__":
    main()
