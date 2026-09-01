# -*- coding: utf-8 -*-
"""Enable ``python -m jykj_ocr ...``.

Delegates to :func:`jykj_ocr.cli.main` so that ``python -m jykj_ocr`` mirrors
the ``jykj-ocr`` console-script entry point declared in ``pyproject.toml``.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
