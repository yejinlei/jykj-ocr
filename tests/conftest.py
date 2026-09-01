# -*- coding: utf-8 -*-
"""Shared pytest fixtures for jykj_ocr tests.

The package is not installed, so ``src/`` is put on ``sys.path`` here.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture
def sample_image():
    """A small PIL image with text drawn on it."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 44), "HELLO OCR 2026", fill="black")
    return image
