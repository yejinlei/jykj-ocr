# -*- coding: utf-8 -*-
"""Lazy engine registration and public engine API.

Importing this package never imports ``rapidocr``, ``openai`` or ``PIL`` —
those are pulled in only when a caller requests an engine that needs them.
"""

from __future__ import annotations

from typing import Callable

from .base import (
    BaseEngine,
    EngineError,
    EngineNotAvailable,
    InputError,
    PageImage,
    available_engines,
    create_engine,
    describe_engines,
    load_image,
    register,
    register_lazy,
)


def _import_rapidocr() -> None:
    from ..engines import rapidocr_engine  # noqa: F401


def _import_multimodal() -> None:
    from ..engines import multimodal_engine  # noqa: F401
    from ..engines import siliconflow_engine  # noqa: F401


# Deferred so that importing jykj_ocr never imports rapidocr/openai.
register_lazy("rapidocr", _import_rapidocr)
register_lazy("multimodal", _import_multimodal)
register_lazy("siliconflow", _import_multimodal)

__all__ = [
    "BaseEngine",
    "EngineError",
    "EngineNotAvailable",
    "InputError",
    "PageImage",
    "available_engines",
    "create_engine",
    "describe_engines",
    "load_image",
    "register",
    "register_lazy",
]
