# -*- coding: utf-8 -*-
"""Public API for jykj_ocr."""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import Config, EngineConfig, load_config
from .engine import create_engine
from .engine.inputs import attach_pil, load as load_source
from .engine.registry import build_strategy, engines_from_config, resolve_retry_check
from .models import OCRResult
from .strategy import StrategyEngine, StrategyError, TimedOCR

__version__ = "0.1.0"


def get_logger(name: str = "jykj_ocr") -> logging.Logger:
    """Configure module logging once and return the root logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def ocr(
    source: str,
    *,
    engine: Optional[str] = None,
    config: Optional[Config] = None,
    config_path: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 200,
    retries: int = 1,
) -> List[OCRResult]:
    """Run OCR on ``source`` and return one :class:`OCRResult` per page.

    Args:
        source: Path to an image/PDF, or an ``http(s)://`` URL.
        engine: Engine to force (e.g. ``"rapidocr"``, ``"siliconflow"``).
            When ``None``, the strategy chain from config is used.
        config: A pre-built :class:`Config`; overrides ``config_path``.
        config_path: Path to a config file (or ``JYKJ_OCR_CONFIG`` env var).
        max_pages: Cap on PDF pages to process.
        dpi: PDF rasterisation density.
        retries: Per-engine retry count used by the strategy.
    """
    cfg = config or load_config(config_path)

    max_pages = max_pages if max_pages is not None else cfg.pdf.get("max_pages")
    dpi = int(dpi or cfg.pdf.get("dpi", 200) or 200)

    pages = load_source(source, max_pages=max_pages, dpi=dpi)
    if not pages:
        return []

    if engine:
        pipeline = TimedOCR(create_engine(engine, cfg.find_engine(engine)))
    else:
        pipeline = build_strategy(
            engines_from_config(cfg),
            retries=int(cfg.strategy_value("max_retries", retries)),
            retry_check=resolve_retry_check(cfg.strategy),
        )

    return [pipeline.recognise(attach_pil(page)) for page in pages]


def ocr_to_text(
    source: str,
    *,
    engine: Optional[str] = None,
    config: Optional[Config] = None,
    config_path: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 200,
) -> str:
    """Convenience wrapper returning concatenated markdown for all pages."""
    results = ocr(
        source,
        engine=engine,
        config=config,
        config_path=config_path,
        max_pages=max_pages,
        dpi=dpi,
    )
    return "\n\n".join(r.to_markdown() for r in results if r.ok)


__all__ = [
    "Config",
    "EngineConfig",
    "OCRResult",
    "StrategyEngine",
    "StrategyError",
    "TimedOCR",
    "get_logger",
    "load_config",
    "ocr",
    "ocr_to_text",
    "__version__",
]
