# -*- coding: utf-8 -*-
"""Public API for jykj_ocr."""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import Config, EngineConfig, load_config
from .engine.inputs import attach_pil, load as load_source
from .engine.registry import (
    apply_strategy_preset,
    build_engine,
    build_pipeline,
    build_strategy,
    engines_from_config,
    resolve_retry_check,
)
from .models import OCRResult, rebuild_text_from_regions
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


def _finish(result: OCRResult, cfg: Config) -> OCRResult:
    """Apply output post-processing the effective config asks for."""
    if cfg.output_value("reorder_lines"):
        result.text = rebuild_text_from_regions(result)
    return result


def ocr(
    source: str,
    *,
    engine: Optional[str] = None,
    config: Optional[Config] = None,
    config_path: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 200,
    retries: int = 1,
    strategy_name: Optional[str] = None,
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
        strategy_name: One-shot preset — ``local`` / ``vl`` / ``fallback`` /
            ``quality`` — applied to a throwaway copy of the config, so the
            run ignores the configured chain without touching any files.
    """
    cfg = config or load_config(config_path)
    if strategy_name:
        cfg = apply_strategy_preset(cfg, strategy_name)

    max_pages = max_pages if max_pages is not None else cfg.pdf.get("max_pages")
    dpi = int(dpi or cfg.pdf.get("dpi", 200) or 200)

    pages = load_source(source, max_pages=max_pages, dpi=dpi)
    if not pages:
        return []

    if engine:
        # build_engine normalises aliases (``sf`` -> ``siliconflow``) and falls
        # back to a default EngineConfig when the name is not in the config.
        pipeline = TimedOCR(build_engine(engine, cfg))
    elif strategy_name:
        pipeline = build_pipeline(cfg)
    else:
        pipeline = build_strategy(
            engines_from_config(cfg),
            retries=int(cfg.strategy_value("max_retries", retries)),
            retry_check=resolve_retry_check(cfg.strategy),
        )

    return [_finish(pipeline.recognise(attach_pil(page)), cfg) for page in pages]


def ocr_to_text(
    source: str,
    *,
    engine: Optional[str] = None,
    config: Optional[Config] = None,
    config_path: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 200,
    strategy_name: Optional[str] = None,
) -> str:
    """Convenience wrapper returning concatenated markdown for all pages."""
    results = ocr(
        source,
        engine=engine,
        config=config,
        config_path=config_path,
        max_pages=max_pages,
        dpi=dpi,
        strategy_name=strategy_name,
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
