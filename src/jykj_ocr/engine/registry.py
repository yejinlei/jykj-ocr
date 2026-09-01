# -*- coding: utf-8 -*-
"""Builds engine instances from config and assembles the strategy chain.

This is the single place where configuration becomes behaviour: which engines
run, in what order, and when a result is good enough to return.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import copy

from ..config import Config, EngineConfig, normalise_engine
from ..strategy import (
    StrategyEngine,
    StrategyFn,
    combine_predicates,
    should_retry_line_overlap,
    should_retry_low_confidence,
    should_retry_no_text,
)
from . import base as engine_pkg

#: Engines known to be remote (OpenAI-compatible endpoints). This is only an
#: allow-list — any OTHER registered engine is treated as local by the
#: presets, so new engines (PaddleOCR, Tesseract, cloud vendors...) need no
#: change here: register them and ``local``/``vl`` pick them up automatically.
#: Set ``JYKJ_OCR_REMOTE_ENGINES="a,b"`` to add more remote names.
_DEFAULT_REMOTE_ENGINES = ("siliconflow", "multimodal")


def remote_engines() -> tuple:
    """Remote engine names: built-in allow-list plus env override."""
    extra = os.getenv("JYKJ_OCR_REMOTE_ENGINES", "")
    names = [n.strip().lower() for n in extra.split(",") if n.strip()]
    return tuple(dict.fromkeys(list(_DEFAULT_REMOTE_ENGINES) + names))


#: Named strategy presets understood by ``strategy.name`` / ``strategy_name=``.
STRATEGY_PRESETS = ("local", "vl", "fallback", "quality")


def build_engine(name: str, config: Config) -> engine_pkg.BaseEngine:
    """Create an engine by name, falling back to config defaults when present."""
    target = normalise_engine(name)
    found = config.find_engine(target) if config else None
    engine_config = found or EngineConfig(name=target)
    return engine_pkg.create_engine(target, engine_config)


def build_strategy(
    engines: Sequence[Any], *, retries: int = 1, retry_check: Optional[StrategyFn] = None
) -> StrategyEngine:
    """Wrap engines in a retry-capable strategy chain."""
    return StrategyEngine(engines, retries=retries, retry_check=retry_check)


def _make_retry_check(mode: str, strategy: Dict[str, Any]) -> Optional[StrategyFn]:
    """Build one named retry predicate (without the composite modes)."""
    if mode in ("none", "first_success"):
        return None
    if mode == "low_confidence":
        return should_retry_low_confidence(float(strategy.get("min_confidence", 0.7)))
    if mode == "line_overlap":
        return should_retry_line_overlap
    return should_retry_no_text


def resolve_retry_check(strategy: Dict[str, Any]) -> Optional[StrategyFn]:
    """Turn ``strategy`` config into a retry predicate, or ``None`` for default.

    ``retry_mode`` values:
      - ``no_text``        (default) retry when no text was recognised
      - ``low_confidence`` retry when mean confidence < ``min_confidence``
      - ``line_overlap``   retry when no text OR a garbled layout (窜行)
      - ``any``            retry when low confidence OR garbled layout
      - ``none``           accept the first successful result
    """
    if not strategy:
        return None
    mode = str(strategy.get("retry_mode", "no_text")).lower()
    if mode == "any":
        return combine_predicates(
            _make_retry_check("low_confidence", strategy),
            should_retry_line_overlap,
        )
    return _make_retry_check(mode, strategy)


def engines_from_config(
    config: Config, names: Optional[Sequence[str]] = None
) -> List[Any]:
    """Instantiate engines, in config order unless ``names`` overrides it."""
    wanted = list(names) if names else [e.name for e in config.enabled_engines()]
    result: List[Any] = []
    for name in wanted:
        if not name:
            continue
        result.append(build_engine(name, config))
    if not result:
        result.append(build_engine("multimodal", config))
    return result


def apply_strategy_preset(config: Config, name: str) -> Config:
    """Return a copy of ``config`` reshaped by a named preset (one-shot semantics).

    Presets (see ``STRATEGY_PRESETS``):
      - ``local``    only local (rapidocr-family) engines; plain no_text retry
      - ``vl``       only remote VL engines; first enabled remote engine wins
      - ``fallback`` every enabled engine, tried in config order
      - ``quality``  fallback + line-overlap demotion + reading-order rebuild

    ``name`` is written into ``strategy["name"]`` so downstream code (text
    reorder) can see which preset produced this config. Unknown names raise
    ``ValueError`` listing the valid presets.

    Adding engines later: a newly registered engine needs no preset change —
    ``local`` keeps it (unless its name is in ``remote_engines()``), ``vl``
    excludes it, and ``fallback``/``quality`` use whatever ``enabled`` flag the
    config gives it. Mark a new remote-only vendor by adding its name to
    ``JYKJ_OCR_REMOTE_ENGINES``.
    """
    key = (name or "").strip().lower()
    if key not in STRATEGY_PRESETS:
        raise ValueError(
            f"unknown strategy {name!r}; choose one of {', '.join(STRATEGY_PRESETS)}"
        )
    cfg = copy.deepcopy(config)
    strategy = dict(cfg.strategy)
    output = dict(cfg.output)

    def _is_remote(engine: EngineConfig) -> bool:
        return normalise_engine(engine.name) in remote_engines()

    if key == "local":
        for engine in cfg.engines:
            engine.enabled = not _is_remote(engine)
        strategy["retry_mode"] = "no_text"
        output.pop("reorder_lines", None)
    elif key == "vl":
        remotes = [e for e in cfg.engines if _is_remote(e)]
        if not any(e.enabled for e in remotes):
            if not remotes:
                raise ValueError(
                    "strategy 'vl' needs a remote engine (siliconflow/multimodal) "
                    "configured; none found"
                )
            for engine in remotes:
                engine.enabled = True
        for engine in cfg.engines:
            if not _is_remote(engine):
                engine.enabled = False
        strategy.setdefault("retry_mode", "no_text")
        output.pop("reorder_lines", None)
    elif key == "fallback":
        strategy.setdefault("retry_mode", "no_text")
        output.pop("reorder_lines", None)
    else:  # quality
        strategy["retry_mode"] = "any"
        output["reorder_lines"] = True

    strategy["name"] = key
    cfg.strategy = strategy
    cfg.output = output
    return cfg


def build_pipeline(
    config: Config, engine_name: Optional[str] = None
) -> StrategyEngine:
    """Assemble the strategy chain exactly the way CLI / API / package all do.

    ``engine_name`` forces a single engine and ignores the configured chain;
    otherwise the enabled engines run in config order under the resolved retry
    predicate. A ``strategy.name`` in the config is treated as documentation of
    the active preset — presets are applied earlier via
    :func:`apply_strategy_preset`, never re-applied here.
    """
    if engine_name:
        engines = [build_engine(engine_name, config)]
    else:
        engines = engines_from_config(config)
    strategy_cfg = config.strategy or {}
    return build_strategy(
        engines,
        retries=int(strategy_cfg.get("max_retries", 1)),
        retry_check=resolve_retry_check(strategy_cfg),
    )


__all__ = [
    "STRATEGY_PRESETS",
    "StrategyEngine",
    "apply_strategy_preset",
    "build_engine",
    "build_pipeline",
    "build_strategy",
    "engines_from_config",
    "remote_engines",
    "resolve_retry_check",
]
