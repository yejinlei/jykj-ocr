# -*- coding: utf-8 -*-
"""Builds engine instances from config and assembles the strategy chain.

This is the single place where configuration becomes behaviour: which engines
run, in what order, and when a result is good enough to return.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..config import Config, EngineConfig, normalise_engine
from ..strategy import StrategyEngine, StrategyFn, should_retry_no_text
from . import base as engine_pkg


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


def resolve_retry_check(strategy: Dict[str, Any]) -> Optional[StrategyFn]:
    """Turn ``strategy`` config into a retry predicate, or ``None`` for default.

    ``retry_mode`` values:
      - ``no_text``        (default) retry when no text was recognised
      - ``low_confidence`` retry when mean confidence < ``min_confidence``
      - ``none``           accept the first successful result
    """
    if not strategy:
        return None
    mode = str(strategy.get("retry_mode", "no_text")).lower()
    if mode in ("none", "first_success"):
        return None
    if mode == "low_confidence":
        from ..strategy import should_retry_low_confidence

        return should_retry_low_confidence(float(strategy.get("min_confidence", 0.7)))
    return should_retry_no_text


def engines_from_config(
    config: Config, names: Optional[Sequence[str]] = None
) -> List[Any]:
    """Instantiate engines, in config order unless ``names`` overrides it."""
    wanted = list(names) if names else [e.name for e in config.engines]
    result: List[Any] = []
    for name in wanted:
        if not name:
            continue
        result.append(build_engine(name, config))
    if not result:
        result.append(build_engine("multimodal", config))
    return result


__all__ = [
    "StrategyEngine",
    "build_engine",
    "build_strategy",
    "engines_from_config",
    "resolve_retry_check",
]
