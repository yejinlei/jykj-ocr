# -*- coding: utf-8 -*-
"""Strategy: decide *which* engine to call, and when to retry with another.

This module is deliberately thin — it orchestrates, never implements OCR itself.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .models import OCRResult

LOGGER = logging.getLogger(__name__)

StrategyFn = Callable[[Any, OCRResult], bool]


def should_retry_no_text(_: Any, result: OCRResult) -> bool:
    """Retry when the engine produced no usable text."""
    return not result.ok


def should_retry_low_confidence(min_confidence: float) -> StrategyFn:
    """Retry when mean region confidence is below ``min_confidence``."""

    def _check(_: Any, result: OCRResult) -> bool:
        regions = [r.confidence for r in result.regions if r.text.strip()]
        if not regions:
            return not result.ok
        return sum(regions) / len(regions) < min_confidence

    return _check


def should_retry_line_overlap(_: Any, result: OCRResult) -> bool:
    """Retry when no text OR a garbled layout (merged/overlapping rows)."""
    from .models import detect_line_overlap

    if not result.ok:
        return True
    return detect_line_overlap(result)


def combine_predicates(*predicates: Optional[StrategyFn]) -> Optional[StrategyFn]:
    """OR several retry predicates; ``None`` entries are ignored."""
    active = [p for p in predicates if p is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _combined(config: Any, result: OCRResult) -> bool:
        return any(p(config, result) for p in active)

    return _combined


class StrategyError(Exception):
    """Raised when no engine in the chain produced a usable result."""


class StrategyEngine:
    """Retry-driven engine chain.

    The first configured engine is tried first; if its result fails the retry
    predicate, it is retried up to ``retries`` more times, then the next engine
    in the chain is tried.
    """

    def __init__(
        self,
        engines: Sequence[Any],
        *,
        retries: int = 1,
        retry_check: Optional[StrategyFn] = None,
    ) -> None:
        self._engines: List[Any] = list(engines)
        self.retries = max(0, int(retries))
        self.retry_check: Optional[StrategyFn] = retry_check

    def engines(self) -> List[str]:
        return [e.name for e in self._engines]

    def _acceptable(self, ctx: Any, result: OCRResult) -> bool:
        if self.retry_check is None:
            return result.ok
        try:
            return not self.retry_check(ctx, result)
        except Exception as exc:  # a broken predicate must not kill recognition
            LOGGER.warning("retry predicate raised %s; accepting result", exc)
            return True

    def recognise(self, image: Any) -> OCRResult:
        """Run the engine chain and return the best result found."""
        if not self._engines:
            raise StrategyError("no engines configured for the strategy")

        attempts: List[OCRResult] = []
        last_error: Optional[Exception] = None

        for engine in self._engines:
            for attempt in range(self.retries + 1):
                label = f"{engine.name} (attempt {attempt + 1}/{self.retries + 1})"
                LOGGER.debug("strategy: trying %s", label)
                try:
                    result = engine.recognise(image)
                except Exception as exc:
                    last_error = exc
                    LOGGER.warning("strategy: %s failed: %s", label, exc)
                    continue
                attempts.append(result)
                if self._acceptable(engine.config, result):
                    return result
                if attempt < self.retries:
                    LOGGER.info("strategy: %s result rejected, retrying", label)

        if attempts:
            best = max(attempts, key=lambda r: (r.ok, len(r.text or "")))
            if best.ok:
                return best
        if last_error is not None and not attempts:
            raise StrategyError(f"all engines failed: {last_error}") from last_error
        raise StrategyError(
            f"strategy exhausted {len(self._engines)} engine(s), "
            f"{len(attempts)} attempt(s), no usable text"
        )

    def summary(self) -> Dict[str, Any]:
        return {"engines": self.engines(), "retries": self.retries}


class TimedOCR:
    """Decorates an engine's ``recognise`` with wall-clock timing."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def recognise(self, image: Any) -> OCRResult:
        started = time.perf_counter()
        result = self._inner.recognise(image)
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result


__all__ = [
    "StrategyEngine",
    "StrategyError",
    "TimedOCR",
    "combine_predicates",
    "should_retry_line_overlap",
    "should_retry_low_confidence",
    "should_retry_no_text",
]
