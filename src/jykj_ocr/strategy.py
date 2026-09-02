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


# ---------------------------------------------------------------------------
# Bestof: run every engine once, pick the best by a chosen score.
# ---------------------------------------------------------------------------

#: Score mode names understood by ``BestofEngine``.
BESTOF_MODES = ("smart", "fastest", "highest_confidence", "longest", "fluency")

_GARBLED_PENALTY = 20.0  # synthetic confidence points subtracted for garbled layout
_FLUENCY_SINGLE_CHAR_PENALTY = 0.3  # fluency pts lost per single-char region (capped)
_FLUENCY_SINGLE_CHAR_CAP = 25.0      # max single-char penalty


def _mean_confidence(result: OCRResult) -> float:
    regions = [r.confidence for r in result.regions if (r.text or "").strip()]
    return sum(regions) / len(regions) if regions else 0.0


def _is_garbled(result: OCRResult) -> bool:
    try:
        from .models import detect_line_overlap
    except Exception:
        return False
    return detect_line_overlap(result)


def _text_parts(result: OCRResult) -> List[str]:
    """Non-empty, stripped text from each region — used by fluency scoring."""
    return [
        (r.text or "").strip()
        for r in result.regions
        if (r.text or "").strip()
    ]


def _fluency_score(result: OCRResult) -> float:
    """Semantic fluency: how much the output reads like natural language.

    Signals:
      - mean phrase length (chars per region): longer phrases → more fluent
      - single-char region ratio: too many fragments → penalised
      - CJK punctuation ratio: presence of sentence markers → more fluent

    Returns a score in roughly ``[-_FLUENCY_SINGLE_CHAR_CAP, +40]``.
    """
    if not result.ok:
        return float("-inf")
    text = result.text or ""
    if not text.strip():
        return 0.0

    parts = _text_parts(result)
    if not parts:
        parts = [text]

    total_chars = sum(len(p) for p in parts)
    if total_chars == 0:
        return 0.0

    # Mean phrase length: rewards longer coherent phrases (cap at 30 chars).
    mean_phrase = total_chars / len(parts)
    phrase_bonus = min(15.0, mean_phrase)

    # Single-char fragment penalty: 166 single-char fragments would hit the cap.
    single_chars = sum(1 for p in parts if len(p) == 1)
    frag_penalty = min(_FLUENCY_SINGLE_CHAR_CAP,
                       single_chars * _FLUENCY_SINGLE_CHAR_PENALTY)

    # CJK punctuation: presence of sentence markers = reads like natural language.
    _CJK_PUNCT = set("，。！？、；：""''（）《》…—""''·—")
    punct_ratio = sum(1 for c in text if c in _CJK_PUNCT) / max(1, len(text))
    punct_bonus = min(5.0, punct_ratio * 200.0)  # a few punct → up to 5 pts

    return phrase_bonus + punct_bonus - frag_penalty


def _score_smart(result: OCRResult) -> float:
    """Composite score: confidence - garbled penalty + text-length + fluency."""
    if not result.ok:
        return float("-inf")
    s = _mean_confidence(result) * 100.0
    if _is_garbled(result):
        s -= _GARBLED_PENALTY
    # gentle nudge toward non-empty text
    s += min(1.0, len((result.text or "") or ""))
    # fluency bonus (heavily weighted — fluency is the human-preferred signal)
    s += _fluency_score(result)
    return s


def _score_fastest(result: OCRResult) -> float:
    return -float(getattr(result, "elapsed_ms", 0) or 0)  # lower latency wins


def _score_highest_confidence(result: OCRResult) -> float:
    return _mean_confidence(result) if result.ok else float("-inf")


def _score_longest(result: OCRResult) -> float:
    return len((result.text or "") or "") if result.ok else float("-inf")


def _score_fluency(result: OCRResult) -> float:
    """Pick the result that reads most like natural language."""
    return _fluency_score(result) if result.ok else float("-inf")


_BESTOF_SCORE: Dict[str, Callable[[OCRResult], float]] = {
    "smart": _score_smart,
    "fastest": _score_fastest,
    "highest_confidence": _score_highest_confidence,
    "longest": _score_longest,
    "fluency": _score_fluency,
}


def resolve_bestof_score(mode: Optional[str]) -> Callable[[OCRResult], float]:
    key = (mode or "smart").strip().lower()
    try:
        return _BESTOF_SCORE[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown bestof mode {mode!r}; choose one of "
            f"{', '.join(BESTOF_MODES)}"
        ) from exc


class BestofEngine:
    """Run all wrapped engines once and return the highest-scoring result.

    Used by the ``bestof`` strategy preset. Differs from :class:`StrategyEngine`
    (which stops on the first acceptable result) — ``bestof`` *always* runs
    every engine and picks the winner by ``score_fn``. This is slower but
    guarantees you get the best of what's available, e.g. choosing between a
    fast local OCR and a slow remote VL model.
    """

    def __init__(
        self,
        engines: Sequence[Any],
        *,
        score_mode: str = "smart",
    ) -> None:
        self._engines: List[Any] = list(engines)
        if not self._engines:
            raise StrategyError("bestof needs at least one engine")
        self.score_fn = resolve_bestof_score(score_mode)

    def engines(self) -> List[str]:
        return [e.name for e in self._engines]

    def recognise(self, image: Any) -> OCRResult:
        """Run every engine, pick the highest-scoring ``ok`` result."""
        scored: List[tuple] = []
        last_error: Optional[Exception] = None

        for engine in self._engines:
            try:
                result = engine.recognise(image)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("bestof: %s failed: %s", engine.name, exc)
                continue
            s = self.score_fn(result)
            scored.append((s, result))
            LOGGER.debug(
                "bestof: %s -> score=%.2f ok=%s", engine.name, s, result.ok
            )

        if not scored:
            raise StrategyError(
                f"bestof: all {len(self._engines)} engine(s) failed"
                + (f": {last_error}" if last_error else "")
            ) from (last_error if last_error else None)

        # prefer highest score, breaking ties by engine order (already inserted)
        scored.sort(key=lambda kv: kv[0], reverse=True)
        winner = scored[0][1]
        if not winner.ok:
            raise StrategyError(
                f"bestof exhausted {len(self._engines)} engine(s), "
                f"{len(scored)} result(s), none ok"
            )
        return winner

    def summary(self) -> Dict[str, Any]:
        return {
            "engines": self.engines(),
            "mode": getattr(self.score_fn, "__name__", "custom"),
        }


__all__ = [
    "StrategyEngine",
    "StrategyError",
    "TimedOCR",
    "BestofEngine",
    "BESTOF_MODES",
    "combine_predicates",
    "should_retry_line_overlap",
    "should_retry_low_confidence",
    "should_retry_no_text",
]
