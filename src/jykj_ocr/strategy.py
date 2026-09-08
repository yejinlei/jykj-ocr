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


def _own_elapsed(inner: Any, image: Any) -> tuple:
    """Time one engine call and return ``(result, elapsed_ms)``.

    ``BestofEngine.recognise`` rewrites ``result.elapsed_ms`` with the total
    wall clock after every engine has run, so ``bestof-fastest`` must be
    scored on the figure captured *before* that rewrite. Without this helper
    every candidate scored ``elapsed_ms=0`` and the mode resolved to a tie by
    config order. A value the engine already set is kept as-is.
    """
    started = time.perf_counter()
    result = inner.recognise(image)
    own = getattr(result, "elapsed_ms", 0) or int((time.perf_counter() - started) * 1000)
    return result, own


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

    @staticmethod
    def _metrics(result: OCRResult) -> Dict[str, Any]:
        """Per-candidate numbers for the decision trace."""
        regions = [r for r in result.regions if (r.text or "").strip()]
        confs = [r.confidence for r in regions]
        return {
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "region_count": len(result.regions),
            "char_count": len(result.text or ""),
            "garbled_layout": _is_garbled(result),
        }

    def _trace(self, reason: str, accepted: Dict[str, Any],
               rejected: List[Dict[str, Any]], fallback: bool,
               errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Shape the decision trace shared by the seq* / cascade* family.

        ``accepted`` is the entry that won; ``rejected`` holds the candidates
        the retry predicate refused, in the order they happened.
        """
        trace = {
            "selected": accepted["engine"],
            "reason": reason,
            "engine_order": self.engines(),
            "retries": self.retries,
            "accepted": [accepted],
            "rejected": rejected,
            "fallback": fallback,
        }
        if errors:
            trace["errors"] = errors
        return trace

    def recognise(self, image: Any) -> OCRResult:
        """Run the engine chain and return the best result found."""
        if not self._engines:
            raise StrategyError("no engines configured for the strategy")

        started = time.perf_counter()
        attempts: List[OCRResult] = []
        rejected: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        errors: List[Dict[str, Any]] = []

        for engine in self._engines:
            for attempt in range(self.retries + 1):
                label = f"{engine.name} (attempt {attempt + 1}/{self.retries + 1})"
                LOGGER.debug("strategy: trying %s", label)
                try:
                    result, own = _own_elapsed(engine, image)
                except Exception as exc:
                    last_error = exc
                    errors.append(
                        {"engine": engine.name,
                         "error": f"{type(exc).__name__}: {exc}"}
                    )
                    LOGGER.warning("strategy: %s failed: %s", label, exc)
                    continue
                entry = {
                    "engine": result.engine or engine.name,
                    "model": result.model,
                    "attempt": attempt + 1,
                    "own_elapsed_ms": own,
                    "ok": result.ok,
                    **self._metrics(result),
                }
                attempts.append(result)
                if self._acceptable(engine.config, result):
                    # Match BestofEngine's accounting: total wall-clock across
                    # all wrapped engines, including rejected attempts and
                    # retries. Engines don't set elapsed_ms themselves; the
                    # strategy layer owns it. Without this line, seq-family
                    # presets reported elapsed_ms=0 in HTTP responses.
                    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
                    result.score = entry["mean_confidence"]
                    result.score_mode = "mean_confidence"
                    result.decision = self._trace(
                        "first_accepted", entry, rejected, False, errors,
                    )
                    result.decision["total_elapsed_ms"] = result.elapsed_ms
                    return result
                rejected.append(entry)
                if attempt < self.retries:
                    LOGGER.info("strategy: %s result rejected, retrying", label)

        if attempts:
            best = max(attempts, key=lambda r: (r.ok, len(r.text or "")))
            if best.ok:
                best.elapsed_ms = int((time.perf_counter() - started) * 1000)
                best.score = self._metrics(best)["mean_confidence"]
                best.score_mode = "mean_confidence"
                # Every candidate was refused by the retry predicate; the
                # best-effort result is the longest usable one. That is a
                # different decision than a clean accept, so say so.
                best.decision = self._trace(
                    "all_rejected_fallback",
                    {"engine": best.engine, "model": best.model, "attempt": 0,
                     "own_elapsed_ms": 0, "ok": True, **self._metrics(best)},
                    rejected, True, errors,
                )
                best.decision["total_elapsed_ms"] = best.elapsed_ms
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


_CJK_PUNCT = set("，。！？、；：""''（）《》…—""''·—")



def _fluency_components(result: OCRResult) -> Dict[str, float]:
    """Parts of :func:`_fluency_score`, laid out individually.

    Returned as a dict so the decision trace can show *which* signal decided
    the ranking, not just the total.
    """
    parts = _text_parts(result)
    if not parts:
        parts = [(result.text or "").strip()]
    total_chars = sum(len(p) for p in parts)
    if not result.ok or total_chars == 0:
        return {
            "phrase_bonus": 0.0,
            "punct_bonus": 0.0,
            "frag_penalty": 0.0,
            "single_chars": 0,
            "mean_phrase_len": 0.0,
        }

    # Mean phrase length: rewards longer coherent phrases (cap at 30 chars).
    mean_phrase = total_chars / len(parts)
    phrase_bonus = min(15.0, mean_phrase)

    # Single-char fragment penalty: 166 single-char fragments would hit the cap.
    single_chars = sum(1 for p in parts if len(p) == 1)
    frag_penalty = min(_FLUENCY_SINGLE_CHAR_PENALTY * single_chars,
                       _FLUENCY_SINGLE_CHAR_CAP)

    # A few sentence markers is enough to earn the full punctuation bonus.
    text = result.text or ""
    punct_ratio = sum(1 for c in text if c in _CJK_PUNCT) / max(1, len(text))
    punct_bonus = min(5.0, punct_ratio * 200.0)

    return {
        "phrase_bonus": round(phrase_bonus, 3),
        "punct_bonus": round(punct_bonus, 3),
        "frag_penalty": round(frag_penalty, 3),
        "single_chars": single_chars,
        "mean_phrase_len": round(mean_phrase, 2),
    }


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
    if not (result.text or "").strip():
        return 0.0
    c = _fluency_components(result)
    return c["phrase_bonus"] + c["punct_bonus"] - c["frag_penalty"]


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
        self.score_mode = (score_mode or "smart").strip().lower()
        self.score_fn = resolve_bestof_score(self.score_mode)

    def engines(self) -> List[str]:
        return [e.name for e in self._engines]

    def recognise(self, image: Any) -> OCRResult:
        """Run every engine, pick the highest-scoring ``ok`` result."""
        scored: List[Dict[str, Any]] = []
        results: List[Any] = []
        last_error: Optional[Exception] = None
        errors: List[Dict[str, Any]] = []
        started = time.perf_counter()

        for engine in self._engines:
            try:
                result, own = _own_elapsed(engine, image)
            except Exception as exc:
                last_error = exc
                errors.append(
                    {"engine": engine.name,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                LOGGER.warning("bestof: %s failed: %s", engine.name, exc)
                continue
            results.append(result)
            # During scoring this is the engine's *own* latency, which is what
            # _score_fastest reads. The winner gets overwritten with the total
            # wall clock below; the per-engine figure survives in the trace.
            result.elapsed_ms = own
            metrics = self._metrics(result)
            score = self.score_fn(result)
            scored.append({
                "engine": result.engine or engine.name,
                "model": result.model,
                "own_elapsed_ms": own,
                "ok": result.ok,
                "score": round(score, 4),
                "mean_confidence": metrics["mean_confidence"],
                "region_count": metrics["region_count"],
                "char_count": metrics["char_count"],
                "garbled_layout": metrics["garbled_layout"],
                "detail": self._score_detail(result, score),
            })
            LOGGER.debug(
                "bestof: %s -> score=%.2f ok=%s",
                result.engine or engine.name, score, result.ok,
            )

        if not scored:
            raise StrategyError(
                f"bestof: all {len(self._engines)} engine(s) failed"
                + (f": {last_error}" if last_error else "")
            ) from (last_error if last_error else None)

        # Highest score wins; an ``ok`` result beats a good score on an empty
        # one. Ties keep config order (sorted is stable).
        ranked = sorted(zip(scored, results), key=lambda p: p[0]["score"],
                        reverse=True)
        picked = next((p for p in ranked if p[1].ok), None)
        if picked is None:
            raise StrategyError(
                f"bestof exhausted {len(self._engines)} engine(s), "
                f"{len(scored)} result(s), none ok"
            )
        winner = picked[1]
        # Total wall clock across every wrapped engine, mirroring
        # StrategyEngine. Engines do not set elapsed_ms themselves.
        winner.elapsed_ms = int((time.perf_counter() - started) * 1000)
        winner.score = picked[0]["score"]
        winner.score_mode = self.score_mode
        winner.decision = {
            "selected": winner.engine,
            "reason": f"highest_{self.score_mode}_score",
            "score_mode": self.score_mode,
            "engine_order": self.engines(),
            "ranked": [c for c, _ in ranked],
            "total_elapsed_ms": winner.elapsed_ms,
        }
        if errors:
            winner.decision["errors"] = errors
        return winner

    def _metrics(self, result: OCRResult) -> Dict[str, Any]:
        """Per-candidate numbers, shared with :class:`StrategyEngine`."""
        regions = [r for r in result.regions if (r.text or "").strip()]
        confs = [r.confidence for r in regions]
        return {
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "region_count": len(result.regions),
            "char_count": len(result.text or ""),
            "garbled_layout": _is_garbled(result),
        }

    def _score_detail(self, result: OCRResult, score: float) -> Dict[str, Any]:
        """Break the score down so the ranking is auditable.

        Only the components the active mode actually uses are reported — a
        ``fastest`` trace should not carry fluency numbers it never read.
        """
        mode = self.score_mode
        if not result.ok:
            return {"excluded": "not_ok"}
        if mode == "smart":
            return {
                "confidence_pts": round(_mean_confidence(result) * 100.0, 3),
                "garbled_penalty": -_GARBLED_PENALTY if _is_garbled(result) else 0.0,
                "length_nudge": round(min(1.0, len(result.text or "")), 3),
                **_fluency_components(result),
            }
        if mode == "fluency":
            return _fluency_components(result)
        if mode == "highest_confidence":
            return {"mean_confidence": round(_mean_confidence(result), 4)}
        if mode == "longest":
            return {"char_count": len(result.text or "")}
        if mode == "fastest":
            return {"own_elapsed_ms": getattr(result, "elapsed_ms", 0)}
        return {"score": round(score, 4)}

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
