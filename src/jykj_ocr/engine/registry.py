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
    BestofEngine,
    StrategyEngine,
    StrategyFn,
    combine_predicates,
    resolve_bestof_score,
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
#
# ``seq*`` presets all use :class:`StrategyEngine` (first-acceptable-wins, in
# engine order). They differ only by retry predicate, ``max_retries`` and
# whether the final result is re-ordered by position::
#
#   seq                 retry_mode=no_text,     max_retries=config
#   seq-any             retry_mode=any,         reorder=on
#   seq-low_conf        retry_mode=low_confidence
#   seq-line_overlap    retry_mode=line_overlap
#
# ``cascade*`` presets also use :class:`StrategyEngine` but set
# ``max_retries=0`` — a rejected attempt jumps straight to the next engine
# instead of retrying the same engine. Use ``cascade*`` when the same
# engine is unlikely to succeed on a second run (e.g. local OCR returning
# garbled text won't improve with another pass on the same image)::
#
#   cascade              retry_mode=no_text,         max_retries=0
#   cascade-low_conf     retry_mode=low_confidence,  max_retries=0
#   cascade-line_overlap retry_mode=line_overlap,    max_retries=0
#
# ``bestof*`` presets use :class:`BestofEngine` — every engine runs once and
# the winner is chosen by a score function; the result is never re-ordered::
#
#   bestof-smart              confidence + garbled penalty + text length + fluency
#   bestof-fastest            lowest elapsed_ms
#   bestof-confidence         highest mean region confidence
#   bestof-longest            longest text
#   bestof-fluency            most natural-sounding language (phrase density +
#                             CJK punctuation, penalises single-char fragments)
#   bestof:<mode>             syntax alias for any of the modes above
#
# Legacy aliases kept for backwards compatibility:
#   fallback  == seq
#   quality   == seq-any
STRATEGY_PRESETS = (
    "local",
    "vl",
    "seq",
    "seq-any",
    "seq-low_conf",
    "seq-line_overlap",
    "cascade",
    "cascade-low_conf",
    "cascade-line_overlap",
    "bestof",
    "bestof-smart",
    "bestof-fastest",
    "bestof-confidence",
    "bestof-longest",
    "bestof-fluency",
    # legacy
    "fallback",
    "quality",
)

#: Allowed ``retry_mode`` values. ``resolve_retry_check`` treats anything
#: outside this list as ``no_text`` — reject the request here instead.
VALID_RETRY_MODES = frozenset(
    ("no_text", "low_confidence", "line_overlap", "any", "none")
)

#: Allowed ``bestof_score_mode`` values; mirrors :func:`resolve_bestof_score`.
VALID_SCORE_MODES = frozenset(
    ("smart", "fastest", "highest_confidence", "longest", "fluency")
)

#: (retry_mode, reorder_lines, is_bestof, bestof_score_mode, max_retries)
#:
#: ``max_retries=None`` means "leave the configured value alone" — this is
#: what ``seq*`` / ``bestof*`` presets do. An explicit integer overrides the
#: config value; ``cascade*`` uses ``0`` to skip same-engine retries.
_SEQ_PRESETS = {
    "local": ("no_text", False, False, None, None),
    "vl": ("no_text", False, False, None, None),
    "seq": ("no_text", False, False, None, None),
    "seq-any": ("any", True, False, None, None),
    "seq-low_conf": ("low_confidence", False, False, None, None),
    "seq-line_overlap": ("line_overlap", False, False, None, None),
    "cascade": ("no_text", False, False, None, 0),
    "cascade-low_conf": ("low_confidence", False, False, None, 0),
    "cascade-line_overlap": ("line_overlap", False, False, None, 0),
    # legacy aliases
    "fallback": ("no_text", False, False, None, None),
    "quality": ("any", True, False, None, None),
    # bestof presets
    "bestof": (None, False, True, "smart", None),
    "bestof-smart": (None, False, True, "smart", None),
    "bestof-fastest": (None, False, True, "fastest", None),
    "bestof-confidence": (None, False, True, "highest_confidence", None),
    "bestof-longest": (None, False, True, "longest", None),
    "bestof-fluency": (None, False, True, "fluency", None),
}


def describe_presets() -> Dict[str, Dict[str, Any]]:
    """Structured description of every named strategy preset.

    Consumed by ``GET /presets`` so callers can discover the full family
    (including ``bestof-fluency``, ``cascade*``…) instead of reading a flat
    string of names in the ``strategy_name`` description.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, (retry_mode, reorder, is_bestof, score_mode, max_retries) in _SEQ_PRESETS.items():
        out[name] = {
            "retry_mode": retry_mode,
            "reorder_lines": reorder,
            "is_bestof": is_bestof,
            "score_mode": score_mode if is_bestof else None,
            "max_retries": max_retries,
            "engine_scope": (
                "remote_vl_only" if name == "vl"
                else "local_only" if name == "local"
                else "all_enabled"
            ),
        }
    out["bestof:<mode>"] = {
        "is_bestof": True,
        "note": "colon syntax alias for bestof-<mode>",
    }
    return out


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
      - ``seq*``     first-acceptable-wins (:class:`StrategyEngine`); differ by
                     retry predicate and whether the final result is re-ordered
      - ``bestof*``  every engine runs once (:class:`BestofEngine`); the winner
                     is picked by a score function; never re-ordered

    ``name`` is written into ``strategy["name"]`` so downstream code (text
    reorder, pipeline assembly) can see which preset produced this config.
    Unknown names raise ``ValueError`` listing the valid presets.

    Adding engines later: a newly registered engine needs no preset change —
    ``local`` keeps it (unless its name is in ``remote_engines()``), ``vl``
    excludes it, and ``seq*``/``bestof*`` use whatever ``enabled`` flag the
    config gives it. Mark a new remote-only vendor by adding its name to
    ``JYKJ_OCR_REMOTE_ENGINES``.
    """
    key = (name or "").strip().lower()
    # ``bestof:<mode>`` syntax — validate the mode before looking up the preset.
    bestof_score_mode = None
    if key.startswith("bestof:"):
        mode = key[len("bestof:"):].strip()
        if not mode:
            raise ValueError(
                f"empty bestof mode {name!r}; choose one of {', '.join(STRATEGY_PRESETS)}"
            )
        try:
            resolve_bestof_score(mode)
        except ValueError:
            raise ValueError(
                f"unknown strategy {name!r}; choose one of {', '.join(STRATEGY_PRESETS)}"
            )
        bestof_score_mode = mode
        key = "bestof"

    if key not in STRATEGY_PRESETS:
        raise ValueError(
            f"unknown strategy {name!r}; choose one of {', '.join(STRATEGY_PRESETS)}"
        )

    cfg = copy.deepcopy(config)
    strategy = dict(cfg.strategy)
    output = dict(cfg.output)

    retry_mode, reorder_lines, is_bestof, _preset_bestof_mode, preset_max_retries = _SEQ_PRESETS[key]
    if is_bestof and bestof_score_mode is None:
        bestof_score_mode = _preset_bestof_mode

    def _is_remote(engine: EngineConfig) -> bool:
        return normalise_engine(engine.name) in remote_engines()

    # ``local`` / ``vl`` reshape which engines are enabled; all other presets
    # leave enabled flags alone.
    if key == "local":
        for engine in cfg.engines:
            engine.enabled = not _is_remote(engine)
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

    # Write the retry mode (bestof family has no retry predicate).
    if retry_mode is not None:
        strategy["retry_mode"] = retry_mode
    else:
        strategy.pop("retry_mode", None)

    # Write max_retries when the preset dictates a specific value (e.g.
    # ``cascade*`` = 0). Presets that leave it alone use ``None`` and keep
    # whatever the config already had — this is how ``seq*`` retains its
    # configured retries.
    if preset_max_retries is not None:
        strategy["max_retries"] = preset_max_retries
    else:
        strategy.pop("max_retries", None)

    # Bestof presets mark themselves so build_pipeline knows to assemble a
    # :class:`BestofEngine` instead of :class:`StrategyEngine`.
    if is_bestof:
        strategy["bestof_mode"] = bestof_score_mode
    else:
        strategy.pop("bestof_mode", None)

    # Reading-order rebuild flag (only ``seq-any`` / ``quality`` set it).
    if reorder_lines:
        output["reorder_lines"] = True
    else:
        output.pop("reorder_lines", None)

    strategy["name"] = key
    cfg.strategy = strategy
    cfg.output = output
    return cfg


def build_pipeline(
    config: Config, engine_name: Optional[str] = None
) -> object:
    """Assemble the strategy chain exactly the way CLI / API / package all do.

    ``engine_name`` forces a single engine and ignores the configured chain;
    otherwise the enabled engines run in config order under the resolved retry
    predicate. A ``strategy.name`` in the config is treated as documentation of
    the active preset — presets are applied earlier via
    :func:`apply_strategy_preset`, never re-applied here.

    Bestof presets (strategy["bestof_mode"] set) assemble a
    :class:`BestofEngine` instead of :class:`StrategyEngine`.
    """
    if engine_name:
        engines = [build_engine(engine_name, config)]
    else:
        engines = engines_from_config(config)
    strategy_cfg = config.strategy or {}

    # Bestof family — run every engine, pick the winner by a score function.
    # A forced engine_name means "don't use the strategy, just this one" — skip
    # bestof in that case.
    bestof_mode = strategy_cfg.get("bestof_mode")
    if bestof_mode and not engine_name:
        return BestofEngine(engines, score_mode=bestof_mode)

    return build_strategy(
        engines,
        retries=int(strategy_cfg.get("max_retries", 1)),
        retry_check=resolve_retry_check(strategy_cfg),
    )


__all__ = [
    "STRATEGY_PRESETS",
    "VALID_RETRY_MODES",
    "VALID_SCORE_MODES",
    "StrategyEngine",
    "apply_strategy_preset",
    "build_engine",
    "build_pipeline",
    "build_strategy",
    "describe_presets",
    "engines_from_config",
    "remote_engines",
    "resolve_retry_check",
]
