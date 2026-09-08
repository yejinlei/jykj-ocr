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
    StrategyError,
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
_DEFAULT_REMOTE_ENGINES = ("multimodal",)


def remote_engines() -> tuple:
    """Remote engine names: built-in allow-list plus env override."""
    extra = os.getenv("JYKJ_OCR_REMOTE_ENGINES", "")
    names = [n.strip().lower() for n in extra.split(",") if n.strip()]
    return tuple(dict.fromkeys(list(_DEFAULT_REMOTE_ENGINES) + names))


#: Minimum engine count that makes a comparison strategy meaningful.
#:
#: ``bestof*`` runs every engine and picks the best — with one engine there is
#: nothing to compare against, so the "winner" is whatever happened to be
#: configured (it can even lose on merit to an engine that never ran).
#: ``seq*`` / ``cascade*`` need at least two to have anywhere to fall back to;
#: with one engine a retry is the same engine re-reading the same image.
#:
#: Overridable per deployment via ``JYKJ_OCR_MIN_ENGINES`` (clamped to 1..99).
def min_engines_for_strategy() -> int:
    """Required engine count, read live so tests and deployments can override it."""
    raw = os.getenv("JYKJ_OCR_MIN_ENGINES", "").strip()
    try:
        value = int(raw) if raw else 2
    except ValueError:
        value = 2
    return min(99, max(1, value))


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
#:
#: Keys are the *public* preset spellings (``seq``, not ``_seq``). Configs,
#: ``strategy_name``, and the CLI all type them bare. The ``/ocr/{preset}``
#: route accepts an optional leading ``_`` as an explicit "this is a preset,
#: not an engine" marker, and strips it before calling
#: :func:`apply_strategy_preset` — see ``server._resolve_preset_route``.
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

#: Engine-count floor per preset, before the global ``min_engines_for_strategy()``.
#:
#: ``None`` = the global minimum for seq/cascade, and "no gate" for ``local``
#: and ``vl`` (both are scoped presets whose one-engine result is by design —
#: ``local`` matches ``config.local.yaml``; ``vl`` guarantees ≥1 remote or
#: raises ``ValueError`` on its own).
MIN_ENGINES_REQUIRED = {
    "local": None,
    "vl": 1,
    "bestof": 2,
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
            "min_engines": _required_engine_count(name),
        }
    out["bestof:<mode>"] = {
        "is_bestof": True,
        "min_engines": MIN_ENGINES_REQUIRED["bestof"],
        "note": "colon syntax alias for bestof-<mode>",
    }
    return out


def build_engine(name: str, config: Config, engine_config: Optional[EngineConfig] = None) -> engine_pkg.BaseEngine:
    """Create an engine by name, falling back to config defaults when present.

    ``engine_config`` (optional) overrides the config lookup. Passing it is
    how ``engines_from_config`` builds one instance per entry when the
    config has multiple entries of the same type (e.g. two ``multimodal``
    entries pointing at different vendors — ``find_engine`` returns only
    the first, so we need to hand each entry its own ``EngineConfig``).
    """
    target = normalise_engine(name)
    found = engine_config
    if found is None:
        found = config.find_engine(target) if config else None
    return engine_pkg.create_engine(target, found or EngineConfig(name=target))


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
    """Instantiate engines, in config order unless ``names`` overrides it.

    Iterates over *entries*, not names — so a config with multiple
    ``multimodal`` instances (different vendors / models / accounts) yields
    one engine per entry, each with its own ``base_url`` / ``model`` /
    ``api_key``.
    """
    if names:
        return [build_engine(n, config) for n in names if n]
    instances = [
        build_engine(e.name, config, engine_config=e)
        for e in config.engines if e.enabled
    ]
    if not instances:
        instances.append(build_engine("multimodal", config))
    return instances


def _enabled_engine_entries(config: Config) -> List[EngineConfig]:
    """Enabled entries as :class:`EngineConfig` objects, without building them.

    Counting enabled engines must not instantiate them: a remote entry without
    a resolvable endpoint raises :class:`EngineNotAvailable` at construction
    time, which would make a purely arithmetical gate turn into a config error
    (and the gate would then never report its own "only N enabled" message).
    ``local`` / ``vl`` scope flips happen earlier, so the count here is the
    chain length that will actually run. An empty config is treated as the
    default single multimodal engine, matching :func:`engines_from_config`.
    """
    enabled = [e for e in config.engines if e.enabled]
    return enabled or [EngineConfig(name="multimodal")]


def apply_strategy_preset(config: Config, name: str) -> Config:
    """Return a copy of ``config`` reshaped by a named preset (one-shot semantics).

    Presets (see ``STRATEGY_PRESETS``):
      - ``local``    only local (rapidocr-family) engines; plain no_text retry
      - ``vl``       exactly one remote VL engine — the first *enabled* one in
                     config order (the first one overall when none is enabled),
                     so the returned model is deterministic
      - ``seq*``     first-acceptable-wins (:class:`StrategyEngine`); differ by
                     retry predicate and whether the final result is re-ordered
      - ``bestof*``  every engine runs once (:class:`BestofEngine`); the winner
                     is picked by a score function; never re-ordered

    A leading ``_`` is tolerated and dropped (``_seq`` ≡ ``seq``) — the
    ``/ocr/{preset}`` route uses it as an explicit "preset, not engine" marker.

    ``name`` is written into ``strategy["name"]`` so downstream code (text
    reorder, pipeline assembly) can see which preset produced this config.
    Unknown names raise ``ValueError`` listing the valid presets.

    Adding engines later: a newly registered engine needs no preset change —
    ``local`` keeps it (unless its name is in ``remote_engines()``), ``vl``
    excludes it, and ``seq*``/``bestof*`` use whatever ``enabled`` flag the
    config gives it. Mark a new remote-only vendor by adding its name to
    ``JYKJ_OCR_REMOTE_ENGINES``.
    """
    key = (name or "").strip()
    # Accept the ``_`` prefix too: the ``/ocr/{preset}`` route lets a caller
    # mark a value as "preset, not engine", and strips it before landing here
    # — but tolerate it directly so a hand-written preset name is portable.
    if key.startswith("_"):
        key = key[1:]
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

    if key not in _SEQ_PRESETS:
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
        if not remotes:
            raise ValueError(
                "strategy 'vl' needs a remote engine (multimodal) "
                "configured; none found"
            )
        # Keep exactly one remote: the first enabled one in config order, or
        # the first one overall when none is enabled. A multi-remote retry
        # chain makes the returned model depend on which result cleared the
        # retry predicate — the same image can come back from a different
        # model, which is not what "give me the VL answer" should mean.
        chosen = next((e for e in remotes if e.enabled), remotes[0])
        for engine in cfg.engines:
            engine.enabled = engine is chosen

    # Write the retry mode (bestof family has no retry predicate).
    if retry_mode is not None:
        strategy["retry_mode"] = retry_mode
    else:
        strategy.pop("retry_mode", None)

    # Write max_retries when the preset dictates a specific value (e.g.
    # ``cascade*`` = 0). Presets that leave it alone use ``None`` and keep
    # whatever the config already had — this is how ``seq*`` retains its
    # configured retries. Only bestof drops it: that family has no retry
    # predicate, so a stale value would just mislead the trace.
    if preset_max_retries is not None:
        strategy["max_retries"] = preset_max_retries
    elif is_bestof:
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

    # ``strategy`` and ``output`` are shallow copies mutated above — write them
    # back, along with the preset name so downstream code can see which preset
    # produced this config.
    strategy["name"] = key
    cfg.strategy = strategy
    cfg.output = output

    # Count gate. Runs last, after the ``local``/``vl`` scope flip, so it counts
    # the engines that will actually run — ``local`` leaves the remaining local
    # ones enabled, ``vl`` keeps exactly one remote. Applied here rather than
    # only in build_pipeline so the short-chain failure is a 422 at the request
    # boundary; every entry path (CLI --strategy-name, body strategy_name,
    # /ocr/{preset}, the Python API) funnels through this function.
    _check_strategy_engine_count(cfg, key)
    return cfg


def _required_engine_count(preset: str) -> Optional[int]:
    """Engine-count floor for a preset, or ``None`` to skip the check.

    ``local`` is exempt outright: ``config.local.yaml`` is one local engine by
    design, so the gate would reject a legitimate deployment. ``vl`` reports a
    floor of one but is not gated — the requirement is met anyway, or
    :func:`apply_strategy_preset` raises ``ValueError`` when no remote exists.
    ``seq*`` / ``cascade*`` fall back to the global minimum; ``bestof*`` never
    goes below two — with one engine there is nothing to compare against.
    """
    if preset in ("local", "vl"):
        return MIN_ENGINES_REQUIRED[preset]
    return max(MIN_ENGINES_REQUIRED.get(preset, min_engines_for_strategy()),
               min_engines_for_strategy())


def _check_strategy_engine_count(config: Config, preset: str) -> None:
    """Refuse a comparison strategy whose engine list is too short.

    This is the failure mode worth failing fast on: a single enabled engine
    under ``bestof`` still returns an answer, so the caller sees a confident
    ``decision`` and never learns the other candidate errored out. ``seq*``
    with one engine is the same story minus the score — a retry is the same
    engine re-reading the same image. Failing here surfaces it as a 422 with
    the offending count instead of a plausible-looking result.

    Called from :func:`apply_strategy_preset` (so every entry path — CLI
    ``--strategy-name``, ``strategy_name`` on the body, ``/ocr/{preset}``, the
    Python API — is covered) and again from :func:`build_pipeline` for the
    raw ``score_mode`` knob, which never passes through the preset layer.
    """
    required = _required_engine_count(preset)
    if required is None:
        return
    entries = _enabled_engine_entries(config)
    if len(entries) >= required:
        return
    raise StrategyError(
        f"strategy {preset!r} needs at least {required} engine(s), "
        f"but only {len(entries)} is/are enabled: "
        f"{', '.join(e.name for e in entries) or '(none)'}"
    )


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

    Raises :class:`StrategyError` (HTTP 422) when the effective strategy needs
    more engines than are enabled — see :func:`_check_strategy_engine_count`.
    """
    if engine_name:
        return build_strategy(
            [build_engine(engine_name, config)],
            retries=int((config.strategy or {}).get("max_retries", 1)),
            retry_check=resolve_retry_check(config.strategy or {}),
        )
    strategy_cfg = config.strategy or {}
    preset = str(strategy_cfg.get("name") or "").strip().lstrip("_").lower()
    if not preset and strategy_cfg.get("bestof_mode"):
        # ``score_mode`` knob without a preset name: bestof semantics.
        preset = "bestof"
    if preset:
        _check_strategy_engine_count(config, preset)
    engines = engines_from_config(config)
    return _assemble(engines, strategy_cfg)


def _assemble(engines: Sequence[Any], strategy_cfg: Dict[str, Any]) -> object:
    """Pick the chain class: ``bestof*`` → :class:`BestofEngine`, else Strategy."""
    bestof_mode = strategy_cfg.get("bestof_mode")
    if bestof_mode:
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
    "min_engines_for_strategy",
    "remote_engines",
    "resolve_retry_check",
]
