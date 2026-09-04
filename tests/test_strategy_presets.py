# -*- coding: utf-8 -*-
"""Tests for retry predicates, named strategy presets and pipeline assembly.

All offline — fake engines only, no network, no rapidocr import.
"""

from __future__ import annotations

import copy

import pytest

from jykj_ocr.config import Config, from_mapping
from jykj_ocr.engine.registry import (
    STRATEGY_PRESETS,
    _SEQ_PRESETS,
    apply_strategy_preset,
    build_pipeline,
    engines_from_config,
    remote_engines,
    resolve_retry_check,
)
from jykj_ocr.models import BoundingBox, OCRResult, TextRegion
from jykj_ocr.strategy import (
    BestofEngine,
    StrategyEngine,
    combine_predicates,
    resolve_bestof_score,
    should_retry_line_overlap,
)


def _result(text: str = "hi", confidence: float = 0.95, boxes=None) -> OCRResult:
    if boxes:
        regions = [
            TextRegion(
                text=text,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=confidence,
            )
            for (x1, y1, x2, y2) in boxes
        ]
    else:
        regions = [TextRegion(text=text, confidence=confidence)]
    return OCRResult(text=text, engine="fake", regions=regions)


class TestRetryPredicates:
    def test_line_overlap_rejects_no_text(self):
        assert should_retry_line_overlap(None, OCRResult(text="")) is True

    def test_line_overlap_accepts_clean_layout(self):
        result = _result("ok", boxes=[(0, 0, 100, 12), (0, 20, 90, 32)])
        assert should_retry_line_overlap(None, result) is False

    def test_line_overlap_flags_merged_row(self):
        result = _result("bad", boxes=[(0, 0, 40, 10), (0, 0, 300, 10)])
        assert should_retry_line_overlap(None, result) is True

    def test_combine_predicates_or(self):
        low_conf = _result("t", confidence=0.3)
        clean_high = _result("t", confidence=0.95, boxes=[(0, 0, 60, 12)])
        combined = combine_predicates(
            resolve_retry_check({"retry_mode": "low_confidence", "min_confidence": 0.7}),
            should_retry_line_overlap,
        )
        assert combined(None, low_conf) is True
        assert combined(None, clean_high) is False

    def test_combine_predicates_drops_none(self):
        solo = should_retry_line_overlap
        assert combine_predicates(None, solo) is solo
        assert combine_predicates() is None


class TestResolveRetryCheck:
    def test_no_text_mode(self):
        check = resolve_retry_check({"retry_mode": "no_text"})
        assert callable(check)
        assert check(None, OCRResult(text="")) is True
        assert check(None, _result("t")) is False

    def test_none_and_first_success_are_noop(self):
        assert resolve_retry_check({"retry_mode": "none"}) is None
        assert resolve_retry_check({"retry_mode": "first_success"}) is None

    def test_low_confidence_threshold(self):
        check = resolve_retry_check(
            {"retry_mode": "low_confidence", "min_confidence": 0.7}
        )
        assert check(None, _result("t", confidence=0.3)) is True
        assert check(None, _result("t", confidence=0.95)) is False

    def test_line_overlap_mode(self):
        check = resolve_retry_check({"retry_mode": "line_overlap"})
        assert check(None, _result("t", boxes=[(0, 0, 40, 10), (0, 0, 300, 10)])) is True
        assert check(None, _result("t", boxes=[(0, 0, 60, 12)])) is False

    def test_any_mode_composites(self):
        check = resolve_retry_check({"retry_mode": "any", "min_confidence": 0.7})
        assert check(None, _result("t", confidence=0.2)) is True  # low conf
        merged = _result("t", confidence=0.99, boxes=[(0, 0, 40, 10), (0, 0, 300, 10)])
        assert check(None, merged) is True  # garbled layout
        clean = _result("t", confidence=0.95, boxes=[(0, 0, 60, 12)])
        assert check(None, clean) is False

    def test_empty_strategy_is_default(self):
        assert resolve_retry_check({}) is None


def _config(**strategy) -> Config:
    return from_mapping(
        {
            "engines": [
                {"name": "rapidocr", "enabled": True},
                {"name": "siliconflow", "enabled": True},
                {"name": "multimodal", "enabled": False},
            ],
            "strategy": strategy or {"retry_mode": "no_text"},
        }
    )


class TestApplyStrategyPreset:
    def test_all_presets_round_trip(self):
        for name in STRATEGY_PRESETS:
            cfg = apply_strategy_preset(_config(), name)
            assert cfg.strategy["name"] == name

    def test_local_disables_remotes_only(self):
        cfg = apply_strategy_preset(_config(), "local")
        flags = {e.name: e.enabled for e in cfg.engines}
        assert flags == {"rapidocr": True, "siliconflow": False, "multimodal": False}
        assert cfg.strategy["retry_mode"] == "no_text"
        assert "reorder_lines" not in cfg.output

    def test_vl_disables_locals_keeps_enabled_remote(self):
        cfg = apply_strategy_preset(_config(), "vl")
        flags = {e.name: e.enabled for e in cfg.engines}
        assert flags["rapidocr"] is False
        assert flags["siliconflow"] is True

    def test_vl_enables_disabled_remotes_when_none_enabled(self):
        base = _config()
        for e in base.engines:
            if e.name != "rapidocr":
                e.enabled = False
        cfg = apply_strategy_preset(base, "vl")
        assert any(e.enabled for e in cfg.engines if e.name != "rapidocr")

    def test_vl_without_any_remote_raises(self):
        base = from_mapping({"engines": [{"name": "rapidocr", "enabled": True}]})
        with pytest.raises(ValueError, match="remote"):
            apply_strategy_preset(base, "vl")

    def test_fallback_does_not_touch_enabled_flags(self):
        cfg = apply_strategy_preset(_config(), "fallback")
        flags = {e.name: e.enabled for e in cfg.engines}
        assert flags == {"rapidocr": True, "siliconflow": True, "multimodal": False}

    def test_quality_sets_any_and_reorder(self):
        cfg = apply_strategy_preset(_config(), "quality")
        assert cfg.strategy["retry_mode"] == "any"
        assert cfg.output["reorder_lines"] is True

    def test_unknown_name_raises_with_valid_list(self):
        with pytest.raises(ValueError, match="hybrid"):
            apply_strategy_preset(_config(), "hybrid")

    def test_input_config_never_mutated(self):
        base = _config(retry_mode="no_text")
        before = copy.deepcopy(base)
        apply_strategy_preset(base, "quality")
        assert base.strategy == before.strategy
        assert base.output == before.output
        assert [(e.name, e.enabled) for e in base.engines] == [
            (e.name, e.enabled) for e in before.engines
        ]

    def test_alias_names_are_classified_as_remote(self):
        base = from_mapping(
            {
                "engines": [
                    {"name": "rapidocr", "enabled": True},
                    {"name": "sf", "enabled": True},  # alias of siliconflow
                ]
            }
        )
        cfg = apply_strategy_preset(base, "local")
        # "sf" is normalised to "siliconflow" by from_mapping, then classified remote.
        assert [(e.name, e.enabled) for e in cfg.engines] == [
            ("rapidocr", True),
            ("siliconflow", False),
        ]


class TestRemoteEnginesExtensibility:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("JYKJ_OCR_REMOTE_ENGINES", raising=False)
        assert remote_engines() == ("siliconflow", "multimodal")

    def test_env_override_adds_new_vendor(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_REMOTE_ENGINES", "PaddleCloud, other-vl ")
        names = remote_engines()
        assert "paddlecloud" in names and "other-vl" in names
        assert "siliconflow" in names  # built-ins stay

    def test_env_override_moves_engine_to_vl_side(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_REMOTE_ENGINES", "acme-vl")
        base = from_mapping(
            {
                "engines": [
                    {"name": "rapidocr", "enabled": True},
                    {"name": "acme-vl", "enabled": True},
                ]
            }
        )
        local = apply_strategy_preset(base, "local")
        assert {e.name: e.enabled for e in local.engines} == {
            "rapidocr": True,
            "acme-vl": False,
        }
        vl = apply_strategy_preset(base, "vl")
        assert {e.name: e.enabled for e in vl.engines} == {
            "rapidocr": False,
            "acme-vl": True,
        }


class TestEnginesFromConfigAndPipeline:
    def test_respects_enabled_flag(self):
        engines = engines_from_config(_config())
        assert len(engines) == 2  # multimodal disabled

    def test_explicit_names_ignore_enabled(self, monkeypatch):
        # multimodal is disabled in _config(); an explicit name must still build it.
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        engines = engines_from_config(_config(), ["multimodal"])
        assert len(engines) == 1

    def test_build_pipeline_forced_engine(self):
        pipeline = build_pipeline(_config(), engine_name="rapidocr")
        assert isinstance(pipeline, StrategyEngine)
        assert pipeline.engines() == ["rapidocr"]

    def test_build_pipeline_honours_quality_preset(self):
        cfg = apply_strategy_preset(_config(), "quality")
        pipeline = build_pipeline(cfg)
        assert pipeline.retry_check is not None
        garbled = _result("t", confidence=0.99, boxes=[(0, 0, 40, 10), (0, 0, 300, 10)])
        assert pipeline.retry_check(None, garbled) is True
        low = _result("t", confidence=0.2)
        assert pipeline.retry_check(None, low) is True
        clean = _result("t", confidence=0.95, boxes=[(0, 0, 60, 12)])
        assert pipeline.retry_check(None, clean) is False

    def test_build_pipeline_does_not_reapply_preset(self):
        cfg = _config(retry_mode="no_text")
        cfg.strategy["name"] = "quality"  # stale marker must NOT flip the chain
        pipeline = build_pipeline(cfg)
        empty = OCRResult(text="")
        assert pipeline.retry_check(None, empty) is True  # still no_text
        good = _result("t", confidence=0.3)
        assert pipeline.retry_check(None, good) is False  # not low_confidence


# ---------------------------------------------------------------------------
# seq* presets (seq / seq-any / seq-low_conf / seq-line_overlap)
# ---------------------------------------------------------------------------

class TestSeqPresets:
    def test_seq_retry_mode_map(self):
        # Every seq* preset has the retry_mode it advertises.
        assert _SEQ_PRESETS["seq"][0] == "no_text"
        assert _SEQ_PRESETS["seq-any"][0] == "any"
        assert _SEQ_PRESETS["seq-low_conf"][0] == "low_confidence"
        assert _SEQ_PRESETS["seq-line_overlap"][0] == "line_overlap"

    def test_seq_aliases_do_not_mutate_enabled(self):
        cfg = apply_strategy_preset(_config(), "seq")
        flags = {e.name: e.enabled for e in cfg.engines}
        assert flags == {"rapidocr": True, "siliconflow": True, "multimodal": False}
        assert cfg.strategy["retry_mode"] == "no_text"
        assert "reorder_lines" not in cfg.output

    def test_seq_any_matches_quality(self):
        seq = apply_strategy_preset(_config(), "seq-any")
        quality = apply_strategy_preset(_config(), "quality")
        assert seq.strategy["retry_mode"] == quality.strategy["retry_mode"] == "any"
        assert seq.output["reorder_lines"] is True
        assert quality.output["reorder_lines"] is True

    def test_seq_low_conf_sets_mode(self):
        cfg = apply_strategy_preset(_config(), "seq-low_conf")
        assert cfg.strategy["retry_mode"] == "low_confidence"
        assert cfg.strategy["name"] == "seq-low_conf"

    def test_seq_line_overlap_sets_mode(self):
        cfg = apply_strategy_preset(_config(), "seq-line_overlap")
        assert cfg.strategy["retry_mode"] == "line_overlap"
        assert cfg.strategy["name"] == "seq-line_overlap"

    def test_seq_any_build_pipeline_any_predicate(self):
        cfg = apply_strategy_preset(_config(), "seq-any")
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, StrategyEngine)
        garbled = _result("t", confidence=0.99, boxes=[(0, 0, 40, 10), (0, 0, 300, 10)])
        assert pipeline.retry_check(None, garbled) is True
        low = _result("t", confidence=0.2)
        assert pipeline.retry_check(None, low) is True


# ---------------------------------------------------------------------------
# bestof* presets
# ---------------------------------------------------------------------------

class TestBestofPresets:
    def test_bestof_score_mode_map(self):
        assert _SEQ_PRESETS["bestof"][3] == "smart"
        assert _SEQ_PRESETS["bestof-smart"][3] == "smart"
        assert _SEQ_PRESETS["bestof-fastest"][3] == "fastest"
        assert _SEQ_PRESETS["bestof-confidence"][3] == "highest_confidence"
        assert _SEQ_PRESETS["bestof-longest"][3] == "longest"

    def test_bestof_sets_bestof_mode(self):
        cfg = apply_strategy_preset(_config(), "bestof")
        assert cfg.strategy["bestof_mode"] == "smart"
        assert "retry_mode" not in cfg.strategy
        assert "reorder_lines" not in cfg.output

    def test_bestof_mode_aliases(self):
        cfg_smart = apply_strategy_preset(_config(), "bestof-smart")
        cfg_fastest = apply_strategy_preset(_config(), "bestof-fastest")
        cfg_conf = apply_strategy_preset(_config(), "bestof-confidence")
        cfg_longest = apply_strategy_preset(_config(), "bestof-longest")
        assert cfg_smart.strategy["bestof_mode"] == "smart"
        assert cfg_fastest.strategy["bestof_mode"] == "fastest"
        assert cfg_conf.strategy["bestof_mode"] == "highest_confidence"
        assert cfg_longest.strategy["bestof_mode"] == "longest"

    def test_bestof_colon_syntax(self):
        cfg = apply_strategy_preset(_config(), "bestof:fastest")
        assert cfg.strategy["bestof_mode"] == "fastest"
        assert cfg.strategy["name"] == "bestof"

    def test_bestof_colon_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            apply_strategy_preset(_config(), "bestof:banana")

    def test_bestof_colon_empty_raises(self):
        with pytest.raises(ValueError):
            apply_strategy_preset(_config(), "bestof:")


# ---------------------------------------------------------------------------
# BestofEngine behaviour
# ---------------------------------------------------------------------------

class _FakeEngine:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self._result = result
        self._error = error
        self._config = None

    @property
    def config(self):
        return self._config

    def recognise(self, image):
        if self._error is not None:
            raise self._error
        result = self._result
        result.engine = self.name
        return result


# ---------------------------------------------------------------------------
# StrategyEngine behaviour
# ---------------------------------------------------------------------------

class TestStrategyEngine:
    def test_accepted_result_elapsed_ms_reflects_wall_clock(self):
        """StrategyEngine must stamp the winner with wall-clock elapsed_ms.

        Before this fix, seq-family presets (seq / seq-any / seq-low_conf /
        seq-line_overlap / local / vl / quality) all returned
        ``elapsed_ms=0`` in HTTP responses. ``server._pipeline`` only wrapped
        a *forced single engine* in ``TimedOCR`` — the ``build_pipeline``
        path returned a bare ``StrategyEngine`` that never touched
        ``elapsed_ms``. ``BestofEngine`` got the same fix first.
        """
        import time

        class SleepyEngine:
            def __init__(self, name, text, sleep_s):
                self.name = name
                self._text = text
                self._sleep_s = sleep_s
                self._config = None

            @property
            def config(self):
                return self._config

            def recognise(self, image):
                time.sleep(self._sleep_s)
                result = _result(self._text, confidence=0.9)
                result.engine = self.name
                return result

        engines = [SleepyEngine("sleepy", "ok", 0.06)]
        for retry_check in (
            None,
            combine_predicates(should_retry_line_overlap),
            should_retry_line_overlap,
        ):
            pipeline = StrategyEngine(engines, retry_check=retry_check)
            winner = pipeline.recognise(None)
            assert winner.elapsed_ms >= 50, (
                f"retry_check={retry_check}: elapsed_ms={winner.elapsed_ms}"
            )

    def test_rejected_attempts_include_retry_time(self):
        """When the first engine is rejected and retries, elapsed_ms must
        cover the total wall-clock of all attempts, not just the winner."""
        import time

        class SlowRejectThenGood:
            """First call returns an empty result (retryable), second returns
            a clean one after a longer sleep — retries must show up in
            elapsed_ms."""
            name = "slow"
            config = None

            def __init__(self):
                self.calls = 0

            def recognise(self, image):
                self.calls += 1
                time.sleep(0.04)
                if self.calls == 1:
                    # Empty result — `ok=False` and no regions — trips
                    # should_retry_line_overlap so a retry actually fires.
                    return OCRResult(text="", engine=self.name, regions=[])
                return _result("finally ok", confidence=0.99)

        pipeline = StrategyEngine(
            [SlowRejectThenGood()],
            retries=1,
            retry_check=should_retry_line_overlap,
        )
        winner = pipeline.recognise(None)
        assert winner.text == "finally ok"
        # Two 40ms sleeps = 80ms; allow clock jitter down to 60ms.
        assert winner.elapsed_ms >= 60, (
            f"expected ≥ 60ms for two retries, got {winner.elapsed_ms}"
        )


class TestBestofEngine:
    def _r(self, text, confidence=0.9, elapsed_ms=100):
        result = _result(text, confidence=confidence)
        result.elapsed_ms = elapsed_ms
        return result

    def test_all_fail_raises(self):
        engines = [
            _FakeEngine("a", error=RuntimeError("boom a")),
            _FakeEngine("b", error=RuntimeError("boom b")),
        ]
        bestof = BestofEngine(engines, score_mode="smart")
        with pytest.raises(Exception):
            bestof.recognise(None)

    def test_picks_highest_confidence(self):
        engines = [
            _FakeEngine("low", result=self._r("hi", confidence=0.4, elapsed_ms=50)),
            _FakeEngine("high", result=self._r("hi", confidence=0.99, elapsed_ms=200)),
        ]
        bestof = BestofEngine(engines, score_mode="highest_confidence")
        winner = bestof.recognise(None)
        assert winner.engine == "high"

    def test_picks_fastest(self):
        engines = [
            _FakeEngine("slow", result=self._r("hi", confidence=0.99, elapsed_ms=500)),
            _FakeEngine("fast", result=self._r("hi", confidence=0.6, elapsed_ms=20)),
        ]
        bestof = BestofEngine(engines, score_mode="fastest")
        winner = bestof.recognise(None)
        assert winner.engine == "fast"

    def test_picks_longest(self):
        engines = [
            _FakeEngine("short", result=self._r("hi", confidence=0.99)),
            _FakeEngine("long", result=self._r("hello world", confidence=0.5)),
        ]
        bestof = BestofEngine(engines, score_mode="longest")
        winner = bestof.recognise(None)
        assert winner.engine == "long"

    def test_smart_penalises_garbled(self):
        # A high-confidence garbled result should lose to a clean one.
        clean = _result("ok", confidence=0.85, boxes=[(0, 0, 60, 12)])
        clean.elapsed_ms = 100
        garbled = _result("bad", confidence=0.99, boxes=[(0, 0, 40, 10), (0, 0, 300, 10)])
        garbled.elapsed_ms = 50
        engines = [
            _FakeEngine("garbled", result=garbled),
            _FakeEngine("clean", result=clean),
        ]
        bestof = BestofEngine(engines, score_mode="smart")
        winner = bestof.recognise(None)
        assert winner.engine == "clean"

    def test_summary_reports_mode(self):
        engines = [_FakeEngine("x", result=self._r("hi"))]
        bestof = BestofEngine(engines, score_mode="fastest")
        summary = bestof.summary()
        assert summary["engines"] == ["x"]
        assert "fastest" in summary["mode"]

    def test_winner_elapsed_ms_reflects_actual_wall_clock(self):
        """BestofEngine must stamp the winner with wall-clock elapsed_ms.

        Engines themselves do not set ``elapsed_ms`` — the strategy layer owns
        that accounting. Before this test, bestof responses reported
        ``elapsed_ms=0`` because BestofEngine (unlike StrategyEngine) never
        set it on the winner.
        """
        import time

        class SleepyEngine:
            def __init__(self, name, text, sleep_s, confidence=0.9):
                self.name = name
                self._text = text
                self._sleep_s = sleep_s
                self._confidence = confidence

            def recognise(self, image):
                time.sleep(self._sleep_s)
                result = _result(self._text, confidence=self._confidence)
                result.engine = self.name
                return result

        engines = [
            SleepyEngine("fast", "fast", 0.05),
            SleepyEngine("slow", "longer slow text", 0.10),
        ]
        for mode in ("smart", "fastest", "highest_confidence", "longest"):
            bestof = BestofEngine(engines, score_mode=mode)
            winner = bestof.recognise(None)
            # Both engines sleep ≥ 50ms, so total must exceed 50ms.
            assert winner.elapsed_ms >= 50, (
                f"mode={mode}: winner={winner.engine} elapsed_ms={winner.elapsed_ms}"
            )


class TestBuildPipelineBestof:
    def test_bestof_preset_builds_BestofEngine(self):
        cfg = apply_strategy_preset(_config(), "bestof-smart")
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, BestofEngine)
        assert not isinstance(pipeline, StrategyEngine)
        assert "smart" in pipeline.summary()["mode"]

    def test_fallback_still_builds_StrategyEngine(self):
        cfg = apply_strategy_preset(_config(), "fallback")
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, StrategyEngine)
        assert not isinstance(pipeline, BestofEngine)

    def test_quality_still_builds_StrategyEngine(self):
        cfg = apply_strategy_preset(_config(), "quality")
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, StrategyEngine)
        assert "any" == cfg.strategy["retry_mode"]
        assert cfg.output["reorder_lines"] is True

    def test_forced_engine_always_StrategyEngine(self):
        cfg = apply_strategy_preset(_config(), "bestof-smart")
        pipeline = build_pipeline(cfg, engine_name="rapidocr")
        assert isinstance(pipeline, StrategyEngine)
        assert pipeline.engines() == ["rapidocr"]
