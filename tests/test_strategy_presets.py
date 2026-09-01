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
    apply_strategy_preset,
    build_pipeline,
    engines_from_config,
    remote_engines,
    resolve_retry_check,
)
from jykj_ocr.models import BoundingBox, OCRResult, TextRegion
from jykj_ocr.strategy import (
    StrategyEngine,
    combine_predicates,
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
