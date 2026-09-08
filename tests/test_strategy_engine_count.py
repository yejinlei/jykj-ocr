# -*- coding: utf-8 -*-
"""Engine-count gate for comparison strategies.

``bestof*`` needs a candidate to compare against and ``seq*`` / ``cascade*``
need somewhere to fall back to. With a single enabled engine both degrade into
"the one engine runs and its result looks like a normal success" — the trace
then carries a ``decision`` that hides the fact nothing was ever compared.

The gate lives in ``apply_strategy_preset`` (so CLI ``--strategy-name``, the
HTTP body, ``/ocr/{preset}`` and the Python API all hit it) and again in
``build_pipeline`` (for the bare ``score_mode`` knob, which never passes
through the preset layer).

Two engine families share the doubles from
:mod:`test_decision_trace_and_openapi`: they already monkeypatch the real
``build_pipeline`` / ``apply_strategy_preset`` and only fake ``build_engine``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jykj_ocr.config import from_mapping
from jykj_ocr.engine import registry as reg
from jykj_ocr.strategy import StrategyEngine, StrategyError

from test_decision_trace_and_openapi import _client, _post_png


def _single_engine_cfg():
    """A legitimate one-engine config: only rapidocr enabled.

    This is exactly ``config.local.yaml`` — a real deployment shape, not a
    misconfiguration, which is why ``local`` is exempt from the gate.
    """
    return from_mapping({
        "engines": [
            {"name": "rapidocr", "enabled": True},
            {"name": "multimodal", "enabled": False,
             "model": "PaddleOCR-VL-1.5",
             "base_url": "https://entry.example.com/v1",
             "api_key": "sk-entry"},
        ],
        "strategy": {"max_retries": 0, "min_confidence": 0.6},
        "output": {},
        "pdf": {},
    })


class TestEngineCountGateRegistry:
    """The gate as unit-level assertions, without HTTP."""

    def test_bestof_requires_two(self):
        with pytest.raises(StrategyError) as exc:
            reg.apply_strategy_preset(_single_engine_cfg(), "bestof")
        assert "at least 2 engine(s)" in str(exc.value)
        assert "rapidocr" in str(exc.value)

    def test_seq_requires_two(self):
        with pytest.raises(StrategyError):
            reg.apply_strategy_preset(_single_engine_cfg(), "seq")

    def test_cascade_requires_two(self):
        for name in ("cascade", "cascade-low_conf", "cascade-line_overlap"):
            with pytest.raises(StrategyError):
                reg.apply_strategy_preset(_single_engine_cfg(), name)

    def test_local_is_exempt(self):
        """One local engine is a valid config, not a misconfiguration."""
        cfg = reg.apply_strategy_preset(_single_engine_cfg(), "local")
        assert [(e.name, e.enabled) for e in cfg.engines] == [
            ("rapidocr", True), ("multimodal", False),
        ]
        assert len(reg.engines_from_config(cfg)) == 1

    def test_vl_is_exempt_but_still_needs_one(self):
        """``vl`` only narrows scope; it still needs something to keep."""
        cfg = reg.apply_strategy_preset(_single_engine_cfg(), "vl")
        assert [(e.name, e.enabled) for e in cfg.engines] == [
            ("rapidocr", False), ("multimodal", True),
        ]
        assert len(reg.engines_from_config(cfg)) == 1

        remoteless = from_mapping({"engines": [{"name": "rapidocr", "enabled": True}]})
        with pytest.raises(ValueError):
            reg.apply_strategy_preset(remoteless, "vl")

    @pytest.mark.parametrize("preset", [
        "seq", "seq-any", "seq-low_conf", "seq-line_overlap",
        "cascade", "cascade-low_conf", "cascade-line_overlap",
        "fallback", "quality",
    ])
    def test_all_seq_family_gated(self, preset):
        with pytest.raises(StrategyError):
            reg.apply_strategy_preset(_single_engine_cfg(), preset)

    def test_passes_when_two_engines_are_enabled(self):
        two = from_mapping({
            "engines": [
                {"name": "rapidocr", "enabled": True},
                {"name": "multimodal", "enabled": True,
                 "model": "PaddleOCR-VL-1.5",
                 "base_url": "https://entry.example.com/v1",
                 "api_key": "sk-entry"},
            ]
        })
        for preset in ("seq", "cascade", "bestof", "bestof-smart", "bestof:fastest"):
            cfg = reg.apply_strategy_preset(two, preset)
            assert len(reg.engines_from_config(cfg)) == 2

    def test_gated_by_enabled_flag_not_entry_count(self):
        """Two entries with one disabled is still one engine."""
        cfg = _single_engine_cfg()
        assert len(cfg.engines) == 2
        assert len(reg.engines_from_config(cfg)) == 1

    def test_force_engine_skips_the_gate(self):
        """`--engine rapidocr` is a plain single-engine run, no strategy."""
        pipeline = reg.build_pipeline(_single_engine_cfg(), engine_name="rapidocr")
        assert isinstance(pipeline, StrategyEngine)

    def test_plain_chain_has_no_gate(self):
        """No preset and no ``bestof_mode`` -> an ordinary chain, any size."""
        cfg = _single_engine_cfg()
        assert "name" not in cfg.strategy and "bestof_mode" not in cfg.strategy
        pipeline = reg.build_pipeline(cfg)
        assert isinstance(pipeline, StrategyEngine)

    def test_score_mode_knob_alone_is_gated(self):
        """`score_mode` builds a BestofEngine even with no preset name."""
        cfg = _single_engine_cfg()
        cfg.strategy["bestof_mode"] = "fastest"
        with pytest.raises(StrategyError):
            reg.build_pipeline(cfg)

    def test_min_engines_env_override(self, monkeypatch):
        cfg = _single_engine_cfg()
        monkeypatch.setenv("JYKJ_OCR_MIN_ENGINES", "1")
        assert reg.min_engines_for_strategy() == 1
        assert reg.apply_strategy_preset(cfg, "seq").strategy["name"] == "seq"

        # Two never goes below two, whatever the knob says.
        with pytest.raises(StrategyError):
            reg.apply_strategy_preset(cfg, "bestof")

    def test_min_engines_env_clamps_bad_values(self, monkeypatch):
        cases = [("abc", 2), ("", 2), ("0", 1), ("-5", 1), ("999", 99)]
        for raw, expected in cases:
            monkeypatch.setenv("JYKJ_OCR_MIN_ENGINES", raw)
            assert reg.min_engines_for_strategy() == expected, raw


class TestEngineCountGateHttp:
    """Same gate through the real FastAPI app: 422 with the offending count."""

    def _client_one(self, monkeypatch) -> TestClient:
        """App wired to a one-engine config (rapidocr on, multimodal off)."""
        return _client(monkeypatch, {
            "rapidocr": {"text": "local answer"},
            "multimodal": {"text": "remote answer"},
        }, cfg=_single_engine_cfg())

    def test_preset_route_returns_422(self, monkeypatch):
        c = self._client_one(monkeypatch)
        r = _post_png(c, "/ocr/_bestof", format="json")
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert "at least 2 engine(s)" in detail
        assert "rapidocr" in detail

    def test_every_comparison_route_returns_422(self, monkeypatch):
        c = self._client_one(monkeypatch)
        for route in ("/ocr/_bestof", "/ocr/_bestof-smart", "/ocr/_bestof-fluency",
                      "/ocr/_bestof-fastest", "/ocr/_bestof-confidence",
                      "/ocr/_bestof-longest", "/ocr/_bestof:fastest",
                      "/ocr/_seq", "/ocr/_seq-any", "/ocr/_cascade"):
            r = _post_png(c, route, format="json")
            assert r.status_code == 422, f"{route}: {r.status_code} {r.text}"

    def test_body_strategy_name_returns_422(self, monkeypatch):
        c = self._client_one(monkeypatch)
        r = c.post("/ocr/text", json={
            "image_url": "https://example.com/scan.png",
            "strategy_name": "bestof",
        })
        assert r.status_code == 422, r.text
        assert "at least 2 engine(s)" in r.json()["detail"]

    def test_local_preset_still_works(self, monkeypatch):
        """The one-engine config is still usable — for the local preset."""
        c = self._client_one(monkeypatch)
        r = _post_png(c, "/ocr/_local", format="json")
        assert r.status_code == 200, r.text
        assert r.json()["engine"] == "rapidocr"
        assert r.json()["text"] == "local answer"

    def test_vl_preset_still_works(self, monkeypatch):
        """``vl`` keeps one remote even when the config had it disabled."""
        c = self._client_one(monkeypatch)
        r = _post_png(c, "/ocr/_vl", format="json")
        assert r.status_code == 200, r.text
        assert r.json()["engine"] == "multimodal"
        assert r.json()["text"] == "remote answer"

    def test_forced_engine_still_works(self, monkeypatch):
        """`engine=rapidocr` is not a strategy, so no minimum applies."""
        c = self._client_one(monkeypatch)
        r = _post_png(c, "/ocr", engine="rapidocr", format="json")
        assert r.status_code == 200, r.text
        assert r.json()["engine"] == "rapidocr"

    def test_plain_chain_still_works(self, monkeypatch):
        """No preset at all: the configured chain runs, one engine or not."""
        c = self._client_one(monkeypatch)
        r = _post_png(c, "/ocr", format="json")
        assert r.status_code == 200, r.text
        assert r.json()["engine"] == "rapidocr"

    def test_presets_endpoint_exposes_the_requirement(self, monkeypatch):
        c = self._client_one(monkeypatch)
        r = c.get("/presets")
        assert r.status_code == 200, r.text
        presets = r.json()["presets"]
        assert presets["bestof"]["min_engines"] == 2
        assert presets["bestof-fluency"]["min_engines"] == 2
        assert presets["bestof:<mode>"]["min_engines"] == 2
        assert presets["seq"]["min_engines"] == 2
        assert presets["cascade"]["min_engines"] == 2
        assert presets["local"]["min_engines"] is None
        assert presets["vl"]["min_engines"] == 1
