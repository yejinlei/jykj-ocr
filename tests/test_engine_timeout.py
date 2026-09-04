# -*- coding: utf-8 -*-
"""BaseEngine timeout behaviour.

Every engine — local or remote — is wrapped with a wall-clock timeout so a
stuck OCR worker cannot hold a FastAPI thread forever. This module locks
that behaviour in:

- ``EngineConfig.timeout`` respected (both remote and local engines).
- Timeout raises :class:`EngineError` with ``engine=<name>``.
- ``timeout <= 0`` disables the safety net (operator escape hatch).
- Missing / invalid timeout falls back to ``DEFAULT_ENGINE_TIMEOUT_SECONDS``.
"""
from __future__ import annotations

import time

import pytest

from jykj_ocr.config import EngineConfig
from jykj_ocr.engine.base import (
    BaseEngine,
    DEFAULT_ENGINE_TIMEOUT_SECONDS,
    EngineError,
    EngineNotAvailable,
    PageImage,
)


class _SlowEngine(BaseEngine):
    """Engine whose ``_recognise_impl`` sleeps for a configurable duration."""

    name = "slow"

    def __init__(self, config, sleep_s: float, result: str = "ok"):
        super().__init__(config)
        self._sleep_s = sleep_s
        self._result = result
        self.calls = 0

    def _recognise_impl(self, page):
        self.calls += 1
        time.sleep(self._sleep_s)
        return {"text": self._result}


class _BlastEngine(BaseEngine):
    """Engine that raises EngineNotAvailable — must bypass the timeout wrapper
    and re-raise the same exception type unchanged."""

    name = "unavailable"

    def __init__(self, config):
        super().__init__(config)

    def _recognise_impl(self, page):
        raise EngineNotAvailable("no API key")


def _page() -> PageImage:
    return PageImage(pil_image=None, page=0, width=1, height=1)


class TestEngineTimeout:
    def test_successful_call_under_timeout_returns_result(self):
        engine = _SlowEngine(EngineConfig(name="slow", timeout=5.0), sleep_s=0.05)
        result = engine.recognise(_page())
        assert result.text == "ok"
        assert result.engine == "slow"

    def test_timeout_raises_engine_error_with_engine_name(self):
        engine = _SlowEngine(EngineConfig(name="slow", timeout=0.05), sleep_s=0.4)
        with pytest.raises(EngineError) as excinfo:
            engine.recognise(_page())
        err = excinfo.value
        assert "timed out" in str(err)
        assert err.engine == "slow"
        # Original timeout error is chained.
        assert err.__cause__ is not None

    def test_timeout_le_zero_disables_wrapper(self):
        """An operator can turn the safety net off with timeout=0."""
        engine = _SlowEngine(EngineConfig(name="slow", timeout=0), sleep_s=0.1)
        # If the timeout wrapper were still active with a bogus timeout,
        # this would raise EngineError; instead it should run to completion.
        result = engine.recognise(_page())
        assert result.text == "ok"

    def test_default_timeout_is_120s(self):
        """When EngineConfig.timeout is None, fall back to the module default."""
        cfg = EngineConfig(name="slow", timeout=None)
        engine = _SlowEngine(cfg, sleep_s=0.05)
        assert engine._timeout_seconds() == DEFAULT_ENGINE_TIMEOUT_SECONDS

    def test_invalid_timeout_falls_back_to_default(self):
        cfg = EngineConfig(name="slow", timeout="not-a-number")
        engine = _SlowEngine(cfg, sleep_s=0.05)
        assert engine._timeout_seconds() == DEFAULT_ENGINE_TIMEOUT_SECONDS

    def test_engine_not_available_propagates_unchanged(self):
        """EngineNotAvailable must NOT be wrapped as a timeout EngineError —
        the strategy layer distinguishes 'engine not usable' from 'engine
        failed'."""
        engine = _BlastEngine(EngineConfig(name="unavailable", timeout=1.0))
        with pytest.raises(EngineNotAvailable):
            engine.recognise(_page())

    def test_local_engine_also_has_timeout(self):
        """RapidOCR is a local engine and still needs a wall-clock cap in
        case the ONNX runtime hangs. Verifies rapidocr picks up
        EngineConfig.timeout and the BaseEngine wrapper is in place."""
        from jykj_ocr.engines.rapidocr_engine import RapidOCREngine

        cfg = EngineConfig(name="rapidocr", timeout=5.0)
        engine = RapidOCREngine(cfg)
        # _timeout_seconds() must come from the config, not default to 120.
        assert engine._timeout_seconds() == 5.0

    def test_multimodal_engine_timeout_from_config(self):
        from jykj_ocr.engines.multimodal_engine import MultimodalEngine

        cfg = EngineConfig(
            name="multimodal",
            timeout=15.0,
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        engine = MultimodalEngine(cfg)
        # BaseEngine._timeout_seconds must see the same value that
        # requests.post() uses (both read EngineConfig.timeout).
        assert engine._timeout_seconds() == 15.0
        assert engine.timeout == 15.0
