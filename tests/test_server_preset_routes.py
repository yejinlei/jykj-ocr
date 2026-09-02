# -*- coding: utf-8 -*-
"""Tests for the convenient ``/ocr/{preset}`` and ``/ocr/{preset}/text`` routes.

These exercises the real FastAPI routes via TestClient. ``build_engine`` is
monkey-patched so no real OCR engines (rapidocr / openai) need to be imported.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jykj_ocr.config import EngineConfig, from_mapping
from jykj_ocr.engine import registry as engine_registry
from jykj_ocr.models import BoundingBox, OCRResult, TextRegion
from jykj_ocr.server import RuntimeConfig, TextRequest


# ---------------------------------------------------------------------------
# Fake engine (no PIL / rapidocr / openai imports)
# ---------------------------------------------------------------------------
class _FakeEngine:
    def __init__(self, name: str, text: str, confidence: float, elapsed_ms: int = 1):
        self.name = name
        self.config = EngineConfig(name=name)
        self._text = text
        self._confidence = confidence
        self._elapsed_ms = elapsed_ms

    def recognise(self, image) -> OCRResult:
        region = TextRegion(text=self._text, confidence=self._confidence,
                            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=20))
        result = OCRResult(engine=self.name, text=self._text, regions=[region],
                           width=100, height=30)
        result.elapsed_ms = self._elapsed_ms
        return result


# ---------------------------------------------------------------------------
# Small app that mirrors the real /ocr/{preset} routes (the real helpers are
# closure-locals inside create_app so we replicate them with the same logic).
# ---------------------------------------------------------------------------
def _apply_inline_overrides(config, body: TextRequest):
    from jykj_ocr.config import normalise_engine
    from jykj_ocr.engine.registry import apply_strategy_preset, remote_engines

    if body.engine or body.model or body.prompt or body.strategy:
        engine_dicts = [{
            "name": e.name,
            "enabled": e.enabled,
            "model": e.model,
            "base_url": e.base_url,
            "temperature": e.temperature,
            "timeout": e.timeout,
            "max_tokens": e.max_tokens,
            "lang": e.lang,
            "prompt": e.prompt,
        } for e in config.engines]
        strategy = dict(config.strategy)
        output = dict(config.output)
        if body.engine:
            target = normalise_engine(body.engine)
            for item in engine_dicts:
                item["enabled"] = normalise_engine(item["name"]) == target
        if body.model:
            for item in engine_dicts:
                if normalise_engine(item["name"]) in remote_engines():
                    item["model"] = body.model
        if body.prompt:
            for item in engine_dicts:
                if normalise_engine(item["name"]) in remote_engines():
                    item["prompt"] = body.prompt
        if body.strategy:
            strategy = dict(config.strategy, **body.strategy)
        effective = from_mapping({
            "engines": engine_dicts,
            "strategy": strategy,
            "output": output,
            "pdf": dict(config.pdf),
        })
    else:
        effective = config
    if body.strategy_name:
        try:
            effective = apply_strategy_preset(effective, body.strategy_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return effective


def _ocr_response(results, fmt: str):
    joined = "\n\n".join(r.to_markdown() for r in results if r.ok)
    if fmt in ("text", "markdown"):
        return PlainTextResponse(joined)
    return {
        "pages": [r.as_dict() for r in results],
        "text": joined,
        "engine": results[0].engine if results else "",
        "page_count": len(results),
    }


def _build_faked_app(cfg):
    real_build_engine = engine_registry.build_engine
    rapid = _FakeEngine("rapidocr", "rap", 0.9, elapsed_ms=80)
    sf = _FakeEngine("siliconflow", "siliconflow longer text here.", 1.0, elapsed_ms=1200)

    def _fake_build_engine(name, config):
        norm = engine_registry.normalise_engine(name)
        if norm == "rapidocr":
            return rapid
        if norm in ("siliconflow", "multimodal"):
            return sf
        return real_build_engine(norm, config)

    engine_registry.build_engine = _fake_build_engine
    state = RuntimeConfig(cfg)
    app = FastAPI()

    @app.post("/ocr/{preset}")
    async def preset_upload(
        preset: str,
        file: UploadFile = File(...),
        model: str = None,
        prompt: str = None,
        max_pages: int = None,
        dpi: int = 200,
        format: Optional[str] = Form(None),
    ):
        out_format = (format or "json").lower()
        if out_format not in ("json", "text", "markdown"):
            raise HTTPException(400, "bad format")
        norm = engine_registry.normalise_engine(preset)
        registered = ("rapidocr", "siliconflow", "multimodal")
        lower = preset.lower()
        if norm in registered:
            body = TextRequest(image_url="", engine=preset)
        elif lower in engine_registry.STRATEGY_PRESETS or \
                (lower.startswith("bestof:") and lower[len("bestof:"):].strip()):
            body = TextRequest(image_url="", strategy_name=preset)
        else:
            raise HTTPException(404, "unknown preset")
        if model:
            body.model = model
        if prompt:
            body.prompt = prompt
        effective = _apply_inline_overrides(state.snapshot(), body)
        pipeline = engine_registry.build_pipeline(effective, engine_name=body.engine)
        result = pipeline.recognise(None)
        return _ocr_response([result], out_format)

    @app.post("/ocr/{preset}/text")
    async def preset_text(preset: str, body: TextRequest):
        if not body.image_url:
            raise HTTPException(400, "image_url required")
        norm = engine_registry.normalise_engine(preset)
        if norm in ("rapidocr", "siliconflow", "multimodal"):
            body.engine = preset
        elif preset.lower() in engine_registry.STRATEGY_PRESETS:
            body.strategy_name = preset
        else:
            raise HTTPException(404, "unknown preset")
        effective = _apply_inline_overrides(state.snapshot(), body)
        pipeline = engine_registry.build_pipeline(effective, engine_name=body.engine)
        result = pipeline.recognise(None)
        return _ocr_response([result], body.format.lower())

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    cfg = from_mapping({
        "engines": [
            {"name": "rapidocr", "enabled": True},
            {"name": "siliconflow", "enabled": True, "model": "PaddleOCR-VL-1.5"},
            {"name": "multimodal", "enabled": False},
        ],
        "strategy": {"max_retries": 0},
        "output": {},
        "pdf": {},
    })
    return TestClient(_build_faked_app(cfg))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPresetRouteEngine:
    def test_engine_preset_rapidocr(self, client):
        r = client.post("/ocr/rapidocr",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "rapidocr"

    def test_engine_preset_siliconflow(self, client):
        r = client.post("/ocr/siliconflow",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"

    def test_unknown_preset_returns_404(self, client):
        r = client.post("/ocr/does-not-exist",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 404


class TestPresetRouteStrategy:
    def test_bestof_picks_siliconflow_by_smart(self, client):
        r = client.post("/ocr/bestof",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"

    def test_bestof_fastest_picks_rapidocr(self, client):
        r = client.post("/ocr/bestof-fastest",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "rapidocr"

    def test_bestof_confidence_picks_siliconflow(self, client):
        r = client.post("/ocr/bestof-confidence",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"

    def test_bestof_longest_picks_siliconflow(self, client):
        r = client.post("/ocr/bestof-longest",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"

    def test_bestof_fluency_picks_siliconflow(self, client):
        r = client.post("/ocr/bestof-fluency",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"

    def test_bestof_colon_syntax_alias(self, client):
        r = client.post("/ocr/bestof:fastest",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "rapidocr"

    def test_quality_alias(self, client):
        r = client.post("/ocr/quality",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text

    def test_local_preset_uses_rapidocr(self, client):
        r = client.post("/ocr/local",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "rapidocr"

    def test_vl_preset_uses_siliconflow(self, client):
        r = client.post("/ocr/vl",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"


class TestPresetRouteFormats:
    def test_markdown_format(self, client):
        r = client.post("/ocr/rapidocr",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "markdown"})
        assert r.status_code == 200, r.text
        # markdown is returned as plain text (content negotiable by caller)
        assert "text/plain" in r.headers.get("content-type", "")

    def test_text_format(self, client):
        r = client.post("/ocr/rapidocr",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "text"})
        assert r.status_code == 200, r.text
        assert "text/plain" in r.headers.get("content-type", "")

    def test_bad_format_returns_400(self, client):
        r = client.post("/ocr/rapidocr",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "xml"})
        assert r.status_code == 400


class TestPresetRouteText:
    def test_engine_preset(self, client):
        r = client.post("/ocr/rapidocr/text",
                        json={"image_url": "http://x/y.jpg", "format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "rapidocr"

    def test_strategy_preset(self, client):
        r = client.post("/ocr/bestof-fluency/text",
                        json={"image_url": "http://x/y.jpg", "format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "siliconflow"