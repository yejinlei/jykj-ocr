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
    """Engine double whose returned text echoes the resolved endpoint.

    ``to_markdown()`` renders only region text, and ``recognise()`` builds the
    region from ``_text`` — so appending the resolved ``base_url`` / ``api_key``
    here lets a test assert a per-request override reached the engine. Empty
    values contribute nothing, so existing assertions on the plain text keep
    passing.
    """

    def __init__(self, name: str, text: str, confidence: float, elapsed_ms: int = 1,
                 base_url: str = "", api_key: str = ""):
        self.name = name
        self.config = EngineConfig(name=name, base_url=base_url, api_key=api_key)
        self._text = text
        if base_url or api_key:
            self._text = f"{text}|{base_url}|{api_key}"
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

    if body.engine or body.model or body.base_url or body.api_key \
            or body.prompt or body.strategy:
        engine_dicts = [{
            "name": e.name,
            "enabled": e.enabled,
            "model": e.model,
            "base_url": e.base_url,
            "api_key": e.api_key,
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
        if body.base_url:
            for item in engine_dicts:
                if normalise_engine(item["name"]) in remote_engines():
                    item["base_url"] = body.base_url
        if body.api_key:
            for item in engine_dicts:
                if normalise_engine(item["name"]) in remote_engines():
                    item["api_key"] = body.api_key
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
    # Echo base_url / api_key back out of the engine so a test can see whether
    # a per-request override reached the engine's config, not just the body.
    rapid = _FakeEngine("rapidocr", "rap", 0.9, elapsed_ms=80)

    def _fake_build_engine(name, config, engine_config=None):
        norm = engine_registry.normalise_engine(name)
        if norm == "rapidocr":
            return rapid
        if norm == "multimodal":
            ec = engine_config or EngineConfig(name="multimodal")
            return _FakeEngine(
                "multimodal", "multimodal longer text here.", 1.0, elapsed_ms=1200,
                base_url=ec.resolved_base_url,
                api_key=ec.resolved_api_key,
            )
        return real_build_engine(norm, config)

    engine_registry.build_engine = _fake_build_engine
    state = RuntimeConfig(cfg)
    app = FastAPI()

    @app.post("/ocr/{preset}")
    async def preset_upload(
        preset: str,
        file: UploadFile = File(...),
        model: str = None,
        base_url: Optional[str] = Form(None),
        api_key: Optional[str] = Form(None),
        prompt: str = None,
        max_pages: int = None,
        dpi: int = 200,
        format: Optional[str] = Form(None),
    ):
        out_format = (format or "json").lower()
        if out_format not in ("json", "text", "markdown"):
            raise HTTPException(400, "bad format")
        norm = engine_registry.normalise_engine(preset)
        registered = ("rapidocr", "multimodal")
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
        if base_url:
            body.base_url = base_url
        if api_key:
            body.api_key = api_key
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
        if norm in ("rapidocr", "multimodal"):
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
            {"name": "multimodal", "enabled": True, "model": "PaddleOCR-VL-1.5"},
            {"name": "multimodal", "enabled": False, "model": "doubao-1-5-vision-pro-32k"},
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

    def test_engine_preset_multimodal(self, client):
        r = client.post("/ocr/multimodal",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"

    def test_siliconflow_alias_route_reports_multimodal(self, client):
        """``/ocr/siliconflow`` 曾是独立引擎的路由。现在 siliconflow 归一化
        为 multimodal:入口仍通(向后兼容),但结果引擎标为 multimodal 而不是
        siliconflow——不存在第三个引擎类型。"""
        r = client.post("/ocr/siliconflow",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"

    def test_unknown_preset_returns_404(self, client):
        r = client.post("/ocr/does-not-exist",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 404


class TestPresetRouteStrategy:
    def test_bestof_picks_multimodal_by_smart(self, client):
        r = client.post("/ocr/bestof",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"

    def test_bestof_fastest_picks_rapidocr(self, client):
        r = client.post("/ocr/bestof-fastest",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "rapidocr"

    def test_bestof_confidence_picks_multimodal(self, client):
        r = client.post("/ocr/bestof-confidence",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"

    def test_bestof_longest_picks_multimodal(self, client):
        r = client.post("/ocr/bestof-longest",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"

    def test_bestof_fluency_picks_multimodal(self, client):
        r = client.post("/ocr/bestof-fluency",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"

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

    def test_vl_preset_uses_multimodal(self, client):
        r = client.post("/ocr/vl",
                        files={"file": ("img.jpg", b"dummy", "image/jpeg")},
                        data={"format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["pages"][0]["engine"] == "multimodal"


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
        assert r.json()["pages"][0]["engine"] == "multimodal"


# ---------------------------------------------------------------------------
# GET /presets — tests the real app (the fake app above doesn't define it).
# ---------------------------------------------------------------------------
class TestPresetsEndpoint:
    @pytest.fixture
    def real_client(self):
        from jykj_ocr.server import create_app
        return TestClient(create_app())

    def test_returns_all_named_presets(self, real_client):
        r = real_client.get("/presets")
        assert r.status_code == 200
        body = r.json()
        assert "presets" in body
        for name in ("bestof-fluency", "fallback", "quality", "seq", "seq-any",
                     "local", "vl", "bestof"):
            assert name in body["presets"], f"missing {name}"

    def test_bestof_fluency_metadata(self, real_client):
        body = real_client.get("/presets").json()["presets"]
        entry = body["bestof-fluency"]
        assert entry["is_bestof"] is True
        assert entry["score_mode"] == "fluency"

    def test_local_and_vl_engine_scope(self, real_client):
        body = real_client.get("/presets").json()["presets"]
        assert body["local"]["engine_scope"] == "local_only"
        assert body["vl"]["engine_scope"] == "remote_vl_only"
        assert body["seq"]["engine_scope"] == "all_enabled"

    def test_colon_alias_present(self, real_client):
        body = real_client.get("/presets").json()["presets"]
        assert "bestof:<mode>" in body
        assert body["bestof:<mode>"]["is_bestof"] is True

    def test_every_operation_has_a_tag(self, real_client):
        """Every endpoint must carry a Swagger tag so /docs shows grouping."""
        spec = real_client.get("/openapi.json").json()
        for path, ops in spec["paths"].items():
            for method, op in ops.items():
                assert op.get("tags"), f"{method.upper()} {path} is untagged"


class TestEnginesEndpointMultipleInstances:
    """GET /engines' `configured` field must distinguish multiple
    ``multimodal`` instances in a multi-provider config."""

    @pytest.fixture
    def multi_client(self, tmp_path, monkeypatch):
        """A client whose server loads a config file we control."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("JYKJ_OCR_MULTIMODAL_API_KEY", raising=False)
        monkeypatch.delenv("MULTIMODAL_API_KEY", raising=False)
        monkeypatch.delenv("JYKJ_OCR_MULTIMODAL_API_KEY", raising=False)
        yield tmp_path, monkeypatch

    def _client_from_yaml(self, tmp_path, monkeypatch, yaml_body):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml_body, encoding="utf-8")
        monkeypatch.setenv("JYKJ_OCR_CONFIG", str(path))
        from jykj_ocr.server import create_app
        return TestClient(create_app())

    def test_multiple_multimodal_instances_are_distinguishable(self, multi_client):
        """A config listing two multimodal entries with different (base_url,
        model) must surface both in ``/engines.configured``, not collapse to
        a single string name."""
        tmp_path, monkeypatch = multi_client
        yaml_body = """
engines:
  - name: multimodal
    base_url: https://api.siliconflow.cn/v1
    model: PaddlePaddle/PaddleOCR-VL-1.5
    api_key: sk-sf
  - name: multimodal
    base_url: https://ark.cn-beijing.volces.com/api/v3
    model: doubao-1-5-vision-pro-32k
    api_key: sk-ark
"""
        client = self._client_from_yaml(tmp_path, monkeypatch, yaml_body)
        r = client.get("/engines")
        assert r.status_code == 200
        configured = r.json()["configured"]
        assert len(configured) == 2
        # Each entry exposes a (name, model, base_url) fingerprint so two
        # same-named instances are still distinguishable by the client.
        models = sorted(e["model"] for e in configured)
        base_urls = sorted(e["base_url"] for e in configured)
        assert models == ["PaddlePaddle/PaddleOCR-VL-1.5", "doubao-1-5-vision-pro-32k"]
        assert base_urls == [
            "https://api.siliconflow.cn/v1",
            "https://ark.cn-beijing.volces.com/api/v3",
        ]
        # No entry leaks an api_key.
        for entry in configured:
            assert "api_key" not in entry
            assert "has_api_key" not in entry

    def test_same_provider_different_accounts_stay_distinct(self, multi_client):
        """Two multimodal entries for the same platform with the same
        base_url+model but different api_key are separate accounts — dedupe
        must not collapse them."""
        tmp_path, monkeypatch = multi_client
        yaml_body = """
engines:
  - name: multimodal
    base_url: https://api.siliconflow.cn/v1
    model: PaddlePaddle/PaddleOCR-VL-1.5
    api_key: sk-sf-a
  - name: multimodal
    base_url: https://api.siliconflow.cn/v1
    model: PaddlePaddle/PaddleOCR-VL-1.5
    api_key: sk-sf-b
"""
        client = self._client_from_yaml(tmp_path, monkeypatch, yaml_body)
        r = client.get("/engines")
        configured = r.json()["configured"]
        assert len(configured) == 2
        assert all(e["name"] == "multimodal" for e in configured)
        assert all(e["model"] == "PaddlePaddle/PaddleOCR-VL-1.5" for e in configured)

    def test_config_endpoint_shows_has_api_key_bool_not_plaintext(self, multi_client):
        """GET /config must expose ``has_api_key`` (bool) but never echo the
        key itself — critical when a single deployment has many keys."""
        tmp_path, monkeypatch = multi_client
        yaml_body = """
engines:
  - name: multimodal
    base_url: https://api.siliconflow.cn/v1
    model: PaddlePaddle/PaddleOCR-VL-1.5
    api_key: sk-secret-1
"""
        client = self._client_from_yaml(tmp_path, monkeypatch, yaml_body)
        r = client.get("/config")
        assert r.status_code == 200
        body = r.json()
        assert body["engines"][0]["has_api_key"] is True
        assert "api_key" not in body["engines"][0]
        serialized = r.text
        assert "sk-secret-1" not in serialized


class TestStrategyKnobs:
    """Per-request ``retry_mode`` / ``score_mode`` / ``max_retries`` knobs.

    Exercised against :func:`_apply_inline_overrides` directly — the same
    code path the real ``POST /ocr/{preset}/text`` handler uses. No engine
    calls, no PIL/rapidocr/openai imports.
    """

    def _cfg(self):
        from jykj_ocr.config import from_mapping
        return from_mapping({
            "engines": [
                {"name": "rapidocr", "enabled": True},
                {"name": "multimodal", "enabled": True, "model": "PaddleOCR-VL-1.5"},
                {"name": "multimodal", "enabled": False, "model": "doubao-1-5-vision-pro-32k"},
            ],
            "strategy": {"name": "seq", "max_retries": 1},
            "output": {},
            "pdf": {},
        })

    def test_retry_mode_knob_overrides_config(self):
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", retry_mode="line_overlap")
        effective = _apply_inline_overrides(self._cfg(), body)
        assert effective.strategy["retry_mode"] == "line_overlap"

    def test_score_mode_knob_sets_bestof_mode(self):
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", score_mode="fastest")
        effective = _apply_inline_overrides(self._cfg(), body)
        assert effective.strategy["bestof_mode"] == "fastest"

    def test_max_retries_knob_overrides_config(self):
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", max_retries=3)
        effective = _apply_inline_overrides(self._cfg(), body)
        assert effective.strategy["max_retries"] == 3

    def test_retry_mode_knob_wins_over_preset(self):
        """Explicit ``retry_mode`` beats the ``seq`` preset's default."""
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", strategy_name="seq", retry_mode="any")
        effective = _apply_inline_overrides(self._cfg(), body)
        assert effective.strategy["retry_mode"] == "any"

    def test_score_mode_knob_wins_over_preset(self):
        """Explicit ``score_mode`` beats ``bestof`` preset's smart default."""
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(
            image_url="", strategy_name="bestof", score_mode="longest"
        )
        effective = _apply_inline_overrides(self._cfg(), body)
        assert effective.strategy["bestof_mode"] == "longest"

    def test_invalid_retry_mode_returns_400(self):
        from fastapi import HTTPException
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", retry_mode="bogus")
        with pytest.raises(HTTPException) as excinfo:
            _apply_inline_overrides(self._cfg(), body)
        assert excinfo.value.status_code == 400

    def test_invalid_score_mode_returns_400(self):
        from fastapi import HTTPException
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", score_mode="bogus")
        with pytest.raises(HTTPException) as excinfo:
            _apply_inline_overrides(self._cfg(), body)
        assert excinfo.value.status_code == 400

    def test_max_retries_negative_rejected(self):
        """Pydantic ``ge=0`` on ``TextRequest.max_retries`` blocks negative."""
        from jykj_ocr.server import TextRequest
        with pytest.raises(Exception):
            TextRequest(image_url="", max_retries=-1)

    def test_max_retries_zero_means_cascade(self):
        """The knob form of cascade: ``max_retries=0`` on ``/ocr/seq``."""
        from jykj_ocr.engine.registry import build_pipeline
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        body = TextRequest(image_url="", strategy_name="seq", max_retries=0)
        effective = _apply_inline_overrides(self._cfg(), body)
        pipeline = build_pipeline(effective)
        # max_retries=0 → StrategyEngine.retries == 0 (cascade semantics).
        assert pipeline.retries == 0

    def test_presets_endpoint_lists_cascade(self):
        """GET /presets must include the new cascade family."""
        from jykj_ocr.server import create_app
        client = TestClient(create_app())
        body = client.get("/presets").json()["presets"]
        for name in ("cascade", "cascade-low_conf", "cascade-line_overlap"):
            assert name in body
            assert body[name]["max_retries"] == 0
            assert body[name]["retry_mode"] in (
                "no_text", "low_confidence", "line_overlap"
            )


class TestPerRequestEndpointOverride:
    """Per-request ``base_url`` / ``api_key`` knobs — the ``/vl`` flexibility.

    Neither falls back silently: an omitted value keeps the entry's own
    ``base_url`` / ``api_key``, which is where the env-var chain
    (``OPENAI_BASE_URL_*`` / ``JYKJ_OCR_*_API_KEY`` / ``OPENAI_API_KEY``)
    kicks in via ``resolved_base_url`` / ``resolved_api_key``. Only
    ``POST /config`` with ``null`` reverts an explicit value.

    The knobs are applied to *every* remote entry (like ``model`` / ``prompt``),
    and ``vl`` disables — not deletes — the extra instances, so assertions go
    through ``apply_strategy_preset(..., "vl")`` and read the surviving entry.
    """

    def _cfg(self):
        from jykj_ocr.config import from_mapping
        return from_mapping({
            "engines": [
                {"name": "rapidocr", "enabled": True},
                {"name": "multimodal", "enabled": True, "model": "PaddleOCR-VL-1.5",
                 "base_url": "https://api.siliconflow.cn/v1", "api_key": "sk-entry"},
                {"name": "multimodal", "enabled": False, "model": "doubao-1-5-vision-pro-32k"},
            ],
            "strategy": {"max_retries": 0},
            "output": {},
            "pdf": {},
        })

    def _vl_entry(self, body):
        """The one surviving remote entry after the ``vl`` preset."""
        from jykj_ocr.engine.registry import apply_strategy_preset
        from jykj_ocr.server import _apply_inline_overrides
        effective = _apply_inline_overrides(self._cfg(), body)
        return apply_strategy_preset(effective, "vl").engines[1]

    def test_api_key_overrides_remote_entry(self):
        from jykj_ocr.server import TextRequest
        remote = self._vl_entry(TextRequest(image_url="", api_key="sk-request"))
        assert remote.api_key == "sk-request"
        # Endpoint and model are untouched by the key knob.
        assert remote.base_url == "https://api.siliconflow.cn/v1"
        assert remote.model == "PaddleOCR-VL-1.5"

    def test_base_url_overrides_remote_entry(self):
        from jykj_ocr.server import TextRequest
        remote = self._vl_entry(TextRequest(image_url="", base_url="https://ark.example.com/api/v3"))
        assert remote.base_url == "https://ark.example.com/api/v3"
        # The key is untouched, so it still resolves from the entry or env.
        assert remote.api_key == "sk-entry"

    def test_omitted_credentials_keep_entry_values(self):
        """A plain ``/vl`` request must not wipe the configured credentials —
        that is what makes the env-var fallback survive."""
        from jykj_ocr.server import TextRequest
        remote = self._vl_entry(TextRequest(image_url="x.png"))
        assert remote.api_key == "sk-entry"
        assert remote.base_url == "https://api.siliconflow.cn/v1"

    def test_empty_string_credentials_are_a_noop(self):
        """An empty Form value (``-F api_key=""``) is falsy and must not clear
        the entry's key the way ``POST /config`` with ``null`` does."""
        from jykj_ocr.server import TextRequest
        remote = self._vl_entry(TextRequest(image_url="x.png", api_key="", base_url=""))
        assert remote.api_key == "sk-entry"
        assert remote.base_url == "https://api.siliconflow.cn/v1"

    def test_model_and_credential_combination(self):
        from jykj_ocr.server import TextRequest
        remote = self._vl_entry(TextRequest(
            image_url="x.png",
            model="Qwen/Qwen3-VL-30B-A3B-Instruct",
            base_url="https://ark.example.com/api/v3",
            api_key="sk-second-account",
        ))
        assert remote.model == "Qwen/Qwen3-VL-30B-A3B-Instruct"
        assert remote.base_url == "https://ark.example.com/api/v3"
        assert remote.api_key == "sk-second-account"

    def test_override_does_not_mutate_base_config(self):
        """One-shot semantics: the caller's config is never rewritten."""
        from jykj_ocr.server import _apply_inline_overrides, TextRequest
        cfg = self._cfg()
        _apply_inline_overrides(cfg, TextRequest(image_url="x.png", api_key="sk-request"))
        assert cfg.engines[1].api_key == "sk-entry"
        assert cfg.engines[2].api_key == ""

    def test_form_field_reaches_the_engine(self, client):
        """End-to-end: ``-F api_key`` / ``-F base_url`` on ``/ocr/{preset}``
        lands in the engine's config, not just the request body."""
        r = client.post(
            "/ocr/vl",
            files={"file": ("img.jpg", b"dummy", "image/jpeg")},
            data={
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "base_url": "https://ark.example.com/api/v3",
                "api_key": "sk-per-request",
                "format": "json",
            },
        )
        assert r.status_code == 200, r.text
        assert "sk-per-request" in r.json()["pages"][0]["text"]
        assert "ark.example.com" in r.json()["pages"][0]["text"]

    def test_credential_knob_ignored_on_local_engine_route(self, client):
        """The knobs only touch remote engines; a local-only route ignores them
        rather than silently smuggling a key into an offline engine."""
        r = client.post(
            "/ocr/local",
            files={"file": ("img.jpg", b"dummy", "image/jpeg")},
            data={"api_key": "sk-per-request", "format": "json"},
        )
        assert r.status_code == 200, r.text
        assert "sk-per-request" not in r.json()["pages"][0]["text"]

    def test_credential_knob_via_json_route(self, client):
        r = client.post(
            "/ocr/vl/text",
            json={
                "image_url": "http://x/y.jpg",
                "api_key": "sk-per-request",
                "format": "json",
            },
        )
        assert r.status_code == 200, r.text
        assert "sk-per-request" in r.json()["pages"][0]["text"]


class TestTextRequestSourceResolution:
    """``TextRequest.source()`` — the three image input forms.

    ``image_data`` is documented as a *complete* data URI, so it must be passed
    through untouched. A regression here turns a valid data URI into
    ``data:application/octet-stream,data:image/...``, which base64-decodes to
    garbage and the caller sees as HTTP 400. Caught by the real-model E2E run.
    """

    def test_image_data_with_prefix_is_untouched(self):
        from jykj_ocr.server import TextRequest
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        body = TextRequest(image_data=uri)
        assert body.source() == uri

    def test_image_data_without_prefix_gets_one(self):
        """Callers who stripped the prefix themselves should still work."""
        from jykj_ocr.server import TextRequest
        body = TextRequest(image_data="/9j/4AAQ")
        src = body.source()
        assert src.startswith("data:")
        assert src.endswith("/9j/4AAQ")
        # Exactly one prefix — the nested form is what broke the endpoint.
        assert src.count("data:") == 1

    def test_image_b64_gets_prefixed(self):
        from jykj_ocr.server import TextRequest
        body = TextRequest(image_b64="/9j/4AAQ")
        assert body.source() == "data:image/octet-stream;base64,/9j/4AAQ"

    def test_image_url_passes_through(self):
        from jykj_ocr.server import TextRequest
        assert TextRequest(image_url="scan.png").source() == "scan.png"

    def test_exactly_one_input_required(self):
        """None or multiple image sources are rejected by ``source()``.

        The invariant is checked in ``source()``, not the constructor — the
        endpoint catches the resulting HTTPException and maps it to 400.
        """
        from fastapi import HTTPException
        from jykj_ocr.server import TextRequest
        for kwargs in ({}, {"image_url": "a.png", "image_b64": "AA=="},
                       {"image_b64": "AA==", "image_data": "AA=="}):
            with pytest.raises(HTTPException) as excinfo:
                TextRequest(**kwargs).source()
            assert excinfo.value.status_code == 400