# -*- coding: utf-8 -*-
"""FastAPI HTTP interface for jykj_ocr.

Two families of endpoints:

1. **OCR 识别** —— ``POST /ocr`` (multipart), ``POST /ocr/text`` (image URL).
2. **临时配置** —— ``GET /config``, ``POST /config``, ``DELETE /config``.
   运行时覆盖模型、引擎、引擎顺序、重试策略等，无需重启或改文件。

Run locally:

    pip install fastapi uvicorn python-multipart
    uvicorn jykj_ocr.server:app --host 0.0.0.0 --port 8000

or ``python -m jykj_ocr serve`` (port from ``JYKJ_OCR_PORT``).

Credentials are never hardcoded; they come from the environment
(``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``, or engine-specific
``SILICONFLOW_API_KEY``) or ``config/config.yaml``.
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .config import Config, EngineConfig, from_mapping, load_config, normalise_engine
from .engine import base as engine_base
from .engine import describe_engines
from .engine.inputs import attach_pil, load as load_source
from .engine.registry import (
    apply_strategy_preset,
    build_engine,
    build_pipeline,
    build_strategy,
    engines_from_config,
    remote_engines,
    resolve_retry_check,
)
from .models import OCRResult, rebuild_text_from_regions
from .strategy import StrategyError, TimedOCR

LOGGER = logging.getLogger(__name__)

TEXT_FORMATS = ("text", "markdown")
VALID_FORMATS = ("json", *TEXT_FORMATS)

#: Config keys that ``POST /config`` is allowed to touch. Anything else is
#: rejected so a caller cannot smuggle in arbitrary keys.
_OVERRIDABLE_KEYS = frozenset(("engines", "strategy", "output", "pdf"))
_ENGINE_FIELDS = {
    "name",
    "enabled",
    "model",
    "base_url",
    "api_key",
    "temperature",
    "timeout",
    "max_tokens",
    "lang",
    "prompt",
    "prompt_file",
}


class RuntimeConfig:
    """Thread-safe holder for the currently effective configuration.

    ``overrides`` is a partial mapping that is merged over ``base`` on every
    read. Because ``from_mapping`` deep-copies, callers cannot mutate state
    through the returned object.
    """

    def __init__(self, base: Config) -> None:
        self._lock = threading.Lock()
        self._base = base
        self._overrides: Dict[str, Any] = {}

    def snapshot(self) -> Config:
        with self._lock:
            merged: Dict[str, Any] = {
                "engines": [
                    {
                        "name": e.name,
                        "enabled": e.enabled,
                        "model": e.model,
                        "base_url": e.base_url,
                        "temperature": e.temperature,
                        "timeout": e.timeout,
                        "max_tokens": e.max_tokens,
                        "lang": e.lang,
                        "prompt": e.prompt,
                        **e.extra,
                    }
                    for e in self._base.engines
                ],
                "strategy": dict(self._base.strategy),
                "output": dict(self._base.output),
                "pdf": dict(self._base.pdf),
            }
            for key, value in self._overrides.items():
                merged[key] = copy.deepcopy(value)
            return from_mapping(merged)

    def override(self, patch: Dict[str, Any]) -> None:
        with self._lock:
            bad = set(patch) - _OVERRIDABLE_KEYS
            if bad:
                raise ValueError(f"unsupported config keys: {sorted(bad)}")
            for key, value in patch.items():
                if value is None:
                    self._overrides.pop(key, None)
                else:
                    self._overrides[key] = copy.deepcopy(value)

    def clear(self) -> None:
        with self._lock:
            self._overrides = {}

    def has_overrides(self) -> bool:
        with self._lock:
            return bool(self._overrides)


def _engine_view(engine: EngineConfig) -> Dict[str, Any]:
    """Config view for clients — never echo API keys."""
    return {
        "name": engine.name,
        "enabled": engine.enabled,
        "model": engine.resolved_model,
        "base_url": engine.resolved_base_url,
        "timeout": engine.timeout,
        "temperature": engine.temperature,
        "max_tokens": engine.max_tokens,
        "lang": engine.lang,
        "has_api_key": bool(engine.resolved_api_key),
    }


def _config_view(config: Config) -> Dict[str, Any]:
    return {
        "engines": [_engine_view(e) for e in config.engines],
        "strategy": dict(config.strategy),
        "output": dict(config.output),
        "pdf": dict(config.pdf),
        "configured_order": [e.name for e in config.engines],
    }


class TextRequest(BaseModel):
    """Request body for the image-URL OCR endpoint."""

    image_url: str
    engine: Optional[str] = None
    engines: Optional[List[str]] = None
    max_pages: Optional[int] = None
    dpi: int = 200
    format: str = "json"
    model: Optional[str] = None
    prompt: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None
    strategy_name: Optional[str] = None


class ConfigRequest(BaseModel):
    """Partial runtime config patch for ``POST /config``.

    Set a field to ``null`` to revert it to the config-file value.
    """

    engines: Optional[List[Dict[str, Any]]] = None
    strategy: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    pdf: Optional[Dict[str, Any]] = None


def _engine_raw(engine: EngineConfig) -> Dict[str, Any]:
    """Plain-dict form of an engine config, safe to feed back to from_mapping."""
    return {
        "name": engine.name,
        "enabled": engine.enabled,
        "model": engine.model,
        "base_url": engine.base_url,
        "temperature": engine.temperature,
        "timeout": engine.timeout,
        "max_tokens": engine.max_tokens,
        "lang": engine.lang,
        "prompt": engine.prompt,
    }


def _apply_inline_overrides(config: Config, body: TextRequest) -> Config:
    """Apply per-request ``engine``/``model``/``prompt``/``strategy``/``strategy_name``.

    This lets a caller pick a different model or a named preset
    (``local``/``vl``/``fallback``/``quality``) for a single request without a
    global ``POST /config``, e.g. ``{"image_url": "...", "strategy_name": "quality"}``.

    Returns a new :class:`Config`; the input is never mutated (one-shot).
    """
    engine_dicts = [_engine_raw(e) for e in config.engines]
    strategy = dict(config.strategy)
    output = dict(config.output)
    changed = False

    if body.engine:
        target = normalise_engine(body.engine)
        for item in engine_dicts:
            item["enabled"] = normalise_engine(item["name"]) == target
        changed = True

    if body.model:
        for item in engine_dicts:
            if normalise_engine(item["name"]) in remote_engines():
                item["model"] = body.model
        changed = True

    if body.prompt:
        for item in engine_dicts:
            if normalise_engine(item["name"]) in remote_engines():
                item["prompt"] = body.prompt
        changed = True

    if body.strategy:
        strategy = dict(config.strategy, **body.strategy)
        changed = True

    if not changed:
        effective = config
    else:
        effective = from_mapping(
            {
                "engines": engine_dicts,
                "strategy": strategy,
                "output": output,
                "pdf": dict(config.pdf),
            }
        )

    if body.strategy_name:
        try:
            effective = apply_strategy_preset(effective, body.strategy_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return effective


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config_path: Optional config file. When omitted, ``load_config`` honours
            ``JYKJ_OCR_CONFIG`` and then falls back to defaults.
    """
    state = RuntimeConfig(load_config(config_path))
    app = FastAPI(
        title="jykj_ocr",
        version="0.1.0",
        description="多引擎 OCR 服务：本地 RapidOCR + 多模态 OCR 大模型（硅基流动等）",
    )

    # -- error mapping ------------------------------------------------------
    @app.exception_handler(engine_base.InputError)
    async def _handle_input_error(_: Any, exc: engine_base.InputError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(engine_base.EngineNotAvailable)
    async def _handle_not_available(
        _: Any, exc: engine_base.EngineNotAvailable
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(engine_base.EngineError)
    async def _handle_engine_error(_: Any, exc: engine_base.EngineError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc), "engine": getattr(exc, "engine", "")},
        )

    @app.exception_handler(StrategyError)
    async def _handle_strategy_error(_: Any, exc: StrategyError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # -- pipeline -----------------------------------------------------------
    def _pipeline(config: Config, engine_name: Optional[str] = None) -> Any:
        """A single forced engine, or the full configured pipeline.

        Delegates to :func:`build_pipeline` so it honours every strategy the
        registry understands — ``seq*`` retry chains *and* ``bestof*`` (which
        requires :class:`BestofEngine`, not :class:`StrategyEngine`).
        """
        if engine_name:
            return TimedOCR(build_engine(engine_name, config))
        return build_pipeline(config)

    def _recognise(
        source: str,
        *,
        config: Config,
        engine_name: Optional[str] = None,
        max_pages: Optional[int] = None,
        dpi: int = 200,
    ) -> List[OCRResult]:
        pages = load_source(source, max_pages=max_pages, dpi=int(dpi))
        if not pages:
            raise engine_base.InputError("no pages could be loaded from the input")
        pipeline = _pipeline(config, engine_name)
        results = [pipeline.recognise(attach_pil(page)) for page in pages]
        if config.output_value("reorder_lines"):
            for result in results:
                result.text = rebuild_text_from_regions(result)
        return results

    def _joined(results: List[OCRResult]) -> str:
        return "\n\n".join(r.to_markdown() for r in results if r.ok)

    def _ocr_response(results: List[OCRResult], fmt: str) -> Any:
        joined = _joined(results)
        if fmt in TEXT_FORMATS:
            return PlainTextResponse(joined)
        return {
            "pages": [r.as_dict() for r in results],
            "text": joined,
            "engine": results[0].engine if results else "",
            "page_count": len(results),
        }

    # -- 状态与临时配置接口 ---------------------------------------------------
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "engines": list(describe_engines().keys())}

    @app.get("/engines")
    async def engines() -> Dict[str, Any]:
        config = state.snapshot()
        return {"engines": describe_engines(), "configured": [e.name for e in config.engines]}

    @app.get("/config")
    async def get_config() -> Dict[str, Any]:
        """当前生效配置（含运行时覆盖），不返回 API key 明文。"""
        return {
            **_config_view(state.snapshot()),
            "overridden": state.has_overrides(),
        }

    @app.post("/config")
    async def set_config(body: ConfigRequest) -> Dict[str, Any]:
        """运行时覆盖配置：设置模型、引擎、策略等。

        例::

            POST /config
            {"engines": [{"name": "siliconflow", "model": "qwen-vl-max"}]}

        例（把快速引擎放在前面）::

            POST /config
            {"strategy": {"max_retries": 2, "retry_mode": "low_confidence"}}

        把字段设为 ``null`` 可还原为配置文件中的值。
        """
        patch = body.model_dump()
        try:
            state.override(patch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        config = state.snapshot()
        LOGGER.info("runtime config overridden: %s", sorted(k for k, v in patch.items() if v))
        return {
            "ok": True,
            **_config_view(config),
            "overridden": state.has_overrides(),
        }

    @app.delete("/config")
    async def reset_config() -> Dict[str, Any]:
        """清除所有运行时覆盖，回到配置文件状态。"""
        state.clear()
        return {"ok": True, **_config_view(state.snapshot()), "overridden": False}

    # -- OCR 识别接口 -------------------------------------------------------
    @app.post("/ocr")
    async def ocr_upload(
        file: UploadFile = File(..., description="图片或 PDF"),
        engine: Optional[str] = Form(None),
        engines: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        strategy: Optional[str] = Form(None),
        strategy_name: Optional[str] = Form(
            None, description="一次性策略预设：local / vl / fallback / quality"
        ),
        max_pages: Optional[int] = Form(None),
        dpi: int = Form(200),
        format: str = Form("json"),
    ) -> Any:
        """识别上传的图片或 PDF 中的文字。

        ``strategy`` / ``model`` 等为 JSON 字符串形式的临时配置，优先级高于
        ``/config`` 的运行时覆盖。``strategy_name`` 按命名预设整体切换引擎链
        （``local`` 仅本地 / ``vl`` 仅大模型 / ``fallback`` 回退链 /
        ``quality`` 回退链+窜行降级+阅读顺序重排），同样只对本请求生效。
        """
        out_format = (format or "json").lower()
        if out_format not in VALID_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown format '{format}'; use json, text or markdown",
            )
        try:
            strategy_map = _parse_json_field(strategy, "strategy")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        body = TextRequest(
            image_url="",
            engine=engine,
            model=model,
            prompt=prompt,
            strategy=strategy_map,
            strategy_name=strategy_name,
        )
        effective = _apply_inline_overrides(state.snapshot(), body)

        suffix = os.path.splitext(file.filename or "upload.bin")[1] or ".bin"
        tmp: Optional[tempfile.NamedTemporaryFile] = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(await file.read())
            tmp.close()
            results = _recognise(
                tmp.name,
                config=effective,
                engine_name=engine,
                max_pages=max_pages,
                dpi=dpi,
            )
        finally:
            if tmp is not None:
                path = tmp.name
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return _ocr_response(results, out_format)

    @app.post("/ocr/text")
    async def ocr_text(body: TextRequest) -> Any:
        """识别图片 URL（``http(s)://``）中的文字。"""
        if not body.image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        effective = _apply_inline_overrides(state.snapshot(), body)
        results = _recognise(
            body.image_url,
            config=effective,
            engine_name=body.engine,
            max_pages=body.max_pages,
            dpi=body.dpi,
        )
        return _ocr_response(results, body.format.lower())

    return app


def _parse_json_field(value: Optional[str], field: str) -> Optional[Dict[str, Any]]:
    """Decode an optional JSON object supplied as a string form field."""
    if value is None or not str(value).strip():
        return None
    import json

    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise ValueError(f"'{field}' must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"'{field}' must be a JSON object")
    return parsed


#: Default app instance, configurable through ``JYKJ_OCR_CONFIG``.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "jykj_ocr.server:app",
        host="0.0.0.0",
        port=int(os.getenv("JYKJ_OCR_PORT", "8000")),
        reload=False,
    )
