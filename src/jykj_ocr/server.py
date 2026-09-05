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
from pydantic import BaseModel, Field

from .config import Config, EngineConfig, from_mapping, load_config, normalise_engine
from .engine import base as engine_base
from .engine import describe_engines
from .engine.inputs import attach_pil, load as load_source
from .engine.registry import (
    STRATEGY_PRESETS,
    VALID_RETRY_MODES,
    VALID_SCORE_MODES,
    apply_strategy_preset,
    build_engine,
    build_pipeline,
    build_strategy,
    describe_presets,
    engines_from_config,
    remote_engines,
    resolve_retry_check,
)
from .models import OCRResult, rebuild_text_from_regions
from .strategy import StrategyError, TimedOCR

LOGGER = logging.getLogger(__name__)

TEXT_FORMATS = ("text", "markdown")
VALID_FORMATS = ("json", *TEXT_FORMATS)

# OpenAPI 响应组件:统一的错误响应结构
_ERROR_RESPONSES = {
    400: {"description": "坏输入(文件不存在/参数非法/strategy JSON 非对象)"},
    404: {"description": "未知 preset:路由既非已注册引擎也非策略预设"},
    422: {"description": "引擎不可用(缺依赖/key)或策略链耗尽"},
    502: {"description": "引擎调用失败(超时/HTTP 402/网络错误)"},
}

_PRESET_EXAMPLES = (
    "local, vl, seq, seq-any, seq-low_conf, seq-line_overlap, "
    "cascade, cascade-low_conf, cascade-line_overlap, "
    "bestof, bestof-smart, bestof-fastest, bestof-confidence, "
    "bestof-longest, bestof-fluency, fallback, quality, bestof:<mode>"
)

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
                        "api_key": e.api_key,
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
    """Request body for the image-input OCR endpoints.

    Exactly one of ``image_url`` / ``image_b64`` / ``image_data`` must be
    provided. They are resolved into a single source string before passing to
    :func:`engine.inputs.load`, so all three are handled uniformly.
    """

    image_url: Optional[str] = Field(
        None,
        description=(
            "图片 URL(本地路径 / http(s):// URL)。与 image_b64、image_data 三选一,"
            "必须恰好传一个。例:`\"https://example.com/scan.png\"` 或 `\"/tmp/img.jpg\"`。"
        ),
        example="https://example.com/scan.png",
    )
    image_b64: Optional[str] = Field(
        None,
        description=(
            "base64 编码的图片字节(不含 `data:` 前缀)。服务端自动补 "
            "`data:image/octet-stream;base64,` 前缀。例:`\"iVBORw0KGgoAAAANSUhEUg...\"`。"
        ),
    )
    image_data: Optional[str] = Field(
        None,
        description=(
            "完整 data URI(含 `data:image/...;base64,` 前缀),如浏览器 `<img src>` 直传。"
            "例:`\"data:image/png;base64,iVBORw0KGgo...\"`。"
        ),
    )
    engine: Optional[str] = Field(None, description="强制单引擎(如 rapidocr / siliconflow),覆盖 strategy")
    max_pages: Optional[int] = Field(None, description="PDF 页数上限")
    dpi: int = Field(200, description="PDF 渲染 DPI")
    format: str = Field("json", description="输出格式:json / text / markdown")
    model: Optional[str] = Field(None, description="覆盖远程引擎的 model 名")
    prompt: Optional[str] = Field(None, description="覆盖远程引擎的 prompt")
    strategy: Optional[Dict[str, Any]] = Field(None, description="临时策略对象(合并进 config.strategy)")
    strategy_name: Optional[str] = Field(
        None, description=f"一次性策略预设:{_PRESET_EXAMPLES}"
    )
    retry_mode: Optional[str] = Field(
        None,
        description=(
            "覆盖 seq 家族的 retry 判定:no_text / low_confidence / "
            "line_overlap / any / none。bestof 家族忽略。"
        ),
    )
    score_mode: Optional[str] = Field(
        None,
        description=(
            "覆盖 bestof 家族的评分函数:smart / fastest / "
            "highest_confidence / longest / fluency。seq 家族忽略。"
        ),
    )
    max_retries: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "覆盖同引擎重试次数(仅 seq 家族生效)。0 等价于 cascade "
            "(不重试直接降级到下一引擎),≥1 用第 1 次+该值。"
        ),
    )

    def source(self) -> str:
        """Return the resolved input source.

        Priority: ``image_data`` (raw bytes) > ``image_b64`` (base64) >
        ``image_url`` (path / http / data URI). Exactly one must be set.
        """
        provided = sum(
            1 for v in (self.image_url, self.image_b64, self.image_data) if v
        )
        if provided != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "provide exactly one of image_url, image_b64, or image_data"
                ),
            )
        if self.image_data is not None:
            # image_data is a *complete* data URI (see the field doc above), so a
            # bare prefix would turn it into "data:...,data:image/..." garbage.
            # Tolerate an unprefixed payload too — callers who have already
            # stripped the prefix should not need to re-add it.
            return self.image_data if self.image_data.startswith("data:") \
                else f"data:image/octet-stream;base64,{self.image_data}"
        if self.image_b64 is not None:
            return f"data:image/octet-stream;base64,{self.image_b64}"
        return self.image_url  # type: ignore[return-value]


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
        "api_key": engine.api_key,
        "temperature": engine.temperature,
        "timeout": engine.timeout,
        "max_tokens": engine.max_tokens,
        "lang": engine.lang,
        "prompt": engine.prompt,
    }


def _apply_inline_overrides(config: Config, body: TextRequest) -> Config:
    """Apply per-request ``engine``/``model``/``prompt``/``strategy``/``strategy_name``.

    This lets a caller pick a different model or a named preset
    (``local``/``vl``/``seq*``/``bestof*``/``fallback``/``quality``) for a single
    request without a global ``POST /config``, e.g. ``{"image_url": "...",
    "strategy_name": "bestof"}``.

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

    # ``retry_mode`` / ``score_mode`` / ``max_retries`` — per-request
    # strategy knobs. Each overrides only what it's meant for: retry_mode
    # touches seq-family retry predicates; score_mode is picked up by
    # build_pipeline's bestof branch; max_retries applies to StrategyEngine
    # only. Invalid values return 400 rather than being silently ignored.
    if body.retry_mode is not None:
        if body.retry_mode not in VALID_RETRY_MODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown retry_mode {body.retry_mode!r}; choose one of "
                    f"{', '.join(sorted(VALID_RETRY_MODES))}"
                ),
            )
        strategy["retry_mode"] = body.retry_mode
        changed = True

    if body.score_mode is not None:
        if body.score_mode not in VALID_SCORE_MODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown score_mode {body.score_mode!r}; choose one of "
                    f"{', '.join(sorted(VALID_SCORE_MODES))}"
                ),
            )
        strategy["bestof_mode"] = body.score_mode
        changed = True

    if body.max_retries is not None:
        strategy["max_retries"] = body.max_retries
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
        # Knobs re-apply AFTER the preset so an explicit ``retry_mode`` /
        # ``score_mode`` / ``max_retries`` wins over the preset default.
        # ``apply_strategy_preset`` returns a fresh deepcopy, so mutating
        # its ``strategy`` dict here does not leak back to ``config``.
        if body.retry_mode is not None:
            effective.strategy["retry_mode"] = body.retry_mode
        if body.score_mode is not None:
            effective.strategy["bestof_mode"] = body.score_mode
        if body.max_retries is not None:
            effective.strategy["max_retries"] = body.max_retries
    return effective


def _load_dotenv_once(path: str = ".env") -> None:
    from .config import load_dotenv as _loader

    _loader(path)


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config_path: Optional config file. When omitted, ``load_config`` honours
            ``JYKJ_OCR_CONFIG`` and then falls back to defaults.
    """
    _load_dotenv_once()
    state = RuntimeConfig(load_config(config_path))
    app = FastAPI(
        title="jykj_ocr",
        version="0.1.0",
        description=(
            "多引擎 OCR 服务：本地 RapidOCR(离线) + 多模态 OCR 大模型（硅基流动 / 任意 "
            "OpenAI 兼容端点），通过策略层编排引擎调用顺序与重试逻辑。\n\n"
            "## 策略预设\n\n"
            "一次性 `strategy_name`(不改动服务端配置):`local` / `vl` / "
            "`seq` / `seq-any` / `seq-low_conf` / `seq-line_overlap` / "
            "`cascade` / `cascade-low_conf` / `cascade-line_overlap` / "
            "`bestof` / `bestof-smart` / `bestof-fastest` / "
            "`bestof-confidence` / `bestof-longest` / `bestof-fluency` / "
            "`fallback` / `quality` / `bestof:<mode>`。\n\n"
            "策略旋钮(per-request):`retry_mode`(no_text/low_confidence/"
            "line_overlap/any/none)、`score_mode`(smart/fastest/"
            "highest_confidence/longest/fluency)、`max_retries`(0 等价于 cascade)。\n\n"
            "## 图片来源\n\n"
            "图片/PDF 文件(multipart)、`http(s)://` URL、本地路径、纯 base64、"
            "完整 data URI 均支持。\n\n"
            "## 运行时配置\n\n"
            "`POST /config` 可热改模型/引擎/策略,无需重启,不回显 API key 明文。"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        tags=[
            {
                "name": "OCR 识别",
                "description": (
                    "图片/PDF 文字识别。四种 OCR 端点返回结构完全一致:"
                    "{pages, text, engine, page_count};"
                    "format=text/markdown 时退化为纯文本。"
                ),
            },
            {
                "name": "配置与状态",
                "description": (
                    "查看或热改运行时配置。API key 只接受写入,GET /config 不回显明文;"
                    "DELETE /config 清除所有运行时覆盖,回到配置文件状态。"
                ),
            },
        ],
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
    @app.get("/health", tags=["配置与状态"],
         summary="健康检查",
         description="返回服务状态与已注册引擎名列表(不含当前配置)。")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "engines": list(describe_engines().keys())}

    @app.get("/engines", tags=["配置与状态"],
             summary="可用引擎列表",
             description=(
                 "返回所有已注册引擎的描述 + 当前配置中的引擎清单。`configured` "
                 "是实例级列表(每个 multimodal 条目各占一行,附带 model/base_url "
                 "指纹),便于在多实例部署下区分同名引擎。"
             ))
    async def engines() -> Dict[str, Any]:
        config = state.snapshot()
        return {
            "engines": describe_engines(),
            "configured": [
                {
                    "name": e.name,
                    "resolved_name": e.resolved_name,
                    "enabled": e.enabled,
                    "model": e.resolved_model,
                    "base_url": e.resolved_base_url,
                }
                for e in config.engines
            ],
        }

    @app.get("/presets", tags=["配置与状态"],
             summary="可用策略预设",
             description=(
                 "列出全部命名策略预设(含评分模式、重试判定、引擎范围),与 CLI "
                 "`--strategy-name`、HTTP `strategy_name` 参数、`/ocr/{preset}` 路由"
                 "一一对应。14 个显式预设 + `bestof:<mode>` 冒号语法别名,共 15 项。"
             ),
             responses={200: {"description": "预设清单(name → 元数据)"}})
    async def presets() -> Dict[str, Any]:
        return {"presets": describe_presets()}


    @app.get("/config", tags=["配置与状态"],
             summary="当前生效配置",
             description="返回当前生效配置(配置文件 + 运行时覆盖合并);不返回 API key 明文,仅有 has_api_key 布尔。")
    async def get_config() -> Dict[str, Any]:
        """当前生效配置（含运行时覆盖），不返回 API key 明文。"""
        return {
            **_config_view(state.snapshot()),
            "overridden": state.has_overrides(),
        }

    @app.post("/config", tags=["配置与状态"],
              summary="运行时覆盖配置",
              description=(
                  "部分覆盖运行时配置:设置模型、引擎、策略等,无需重启。"
                  "把字段设为 null 可还原为配置文件中的值。可覆盖的顶层键:"
                  "engines / strategy / output / pdf。"
                  "\n\n示例: {\"engines\": [{\"name\": \"siliconflow\", \"model\": \"qwen-vl-max\"}]} 或 {\"strategy\": {\"max_retries\": 2}}"
              ),
              responses={400: {"description": "不支持的配置键或值非法"}},
              openapi_extra={
                  "examples": {
                      "切换模型": {
                          "summary": "切换 siliconflow 模型",
                          "value": {"engines": [{"name": "siliconflow", "model": "qwen-vl-max"}]},
                      },
                      "调整重试策略": {
                          "summary": "低置信度时多重试一次",
                          "value": {"strategy": {"max_retries": 2, "retry_mode": "low_confidence"}},
                      },
                  }
              })
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

    @app.delete("/config", tags=["配置与状态"],
             summary="清除运行时覆盖",
             description="回退到配置文件状态,丢弃所有 POST /config 的覆盖。")
    async def reset_config() -> Dict[str, Any]:
        """清除所有运行时覆盖，回到配置文件状态。"""
        state.clear()
        return {"ok": True, **_config_view(state.snapshot()), "overridden": False}

    # -- OCR 识别接口 -------------------------------------------------------
    @app.post("/ocr", tags=["OCR 识别"],
              summary="上传图片/PDF 识别",
              description="multipart 上传文件或 URL 识别,支持一次指定 engine / strategy_name / model 等覆盖。",
              responses=_ERROR_RESPONSES | {
                  200: {"description": "识别成功,统一结构 {pages, text, engine, page_count}"}
              })
    async def ocr_upload(
        file: UploadFile = File(..., description="图片或 PDF"),
        engine: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        strategy: Optional[str] = Form(None),
        strategy_name: Optional[str] = Form(
            None, description=f"一次性策略预设：{_PRESET_EXAMPLES}"
        ),
        max_pages: Optional[int] = Form(None),
        dpi: int = Form(200),
        format: Optional[str] = Form(None),
    ) -> Any:
        """识别上传的图片或 PDF 中的文字。

        ``strategy`` / ``model`` 等为 JSON 字符串形式的临时配置，优先级高于
        ``/config`` 的运行时覆盖。``strategy_name`` 按命名预设整体切换引擎链
        （``local`` 仅本地 / ``vl`` 仅 VL / ``seq*`` 顺序回退 /
        ``bestof*`` 多引擎择优 / ``fallback`` 回退链 / ``quality`` 回退链+窜行降级+
        阅读顺序重排），同样只对本请求生效。
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

    @app.post("/ocr/text", tags=["OCR 识别"],
          summary="按图片 URL/base64/data URI 识别",
          description="JSON body:图片三选一 image_url / image_b64 / image_data(必须恰好传一个)。",
          responses=_ERROR_RESPONSES | {
              200: {"description": "识别成功,统一结构 {pages, text, engine, page_count}"}
          })
    async def ocr_text(body: TextRequest) -> Any:
        """识别图片中的文字。

        Body 三选一:
          - ``image_url`` —— 本地路径或 ``http(s)://`` URL
          - ``image_b64``  —— base64 编码的图片字节(自动加 data URI)
          - ``image_data`` —— data URI(如 ``data:image/png;base64,...``)
        """
        try:
            src = body.source()
        except HTTPException:
            raise
        effective = _apply_inline_overrides(state.snapshot(), body)
        results = _recognise(
            src,
            config=effective,
            engine_name=body.engine,
            max_pages=body.max_pages,
            dpi=body.dpi,
        )
        return _ocr_response(results, body.format.lower())

    # -- 便捷端点：路由即策略 -----------------------------------------------
    @app.post("/ocr/{preset}", tags=["OCR 识别"],
          summary="路由即策略:上传文件",
          description=(
              "preset 路径参数自动识别:匹配已注册引擎名(如 rapidocr/siliconflow)等价于"
              "强制单引擎;匹配策略预设名(如 local/vl/seq/bestof/fallback/quality)等价于"
              "strategy_name=preset;bestof:<mode> 冒号语法也支持。"
              "其他值返回 404 并列出全部可用选项。"
          ),
          responses=_ERROR_RESPONSES | {
              200: {"description": "识别成功,统一结构 {pages, text, engine, page_count}"}
          })
    async def ocr_preset_upload(
        preset: str,
        file: UploadFile = File(..., description="图片或 PDF"),
        model: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        max_pages: Optional[int] = Form(None),
        dpi: int = Form(200),
        format: Optional[str] = Form(None),
        retry_mode: Optional[str] = Form(None, description="覆盖 seq 家族 retry 判定(no_text/low_confidence/line_overlap/any/none)"),
        score_mode: Optional[str] = Form(None, description="覆盖 bestof 家族评分函数(smart/fastest/highest_confidence/longest/fluency)"),
        max_retries: Optional[int] = Form(None, ge=0, description="同引擎重试次数(0 等价于 cascade)"),
    ) -> Any:
        """路由即策略的专用接口：``POST /ocr/{preset}``。

        ``preset`` 路径参数自动识别:
          - 若匹配已注册引擎名(rapidocr / siliconflow / multimodal),等价于
            ``POST /ocr ... -F engine=preset``(强制单引擎);
          - 否则按策略预设名(local / vl / seq* / cascade* / bestof* / legacy 别名 /
            bestof:<mode>)处理,等价于 ``strategy_name=preset``。

        例::

            POST /ocr/rapidocr         # 只跑 rapidocr
            POST /ocr/siliconflow      # 只跑硅基流动
            POST /ocr/bestof           # 所有引擎择优
            POST /ocr/bestof-fluency   # 语义流畅度优先
            POST /ocr/cascade          # 首个不达标立即降级到下一引擎(不重试)
            POST /ocr/quality          # 窜行降级 + 阅读顺序重排
            POST /ocr/vl               # 仅远程大模型

        模型 / prompt / 格式 / 策略旋钮等仍可覆盖:
            POST /ocr/siliconflow ... -F "model=qwen-vl-max" -F "format=text"
            POST /ocr/seq     ... -F "retry_mode=line_overlap"
            POST /ocr/bestof  ... -F "score_mode=fastest"
            POST /ocr/seq     ... -F "max_retries=0"   # 等价于 cascade
        """
        out_format = (format or "json").lower()
        if out_format not in VALID_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown format '{format}'; use json, text or markdown",
            )
        norm = normalise_engine(preset)
        registered = describe_engines()
        lower = preset.lower()
        if norm in registered:
            body = TextRequest(image_url="", engine=preset)
        elif lower in STRATEGY_PRESETS or (lower.startswith("bestof:") and
              lower[len("bestof:"):].strip()):
            body = TextRequest(image_url="", strategy_name=preset)
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown preset '{preset}': choose an engine "
                    f"({', '.join(sorted(registered))}) or a strategy preset "
                    f"({', '.join(STRATEGY_PRESETS)}, or bestof:<mode>)"
                ),
            )
        if model:
            body.model = model
        if prompt:
            body.prompt = prompt
        if retry_mode:
            body.retry_mode = retry_mode
        if score_mode:
            body.score_mode = score_mode
        if max_retries is not None:
            body.max_retries = max_retries
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
                engine_name=body.engine,
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

    @app.post("/ocr/{preset}/text", tags=["OCR 识别"],
              summary="路由即策略:JSON body 识别",
              description=(
                  "与 /ocr/text 同义,body 三选一(image_url / image_b64 / image_data);"
                  "preset 路径参数自动识别:匹配引擎名等价于强制单引擎,匹配策略预设名"
                  "等价于 strategy_name=preset;bestof:<mode> 冒号语法也支持。"
                  "其他值返回 404 并列出全部可用选项。\n\n"
                  "示例(用 image_url 传网络图片):\n"
                  "```json\n"
                  "{ \"image_url\": \"https://example.com/scan.png\", \"format\": \"json\" }\n"
                  "```\n"
                  "示例(用 image_b64 传 base64 图片):\n"
                  "```json\n"
                  "{ \"image_b64\": \"iVBORw0KGgoAAAANSUhEUg...\" }\n"
                  "```"
              ),
              responses=_ERROR_RESPONSES | {
                  200: {"description": "识别成功,统一结构 {pages, text, engine, page_count}"}
              },
              openapi_extra={
                  "requestBody": {
                      "required": True,
                      "content": {
                          "application/json": {
                              "schema": {"$ref": "#/components/schemas/TextRequest"},
                              "examples": {
                                  "image_url": {
                                      "summary": "按图片 URL 识别",
                                      "value": {
                                          "image_url": "https://example.com/scan.png",
                                          "format": "json",
                                      },
                                  },
                                  "image_b64": {
                                      "summary": "按 base64 图片识别",
                                      "value": {
                                          "image_b64": "iVBORw0KGgoAAAANSUhEUg...",
                                          "format": "json",
                                      },
                                  },
                                  "image_data": {
                                      "summary": "按 data URI 识别",
                                      "value": {
                                          "image_data": "data:image/png;base64,iVBORw0KGgo...",
                                          "format": "json",
                                      },
                                  },
                                  "with_score_mode": {
                                      "summary": "bestof 家族切换评分函数",
                                      "value": {
                                          "image_url": "https://example.com/scan.png",
                                          "score_mode": "fastest",
                                      },
                                  },
                                  "with_retry_mode": {
                                      "summary": "seq 家族切换 retry 判定",
                                      "value": {
                                          "image_url": "https://example.com/scan.png",
                                          "retry_mode": "line_overlap",
                                      },
                                  },
                              },
                          }
                      },
                  }
              })
    async def ocr_preset_url(preset: str, body: TextRequest) -> Any:
        """路由即策略的 JSON 接口:``POST /ocr/{preset}/text``。

        与 :func:`ocr_preset_upload` 同义;body 三选一(image_url / image_b64 /
        image_data),与 :func:`ocr_text` 一致。
        """
        try:
            src = body.source()
        except HTTPException:
            raise
        norm = normalise_engine(preset)
        registered = describe_engines()
        lower = preset.lower()
        if norm in registered:
            body.engine = preset
        elif lower in STRATEGY_PRESETS or (lower.startswith("bestof:") and
              lower[len("bestof:"):].strip()):
            body.strategy_name = preset
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown preset '{preset}': choose an engine "
                    f"({', '.join(sorted(registered))}) or a strategy preset "
                    f"({', '.join(STRATEGY_PRESETS)})"
                ),
            )
        effective = _apply_inline_overrides(state.snapshot(), body)
        results = _recognise(
            src,
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
