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
``OPENAI_API_KEY``) or ``config/config.yaml``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
import urllib.parse
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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

#: Swagger UI's "Generate cURL" substitutes the field's schema type as the
#: literal value into *every* optional parameter. None of those are real
#: overrides, but applying them as if they were is what makes an empty-body
#: request blow up: ``retry_mode=string`` fails validation, ``base_url=string``
#: silently points the request at ``string`` as a provider endpoint. Treat them
#: as "the caller did not provide this" instead.
_OPENAPI_PLACEHOLDERS = frozenset(
    {"string", "number", "integer", "boolean", "array", "object"}
)


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalise an optional override to ``None`` when the caller did not really send one.

    Both blank strings and schema-type placeholders collapse to ``None`` — the
    downstream checks are all truthiness tests, so a cleared field behaves
    exactly like one that was never sent (fall back to the config entry's own
    value and its env-var chain).
    """
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() in _OPENAPI_PLACEHOLDERS:
        return None
    return text


def _clean_format(value: Optional[str]) -> str:
    """Resolve a ``format`` value to a valid format name.

    ``format`` is the one knob whose default is not "absent": it is a Form / body
    field with default ``"json"``. A Swagger placeholder or blank string must
    therefore collapse to that default rather than to ``None`` — a ``None``
    would crash the ``.lower()`` the response helper does with it.
    """
    cleaned = _clean(value)
    return (cleaned or "json").lower()


def _resolve_max_retries(value: Optional[str]) -> Optional[int]:
    """Parse a ``max_retries`` form value.

    ``max_retries`` is the only strategy knob typed ``int`` in the body, so the
    JSON body path already rejects a Swagger placeholder at request parsing —
    but multipart takes it as a string and a blank ``max_retries=`` parses as
    ``None`` and would quietly mean "keep the config default" instead of
    failing loudly. Coerce here so both input paths behave identically;
    callers get ``None`` back for "not sent" and a 400 for garbage.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value >= 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


# OpenAPI 响应组件:统一的错误响应结构
_ERROR_RESPONSES = {
    400: {"description": "坏输入(文件不存在/参数非法/strategy JSON 非对象)"},
    404: {"description": "未知 preset:路由既非已注册引擎也非策略预设"},
    422: {"description": "引擎不可用(缺依赖/key)或策略链耗尽"},
    502: {"description": "引擎调用失败(超时/HTTP 402/网络错误)"},
}

# 统一 200 结构说明。`decision` 说明策略为什么选中了这个引擎(seq 家族给
# accepted/rejected, bestof 家族给 ranked),`score` / `score_mode` 是给
# `decision` 配套的数值与口径;`format=text|markdown` 走纯文本,不含这三项。
_OCR_OK_RESPONSE = {
    200: {
        "description": (
            "识别成功,统一结构 {pages, text, engine, page_count, "
            "score, score_mode, decision}"
        )
    }
}

_PRESET_EXAMPLES = (
    "local, vl, seq, seq-any, seq-low_conf, seq-line_overlap, "
    "cascade, cascade-low_conf, cascade-line_overlap, "
    "bestof, bestof-smart, bestof-fastest, bestof-confidence, "
    "bestof-longest, bestof-fluency, fallback, quality, bestof:<mode>"
)

#: Preset spellings for the ``/ocr/{preset}`` path. The ``_`` prefix is how a
#: preset distinguishes itself from an engine that happens to share a name
#: (``vl`` is a preset, but also a plausible custom engine name).
_PRESET_ROUTE_EXAMPLES = (
    "_local, _vl, _seq, _seq-any, _seq-low_conf, _seq-line_overlap, "
    "_cascade, _cascade-low_conf, _cascade-line_overlap, "
    "_bestof, _bestof-smart, _bestof-fastest, _bestof-confidence, "
    "_bestof-longest, _bestof-fluency, _fallback, _quality"
)

#: ``/ocr/{preset}`` 与 ``/ocr/{preset}/text`` 共用的 body 字段里,「这个字段
#: 在哪种引擎范围下没有意义」。key 是字段名,值是失活的 ``engine_scope`` 取值。
#:
#: 这几个字段只对远程引擎生效——本地引擎(rapidocr)不读 URL / 模型 / key /
#: prompt。所以选到 ``local`` 预设时它们必然被忽略,把它们标成 readOnly,
#: Swagger UI 就渲染成不可输入的置灰框,而不是让用户白填一遍。
_DOCS_DISABLED_FIELDS = {
    "model": ("local_only",),
    "base_url": ("local_only",),
    "api_key": ("local_only",),
    "prompt": ("local_only",),
}

#: 附加到被置灰字段 description 末尾的一句提示,说明为什么这里填了也没用。
_DOCS_DISABLED_NOTE = (
    "\n\n当前预设只跑本地引擎,这个字段只对远程引擎生效,填了也不会生效——已置为只读。"
    "需要覆盖远程引擎参数请换用 `_vl` 或其他包含远程引擎的预设。"
)

#: ``preset`` 路径参数的说明。前半段讲清路径参数怎么用,后半段在
#: ``_docs_apply_preset_to_spec`` 里按当前预设补齐实际取值。
_DOCS_PRESET_PARAM_LEAD = (
    "策略由路径决定。`_` 前缀是「这是预设,不是引擎」的标记——`vl` 既是预设名也是"
    "可能的自定义引擎名,所以预设用 `_vl` 表示;裸预设名也能用,只要它同时不是"
    "已注册的引擎名。引擎名直接放路径(如 `/ocr/rapidocr`)等价于强制单引擎。"
    "本路由不接收 retry_mode / score_mode / max_retries。"
)

#: ``/ocr/{preset}`` 的路径参数名。
_DOCS_PRESET_PARAM = "preset"

#: Swagger UI 的 CDN 资源。**刻意不跟着 FastAPI 的 ``swagger-ui-dist@5`` 浮动 tag**——
#: 那个 tag 会悄悄指向新小版本,而新版的 action 名可能改,内嵌脚本就会静默失效。
#: 升级前先在本地起服务、用浏览器确认 ``/docs`` 的置灰仍然生效。
_DOCS_SWAGGER_BUNDLE_URL = (
    "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.32.2/swagger-ui-bundle.js"
)
_DOCS_SWAGGER_CSS_URL = (
    "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.32.2/swagger-ui.css"
)


def _docs_preset_meta(preset: Optional[str]) -> Optional[Dict[str, Any]]:
    """把路径参数值归一成预设元数据;空值 / 未知预设返回 ``None``。

    路径参数接受 ``_`` 前缀(``_local``),但预设表用裸名(``local``)做 key——这里统一
    剥掉前缀。未知预设本就会在运行时 404,所以文档页也不该显示成可正常填写。
    """
    if not preset:
        return None
    return describe_presets().get(preset.lstrip("_"))


def _docs_preset_value(preset: str) -> str:
    """剥掉可选的 ``_`` 前缀,``_local`` 与 ``local`` 等价。"""
    return (preset or "").strip().lstrip("_")


#: 内嵌进 ``/docs`` 的联动脚本:把 Swagger UI 的 spec 拉取换成
#: ``/openapi.json?preset=<preset 输入框里的值>``。
#:
#: 已实测的两个 Swagger UI 行为决定了实现方式:
#:
#: - 带 ``readOnly: true`` 的属性会被**从请求表单里整个不渲染**(不是变灰)——
#:   ``/ocr/{preset}`` 在 ``?preset=_local`` 下只剩 file / max_pages / dpi / format,
#:   所以「不出现输入框」就是「不可输入」,不用改任何渲染逻辑。
#: - ``tr[data-param-name=…][data-param-in="path"]`` 是 UI 给路径参数行加的属性,
#:   没有这个属性的端点(如 ``/ocr``)天然不会被命中。
#:
#: 应用新 spec 用 ``specActions.updateSpec()`` 而不是 ``location.reload()``:
#: 它原地重渲染,保留已展开的 opblock、滚动位置和「Try it out」状态——用户只想要
#: body 字段置灰,不该连页面一起刷新丢光。地址栏的 ``?preset=`` 同步更新,
#: URL 本身就是深链接,刷新 / 分享同一个地址就能还原。
#:
#: UI 实例从 ``window.__jykjOcrUI`` 取(由 ``_docs_swagger_ui_html`` 在 FastAPI 生成的
#: ``const ui = SwaggerUIBundle(...)`` 之后注入一行挂上去——``const`` 是词法声明,
#: 拿不到 ``window.swaggerUI``)。没有实例时降级到整页 reload。
#:
#: 打字不被打断:焦点在 ``/ocr`` 的 body 字段里时,重建 DOM 会冲掉光标和未提交的
#: 内容,所以让路;焦点在 ``preset`` 框里时仍要更新(这就是用户要的「边填边灰」)。
_DOCS_SWAGGER_PRESET_JS = r"""
<script>
(function () {
  var PARAM = 'preset',
      SPEC = '/openapi.json',
      POLL_MS = 120,
      isFn = function (v) { return typeof v === 'function'; },
      ui = window.__jykjOcrUI,
      // 起始值取自 URL 本身——这样首次加载不会触发重渲染(值没变)。
      // 不能从 null 起:每次刷新后 null 都会再触发一次,变成死循环。
      current = preset();

  function preset() {
    var m = document.location.search.match(new RegExp('[?&]' + PARAM + '=([^&]*)'));
    return m ? decodeURIComponent(m[1]).trim().replace(/^_/, '') : '';
  }

  // 用户此刻正在打字的那个框(可能是 preset 框,也可能是 body 里的字段)。
  function focused() {
    var el = document.activeElement;
    if (!el || el.tagName !== 'INPUT') return null;
    return el;
  }

  // 这个框是不是 `/ocr/{preset}` 的路径参数输入框。
  function isPresetInput(el) {
    var tr = el.closest ? el.closest('tr') : null;
    return !!tr && tr.getAttribute('data-param-in') === 'path';
  }

  function pushUrl(v) {
    var here = document.location.href.split('#')[0].split('?')[0];
    var next = here + '?' + PARAM + '=' + encodeURIComponent(v) + document.location.hash;
    if (next !== document.location.href) {
      try { history.replaceState(null, '', next); } catch (e) {}
    }
  }

  function apply(spec) {
    // 原地重渲染:保留已展开的 opblock、滚动位置、「Try it out」状态。
    if (isFn(ui.specActions.updateSpec)) {
      try { ui.specActions.updateSpec(JSON.stringify(spec)); return; } catch (e) {}
    }
    // 降级:整页刷新(会丢展开状态,但总能生效)。
    try { window.location.reload(); } catch (e) {
      console.warn('[jykj_ocr /docs] 无法应用新 spec:', e);
    }
  }

  // 用户直接在 preset 框里敲字:更新地址栏,真正的重渲染交给轮询。
  document.addEventListener('input', function (e) {
    var t = e.target;
    if (t.tagName !== 'INPUT' || t.type !== 'text') return;
    if (!isPresetInput(t)) return;
    pushUrl(t.value.trim().replace(/^_/, ''));
  }, true);

  function loop() {
    var next = preset();
    if (next === current) { setTimeout(loop, POLL_MS); return; }
    // 用户正在打字的不是 preset 框(在填请求参数)→ 重建 DOM 会冲掉光标,让路。
    // preset 框自己打字时仍要更新:pushUrl 已经把地址栏推进,轮询负责把 spec 跟上。
    var f = focused();
    if (f && !isPresetInput(f)) { setTimeout(loop, POLL_MS); return; }
    setTimeout(function () {
      var fresh = preset();
      if (fresh === current) return;
      var f2 = focused();
      if (f2 && !isPresetInput(f2)) return;   // 期间焦点跳到了别的框
      current = fresh;
      fetch(fresh ? (SPEC + '?' + PARAM + '=' + encodeURIComponent(fresh)) : SPEC,
          {cache: 'no-store'}).then(function (r) {
        if (!r.ok) throw new Error('status ' + r.status);
        return r.json();
      }).then(function (spec) { apply(spec); })
        .catch(function (e) { console.warn('[jykj_ocr /docs] 拉取 spec 失败:', e); });
    }, 0);
    setTimeout(loop, POLL_MS);
  }

  loop();
})();
</script>
"""


def _docs_swagger_ui_html(spec_url: str) -> HTMLResponse:
    """Swagger UI 页面,并在 ``</body>`` 前追加 preset 联动脚本。

    复用 FastAPI 的 HTML 模板,只把 CDN 版本钉死,并在 ``SwaggerUIBundle(...)``
    之后注入一行 ``window.__jykjOcrUI = ui;``——脚本要调 ``specActions.updateSpec``
    原地重渲染,而 FastAPI 生成的是 ``const ui``(词法声明),拿不到实例。
    """
    from fastapi.openapi.docs import get_swagger_ui_html

    base = get_swagger_ui_html(
        openapi_url=spec_url,
        title="jykj_ocr - API Docs",
        oauth2_redirect_url="/docs/oauth2-redirect",
        swagger_js_url=_DOCS_SWAGGER_BUNDLE_URL,
        swagger_css_url=_DOCS_SWAGGER_CSS_URL,
    )
    html = base.body.decode("utf-8")
    marker = "})\n    </script>"
    if marker in html:
        html = html.replace(marker, "})\nwindow.__jykjOcrUI = ui;\n    </script>", 1)
    if "</body>" in html:
        html = html.replace("</body>", _DOCS_SWAGGER_PRESET_JS + "\n</body>", 1)
    return HTMLResponse(html)

def _docs_preset_disabled(meta: Dict[str, Any], field: str) -> bool:
    """字段 ``field`` 在当前预设 ``meta`` 下是否失效(应置灰)。"""
    return meta.get("engine_scope") in _DOCS_DISABLED_FIELDS.get(field, ())


def _docs_preset_param_desc(preset: str, meta: Dict[str, Any]) -> str:
    """``preset`` 路径参数的说明:该预设实际生效的取值。"""
    scope_note = {
        "local_only": "仅本地引擎(远程全部禁用)",
        "remote_vl_only": "仅 config 里第一条已启用的远程引擎,其余禁用",
    }.get(meta.get("engine_scope"), "全部已启用引擎")
    retry = (
        f"`retry_mode` = `{meta['retry_mode']}`"
        if meta.get("retry_mode") else "`retry_mode` 不适用(bestof 家族无重试谓词)"
    )
    score = (
        f"`score_mode` = `{meta['score_mode']}`"
        if meta.get("score_mode") else "`score_mode` 不适用(seq / cascade 家族)"
    )
    retries = (
        f" = `{meta['max_retries']}`"
        if meta.get("max_retries") is not None else "沿用配置文件里的值"
    )
    reorder = "按坐标重建阅读顺序" if meta.get("reorder_lines") else "不按坐标重排"
    return (
        f"当前选中 **{preset}** — {scope_note}。{retry};{score};max_retries"
        f"{retries};{reorder}。"
    )


def _docs_apply_preset_to_spec(spec: Dict[str, Any], preset: Optional[str]) -> Dict[str, Any]:
    """``/docs?preset=...`` 用:把当前预设下无效的 body 字段标成 ``readOnly``。

    深拷贝输入,不原地修改。preset 为空或不是已知预设时原样返回——未知预设
    本就会在运行时 404,文档页不该显得能正常填表单。

    只改 ``/ocr/{preset}`` 与 ``/ocr/{preset}/text`` 两个端点:它们的 body
    schema 挂在各自 operation 的 ``requestBody`` 里(multipart 是内联对象,
    JSON 是 ``$ref``),而不是 ``components``,所以直接就地改。
    """
    if not preset:
        return spec
    meta = _docs_preset_meta(preset)
    if meta is None:
        return spec

    out = copy.deepcopy(spec)
    schemas = out["components"]["schemas"]
    for path in ("/ocr/{preset}", "/ocr/{preset}/text"):
        op = out["paths"].get(path, {}).get("post")
        if not op:
            continue
        for media_type in op.get("requestBody", {}).get("content", {}):
            schema = _docs_mark_body_fields(
                op["requestBody"]["content"][media_type]["schema"],
                meta,
                schemas,
                _docs_route_slug(path),
            )
            op["requestBody"]["content"][media_type]["schema"] = schema
        for param in op.get("parameters", []):
            if param.get("in") == "path" and param.get("name") == "preset":
                param["description"] = (
                    _DOCS_PRESET_PARAM_LEAD + "\n" + _docs_preset_param_desc(preset, meta)
                )
    return out


def _docs_route_slug(path: str) -> str:
    """把 ``/ocr/{preset}/text`` 变成 ``ocr_preset_text``,用作 schema 克隆的名字。"""
    return path.strip("/").replace("/", "_").replace("{", "").replace("}", "").replace("-", "_")


def _docs_mark_body_fields(
    schema: Dict[str, Any],
    meta: Dict[str, Any],
    schemas: Dict[str, Any],
    slug: str,
) -> Dict[str, Any]:
    """就地标 readOnly;若 schema 是被多个端点共享的,先克隆一份再改。

    克隆的必要性:`TextRequest` 同时被 `/ocr/text` 和 `/ocr/{preset}/text` 引用,
    直接改 components 里的原 schema 会把 `/ocr/text` 一起置灰——那个端点没选预设,
    不该被置灰。克隆成 `TextRequest_<slug>` 只影响当前 operation。
    """
    ref = schema.get("$ref")
    if ref:
        name = ref.split("/")[-1]
        clone_name = f"{name}_{slug}"
        clone = copy.deepcopy(schemas.get(name, {}))
        _docs_mark_body_fields(clone, meta, schemas, slug)
        schemas[clone_name] = clone
        return {"$ref": f"#/components/schemas/{clone_name}"}
    for field, field_schema in schema.get("properties", {}).items():
        if not _docs_preset_disabled(meta, field):
            continue
        field_schema["readOnly"] = True
        field_schema["description"] = field_schema.get("description", "") + _DOCS_DISABLED_NOTE
    return schema


def _engine_raw(engine: EngineConfig) -> Dict[str, Any]:
    """引擎配置的原始 dict 表示,回喂给 ``from_mapping`` 用。"""
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
        "instance": engine.instance,
    }


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
    "instance",
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
                        "instance": e.instance,
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


def _is_remote(engine: EngineConfig) -> bool:
    """True when the entry talks to an HTTP endpoint (reads URL / model / key)."""
    return normalise_engine(engine.name) in remote_engines()


def _engine_view(engine: EngineConfig) -> Dict[str, Any]:
    """Config view for clients — never echo API keys.

    Local engines (rapidocr) ignore ``base_url`` / ``model`` / ``api_key``
    entirely, so echoing the *resolved* values would be misleading: those
    properties still walk the shared ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``
    fallback chain, meaning a purely offline entry would advertise a remote
    endpoint and ``has_api_key: true`` it never uses. Blank those three for
    local entries; the fields stay present so clients keep their shape.
    """
    if _is_remote(engine):
        model, base_url, has_key = (
            engine.resolved_model,
            engine.resolved_base_url,
            bool(engine.resolved_api_key),
        )
    else:
        model, base_url, has_key = "", "", False
    return {
        "name": engine.name,
        "enabled": engine.enabled,
        "model": model,
        "base_url": base_url,
        "timeout": engine.timeout,
        "temperature": engine.temperature,
        "max_tokens": engine.max_tokens,
        "lang": engine.lang,
        "has_api_key": has_key,
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
    engine: Optional[str] = Field(None, description="强制单引擎(如 rapidocr / multimodal),覆盖 strategy")
    max_pages: Optional[int] = Field(None, description="PDF 页数上限")
    dpi: int = Field(200, description="PDF 渲染 DPI")
    format: str = Field("json", description="输出格式:json / text / markdown")
    model: Optional[str] = Field(None, description="覆盖远程引擎的 model 名")
    prompt: Optional[str] = Field(None, description="覆盖远程引擎的 prompt")
    base_url: Optional[str] = Field(
        None,
        description=(
            "覆盖远程引擎的 base_url(仅本请求生效)。留空或省略时回退到配置条目, "
            "再回退到 OPENAI_BASE_URL_&lt;N&gt; → OPENAI_BASE_URL → "
            "JYKJ_OCR_&lt;NAME&gt;_BASE_URL。与 model / api_key 一起改即可在单次"
            "请求内指向另一个平台。"
        ),
    )
    api_key: Optional[str] = Field(
        None,
        description=(
            "覆盖远程引擎的 api_key(仅本请求生效,不落盘)。留空或省略时按原顺序"
            "回退:配置条目 → JYKJ_OCR_MULTIMODAL_&lt;N&gt;_API_KEY → "
            "MULTIMODAL_&lt;N&gt;_API_KEY → JYKJ_OCR_MULTIMODAL_API_KEY → "
            "MULTIMODAL_API_KEY → OPENAI_API_KEY。与 base_url / model 组合即可在"
            "单次请求内切到另一个平台或账号。"
        ),
    )
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


def _resolve_preset_route(preset: str, body: TextRequest) -> None:
    """Map the ``/ocr/{preset}`` path segment onto ``body``.

    A registered engine name wins when it matches (``rapidocr`` /
    ``multimodal``) — that is how a bare engine spelling stays "force this
    engine". Otherwise the value is a strategy preset, with or without the
    ``_`` prefix (``_seq`` and ``seq`` both mean the ``seq`` preset).

    The ``_`` prefix is what the docs page uses as an unambiguous "this is a
    preset" marker: ``vl`` is a preset, but could also be somebody's custom
    engine name, and ``_vl`` can never be an engine. Either spelling works at
    runtime; the docs page greys out the fields a preset does not take when it
    sees the marker.

    Unknown values get a 404 listing both families, so the caller can copy a
    valid spelling.
    """
    key = (preset or "").strip()
    if normalise_engine(key) in describe_engines():
        body.engine = key
        return
    name = key[1:] if key.startswith("_") else key
    if not name or (name.startswith("bestof:") and not name[len("bestof:"):].strip()):
        raise HTTPException(
            status_code=404,
            detail=_unknown_preset_detail(preset),
        )
    body.strategy_name = name


def _unknown_preset_detail(preset: str) -> str:
    """The 404 body for a path segment that is neither an engine nor a preset."""
    return (
        f"unknown preset '{preset}': choose an engine "
        f"({', '.join(sorted(describe_engines()))}) or a preset route "
        f"({_PRESET_ROUTE_EXAMPLES}, or _bestof:<mode>); a bare preset name "
        f"works too, as long as it is not also a registered engine name"
    )


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
        "instance": engine.instance,
    }


def _apply_inline_overrides(config: Config, body: TextRequest) -> Config:
    """Apply per-request ``engine``/``model``/``base_url``/``api_key``/``prompt``/
    ``strategy``/``strategy_name``.

    This lets a caller pick a different model, provider endpoint, account key, or a
    named preset (``local``/``vl``/``seq*``/``bestof*``/``fallback``/``quality``)
    for a single request without a global ``POST /config``, e.g.
    ``{"image_url": "...", "strategy_name": "bestof"}`` or
    ``{"image_url": "...", "api_key": "sk-..."}``.

    Returns a new :class:`Config`; the input is never mutated (one-shot).
    """
    engine_dicts = [_engine_raw(e) for e in config.engines]
    strategy = dict(config.strategy)
    output = dict(config.output)
    changed = False

    # Swagger's generated curl fills every optional field with the schema type
    # ("string", "integer", ...). Nothing downstream can tell those from real
    # values, so normalise them away here — one place covers all four OCR
    # endpoints. ``max_retries`` is int-typed, so a placeholder there already
    # fails at request parsing (422) and never reaches this function.
    for field in ("engine", "model", "base_url", "api_key", "prompt",
                  "strategy_name", "retry_mode", "score_mode"):
        setattr(body, field, _clean(getattr(body, field)))
    # ``max_retries`` is the only knob the multipart endpoints take as a string
    # (the JSON body types it int), so coerce it here before anything reads it.
    # A blank or placeholder value means "not sent"; anything else that does
    # not parse as a non-negative int is a genuine typo and gets a 400 rather
    # than quietly becoming "keep the config default".
    retries = _resolve_max_retries(body.max_retries)
    if retries is None and _clean(body.max_retries) is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "max_retries must be an integer >= 0; "
                "omit the field to keep the config default"
            ),
        )
    body.max_retries = retries
    body.format = _clean_format(body.format)

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

    # ``api_key`` / ``base_url`` point a single request at another provider or
    # account without a global ``POST /config``. A falsy value is left alone so
    # the entry's own value — or its env-var fallback chain in
    # ``resolved_api_key`` / ``resolved_base_url`` — keeps working (only
    # ``null`` on POST /config reverts).
    if body.api_key:
        for item in engine_dicts:
            if normalise_engine(item["name"]) in remote_engines():
                item["api_key"] = body.api_key
        changed = True

    if body.base_url:
        for item in engine_dicts:
            if normalise_engine(item["name"]) in remote_engines():
                item["base_url"] = body.base_url
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
            "highest_confidence/longest/fluency)、`max_retries`(0 等价于 cascade)。"
            "只适用于通用端点 `POST /ocr` / `POST /ocr/text`;`/ocr/{preset}` 的路由"
            "本身就是策略,这三个旋钮不在其参数列表里。\n\n"
            "## `/ocr/{preset}` 的路径就是策略\n\n"
            "`preset` 填 **`_` 前缀的预设名**(下划线就是「这是预设,不是引擎」的标记)"
            "即走预设:填 `_bestof-fluency` 后,`score_mode` / `retry_mode` / "
            "`max_retries` 三个输入框会被置灰——它们对该预设没有意义。预设与引擎"
            "共用同一条路径,所以预设名需要加前缀,否则 `vl` 既可能是预设也可能是"
            "某个自定义引擎名,无法区分。裸预设名仍可用:走 `strategy_name` 参数或"
            "`GET /presets` 列出的名字,`engine=seq` 同理。\n\n"
            "## 图片来源\n\n"
            "图片/PDF 文件(multipart)、`http(s)://` URL、本地路径、纯 base64、"
            "完整 data URI 均支持。\n\n"
            "## 运行时配置\n\n"
            "`POST /config` 可热改模型/引擎/策略,无需重启,不回显 API key 明文。"
        ),
        docs_url=None,
        redoc_url="/redoc",
        openapi_url=None,
        tags=[
            {
                "name": "OCR 识别",
                "description": (
                    "图片/PDF 文字识别。四种 OCR 端点返回结构完全一致:"
                    "{pages, text, engine, page_count, score, score_mode, decision};"
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
        # One pipeline serves every page, so the decision trace is the same
        # for each — surface it once at the top level and keep the per-page
        # copy inside `pages[]` for clients that read pages independently.
        first = results[0] if results else None
        return {
            "pages": [r.as_dict() for r in results],
            "text": joined,
            "engine": first.engine if first else "",
            "page_count": len(results),
            "score": first.score if first else None,
            "score_mode": first.score_mode if first else "",
            "decision": first.decision if first else None,
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
                 "指纹),便于在多实例部署下区分同名引擎。本地引擎(rapidocr)不读 "
                 "URL / 模型 / key,所以它的 model 与 base_url 恒为空串——只有真正 "
                 "走 HTTP 的条目才有端点指纹。"
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
                    "model": e.resolved_model if _is_remote(e) else "",
                    "base_url": e.resolved_base_url if _is_remote(e) else "",
                }
                for e in config.engines
            ],
        }

    # -- 文档页:与 `/ocr/{preset}` 联动 -----------------------------------------
    # Swagger UI 的 ``/docs`` 只渲染一个静态 spec。这里换成「路径输入框里的值 →
    # 重载带 readOnly 标记的 spec」:字段被标成 readOnly 后,Swagger UI 直接不渲染
    # 它的输入框,等价于「不可填」——无需重写任何渲染逻辑。
    @app.get("/docs", include_in_schema=False, tags=["文档"])
    async def _docs_page(preset: Optional[str] = None):
        """Swagger UI。可选 ``?preset=_bestof`` 直接跳到该预设对应的 spec。"""
        preset_value = _docs_preset_value(preset)
        spec_url = (
            "/openapi.json"
            if not preset_value
            else "/openapi.json?" + urllib.parse.urlencode({"preset": preset_value})
        )
        return _docs_swagger_ui_html(spec_url)

    @app.get("/redoc", include_in_schema=False, tags=["文档"])
    async def _redoc_page():
        from fastapi.openapi.docs import get_redoc_html

        return get_redoc_html(openapi_url="/openapi.json", title="jykj_ocr - Redoc")

    @app.get("/docs/oauth2-redirect", include_in_schema=False, tags=["文档"])
    async def _oauth_redirect():
        return JSONResponse({"ok": True})

    @app.get("/openapi.json", include_in_schema=False, tags=["文档"])
    async def _openapi_spec(preset: Optional[str] = None):
        """OpenAPI spec;``?preset=`` 时把该预设不接收的字段标成 ``readOnly``。

        不带参数时与 FastAPI 默认的 ``/openapi.json`` 完全一致,供 curl / 客户端
        工具消费。带参数时只改 ``/ocr/{preset}`` 与 ``/ocr/{preset}/text`` 的
        body 字段和 ``preset`` 路径参数说明,其余端点原样返回。
        """
        return _docs_apply_preset_to_spec(app.openapi(), preset)

    @app.get("/presets", tags=["配置与状态"],
             summary="可用策略预设",
             description=(
                 "列出全部命名策略预设(含评分模式、重试判定、引擎范围),与 CLI "
                 "`--strategy-name`、HTTP `strategy_name` 参数、`/ocr/{preset}` 路由"
                 "一一对应。17 个显式预设 + `bestof:<mode>` 冒号语法别名,共 18 项。"
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
                  "\n\n示例: {\"engines\": [{\"name\": \"multimodal\", \"model\": \"qwen-vl-max\"}]} 或 {\"strategy\": {\"max_retries\": 2}}"
              ),
              responses={400: {"description": "不支持的配置键或值非法"}},
              openapi_extra={
                  "examples": {
                      "切换模型": {
                          "summary": "切换 multimodal 模型",
                          "value": {"engines": [{"name": "multimodal", "model": "qwen-vl-max"}]},
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
            {"engines": [{"name": "multimodal", "model": "qwen-vl-max"}]}

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
              responses=_ERROR_RESPONSES | _OCR_OK_RESPONSE)
    async def ocr_upload(
        file: UploadFile = File(..., description="图片或 PDF"),
        engine: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        base_url: Optional[str] = Form(None),
        api_key: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        strategy: Optional[str] = Form(None),
        strategy_name: Optional[str] = Form(
            None, description=f"一次性策略预设：{_PRESET_EXAMPLES}"
        ),
        retry_mode: Optional[str] = Form(None),
        score_mode: Optional[str] = Form(None),
        max_retries: Optional[str] = Form(None),
        max_pages: Optional[int] = Form(None),
        dpi: int = Form(200),
        format: Optional[str] = Form(None),
    ) -> Any:
        """识别上传的图片或 PDF 中的文字。

        ``strategy`` / ``model`` 等为 JSON 字符串形式的临时配置，优先级高于
        ``/config`` 的运行时覆盖。``strategy_name`` 按命名预设整体切换引擎链
        （``local`` 仅本地 / ``vl`` 仅 1 条远程 VL / ``seq*`` 顺序回退 /
        ``bestof*`` 多引擎择优 / ``fallback`` 回退链 / ``quality`` 回退链+窜行降级+
        阅读顺序重排），同样只对本请求生效。
        """
        out_format = _clean_format(format)
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
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            strategy=strategy_map,
            strategy_name=strategy_name,
            retry_mode=retry_mode,
            score_mode=score_mode,
            max_retries=max_retries,
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
          description=(
              "JSON body:图片三选一 image_url / image_b64 / image_data(必须恰好传一个)。"
              "可选覆盖:model / base_url / api_key / prompt / strategy / strategy_name "
              "及策略旋钮(retry_mode / score_mode / max_retries),均只对本请求生效;"
              "api_key 与 base_url 省略时用 export 的环境变量或配置文件里的值。"
          ),
          responses=_ERROR_RESPONSES | _OCR_OK_RESPONSE)
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
              "preset 路径参数自动识别,两种取值:\n\n"
              "* 已注册引擎名(`rapidocr` / `multimodal` 等)——等价于"
              " `POST /ocr ... -F engine=preset`,强制单引擎;\n"
              "* **`_` 前缀的预设名**(`_seq` / `_vl` / `_bestof-fluency` …)——"
              "等价于 `strategy_name=预设名`。`_` 是「这是预设,不是引擎」的标记:"
              "预设与引擎共用同一条路径,`vl` 既可能是预设也可能是某个自定义引擎名,"
              "所以预设名需要前缀才能区分。裸预设名走 `strategy_name` 参数或用"
              " `GET /presets` 查看完整清单。\n\n"
              "其他值返回 404 并列出全部可用选项。"
          ),
          responses=_ERROR_RESPONSES | _OCR_OK_RESPONSE)
    async def ocr_preset_upload(
        preset: str,
        file: UploadFile = File(..., description="图片或 PDF"),
        model: Optional[str] = Form(None),
        base_url: Optional[str] = Form(None),
        api_key: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        max_pages: Optional[int] = Form(None),
        dpi: int = Form(200),
        format: Optional[str] = Form(None),
    ) -> Any:
        """路由即策略的专用接口：``POST /ocr/{preset}``。

        ``preset`` 路径参数自动识别:
          - 若匹配已注册引擎名(rapidocr / multimodal),等价于
            ``POST /ocr ... -F engine=preset``(强制单引擎);
          - 若带 ``_`` 前缀,去掉前缀后按策略预设名处理,等价于
            ``strategy_name=<preset>``(``_seq`` → ``seq``,``_bestof:fastest``
            → ``bestof:fastest``,冒号语法也支持)。

        例::

            POST /ocr/rapidocr            # 只跑 rapidocr
            POST /ocr/multimodal          # 只跑远程多模态
            POST /ocr/_seq                # 顺序重试
            POST /ocr/_bestof-fluency     # 所有引擎择优,语义流畅度优先
            POST /ocr/_cascade            # 首个不达标立即降级到下一引擎(不重试)
            POST /ocr/_quality            # 窜行降级 + 阅读顺序重排
            POST /ocr/_vl                 # 仅 1 条远程大模型

        模型 / 平台 / 凭据 / prompt / 格式仍可覆盖;策略本身由路由固定,不再暴露
        retry_mode / score_mode / max_retries:
            POST /ocr/multimodal ... -F "model=qwen-vl-max" -F "format=text"
            POST /ocr/_vl ... -F "model=Qwen/Qwen3-VL-30B-A3B-Instruct" \\
                              -F "base_url=https://api.siliconflow.cn/v1" \\
                              -F "api_key=sk-..."   # 都省略时用 export 或配置里的值
        要换策略旋钮请用通用端点 ``POST /ocr`` 或 ``POST /ocr/text`` 传
        ``strategy_name`` + ``retry_mode`` / ``score_mode`` / ``max_retries``。
        """
        out_format = _clean_format(format)
        if out_format not in VALID_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown format '{format}'; use json, text or markdown",
            )
        body = TextRequest(image_url="")
        _resolve_preset_route(preset, body)
        if model:
            body.model = model
        if base_url:
            body.base_url = base_url
        if api_key:
            body.api_key = api_key
        if prompt:
            body.prompt = prompt
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
                  "preset 路径参数自动识别:匹配引擎名等价于强制单引擎,带 `_` 前缀的"
                  "预设名等价于 strategy_name=预设名(`_seq` → `seq`,冒号语法也支持)。"
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
              responses=_ERROR_RESPONSES | _OCR_OK_RESPONSE,
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
                                  "with_model_and_key": {
                                      "summary": "单次请求切平台/账号(省略 key 时用 export 的值)",
                                      "value": {
                                          "image_url": "https://example.com/scan.png",
                                          "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                                          "base_url": "https://api.siliconflow.cn/v1",
                                          "api_key": "sk-your-key-here",
                                          "prompt": "只输出图片中的文字",
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
        image_data),与 :func:`ocr_text` 一致。带 ``_`` 前缀的预设名等价于
        ``strategy_name=<preset>``。
        """
        try:
            src = body.source()
        except HTTPException:
            raise
        _resolve_preset_route(preset, body)
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
