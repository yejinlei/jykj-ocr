# -*- coding: utf-8 -*-
"""真实模型端到端测试:全部 HTTP 接口 + 全部策略预设。

必须用真实远程模型跑(不走 monkeypatch)。前置条件:
  1. 一个已启动的服务,例如
        OPENAI_API_KEY=*** OPENAI_BASE_URL=https://api.moark.com/v1 \
            .venv/Scripts/python -m jykj_ocr serve
  2. .env 或环境变量里至少有一个远程平台可用(通用 OPENAI_* 即可)。

覆盖:
  A. 接口 —— /health /engines /presets /config(GET/POST/DELETE)
            /ocr /ocr/text /ocr/{preset} /ocr/{preset}/text
  B. 策略 —— 全部命名预设(seq*/cascade*/bestof*/local/vl/fallback/quality)
            + bestof:<mode> 冒号别名 + 未知预设返回 400

结果写入 real_model_e2e_result.json(UTF-8,避免 Windows GBK 控制台编码错误)。
退出码 = 失败用例数。

运行:
    .venv/Scripts/python scripts/real_model_e2e.py [服务地址] [图片路径]
    例: .venv/Scripts/python scripts/real_model_e2e.py http://127.0.0.1:8000 tests/兰亭序.jpeg
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "real_model_e2e_result.json"

# 客户端与服务端都要看到同一个 key,才能派生泄露检测片段。服务自己在启动时读
# .env;这里同样补读一次,否则 key 只写进 .env 时 _leak_fragments() 返回空元组,
# 那条"不泄露 key"的用例会因检查根本没生效而失败(不是真的泄露)。
sys.path.insert(0, str((ROOT / "src").resolve()))
from jykj_ocr.cli import _load_dotenv  # noqa: E402

_load_dotenv(str(ROOT / ".env"))

ALL_PRESETS = [
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
    "bestof:smart",
    "fallback",
    "quality",
]

def _resolved_api_key() -> str:
    """按 config.py 的解析顺序取本机 API key,供泄露检测派生片段。

    真实 key 只存在于 .env(已 gitignore),不能硬编码进脚本——否则提交脚本
    就等于把 key 前缀写进版本库。解析顺序与 config.EngineConfig.resolved_api_key
    保持一致,确保本地与服务端看到的是同一个值。
    """
    return (
        os.getenv("JYKJ_OCR_MULTIMODAL_API_KEY")
        or os.getenv("MULTIMODAL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()


def _leak_fragments() -> tuple:
    """派生 key 片段:命中即判定 GET /config 泄露了明文。

    用 key 的前 8 个字符而非泛化正则——yaml 里 "sk-your-key-here" 这类占位
    示例不算泄露。片段长度低于任何真实 key,本身无法用于调用 API。
    """
    key = _resolved_api_key()
    return (key[:8],) if len(key) >= 8 else ()


def _has_key_flags(cfg: dict) -> dict:
    """收集配置里所有 has_api_key 布尔位,证明脱敏走的是元数据而非明文。"""
    flags = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                path = f"{prefix}.{k}" if prefix else k
                if k == "has_api_key":
                    flags[path] = bool(v)
                else:
                    walk(v, path)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{prefix}[{i}]")

    walk(cfg)
    return flags


def _structural_leak_signals(cfg: dict) -> list:
    """本脚本没拿到 key 时,用返回结构判断脱敏是否真的生效。

    服务端本来就不该吐明文,所以唯一的正面证据是「只暴露 has_api_key 布尔位」。
    查三件事:① 响应里没有任何承载明文的字段(按字段名逐个比对,不能子串匹配——
    ``has_api_key`` 就包含 ``api_key``,子串判断会把正确脱敏的响应误报成泄露);
    ② 远程引擎(有 base_url)报 has_api_key=true,说明服务端确实持有 key 却仍没
    回显;③ 离线引擎(无 base_url)一律 has_api_key=false。三条都成立即结构上脱敏。
    """
    leaky = {"api_key", "apikey", "access_token", "secret_key", "key"}

    def walk(node, prefix=""):
        """收集非空的「泄露嫌疑」字段路径。"""
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                path = f"{prefix}.{k}" if prefix else k
                name = k.lower()
                if name in leaky and name != "has_api_key":
                    shown = v if isinstance(v, (int, float, bool)) or v is None else "<值>"
                    if v not in (None, "", []):
                        out.append(f"{path}={shown}")
                out.extend(walk(v, path))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                out.extend(walk(v, f"{prefix}[{i}]"))
        return out

    echoed = walk(cfg)
    engines = cfg.get("engines", []) or []
    remote_true = [
        e.get("name") for e in engines
        if e.get("base_url") and e.get("has_api_key") is not True]
    offline_true = [
        e.get("name") for e in engines
        if not e.get("base_url") and e.get("has_api_key") is not False]
    signals = []
    if echoed:
        signals.append(f"key field echoed: {echoed}")
    if remote_true:
        signals.append(f"remote engine without has_api_key=true: {remote_true}")
    if offline_true:
        signals.append(f"offline engine with has_api_key=true: {offline_true}")
    return signals


def _default_image() -> Path:
    for name in ("tests/兰亭序.jpeg", "test_image.png", "scan.png"):
        p = ROOT / name
        if p.exists():
            return p
    raise SystemExit("找不到测试图片")


def _server_resolvable_url(base: str, image: Path) -> str:
    """返回服务端自己能读到这张图的 image_url 取值;取不到则空串。

    image_url 不是客户端本地路径——它由**服务端**解析:``http(s)://`` 开头时服务端
    自己下载,否则在服务端文件系统上按路径找。所以本机打本机时,传脚本手里的本地
    路径就能解析;本机打远端时,客户端拿不到服务器上存在的文件,只能由调用方通过
    ``JYKJ_OCR_REMOTE_IMAGE_URL`` 显式指定一个服务端可达的网络 URL(本服务不托管
    静态文件,不能靠 ``<base>/<仓库相对路径>`` 拼出来)。

    返回空串时调用方 SKIP 该用例,而不是 FAIL:REST 客户端本来就无法假定服务器的
    文件系统上有测试图片。
    """
    from urllib.parse import urlparse

    if base.startswith("http"):
        host = urlparse(base).netloc.split(":")[0]
        if host in ("localhost", "127.0.0.1", "0.0.0.0"):
            return str(image)
    return os.getenv("JYKJ_OCR_REMOTE_IMAGE_URL", "").strip()


def _detail_text(resp) -> str:
    """取响应的 detail 文案,非 JSON 响应原样截断返回。"""
    try:
        return str(resp.json().get("detail", ""))[:200]
    except Exception:  # noqa: BLE001
        return resp.text[:200]


def _rec(results: list, tag: str, ok: bool | None, detail: dict) -> None:
    """Record one case. ``ok=None`` marks a skip (not testable here), not a fail."""
    results.append({"tag": tag, "ok": None if ok is None else bool(ok),
                    "detail": detail})
    mark = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    # 只打印 ASCII,避免 GBK 控制台编码错误
    print(f"[{mark}] {tag}", flush=True)


def test_endpoints(base: str, image: Path, results: list) -> None:
    print("\n=== A. HTTP 接口 ===", flush=True)

    r = requests.get(f"{base}/health", timeout=10)
    _rec(results, "GET /health", r.status_code == 200,
         {"status": r.status_code, "body": r.json()})

    r = requests.get(f"{base}/engines", timeout=10)
    engines = r.json().get("engines", []) if r.status_code == 200 else []
    _rec(results, "GET /engines", r.status_code == 200 and len(engines) > 0,
         {"status": r.status_code,
          "names": [e.get("name") for e in engines] if isinstance(engines, list) else engines})

    r = requests.get(f"{base}/presets", timeout=10)
    presets = r.json().get("presets", {}) if r.status_code == 200 else {}
    missing = [p for p in ALL_PRESETS
               if p.replace("bestof:smart", "bestof-smart") not in presets]
    _rec(results, "GET /presets",
         r.status_code == 200 and "bestof-fluency" in presets and "bestof:<mode>" in presets,
         {"status": r.status_code, "count": len(presets), "missing": missing})

    r = requests.get(f"{base}/config", timeout=10)
    cfg = r.json() if r.status_code == 200 else {}
    blob = json.dumps(cfg, ensure_ascii=False)
    frags = _leak_fragments()
    leaked = any(frag in blob for frag in frags)
    signals = _structural_leak_signals(cfg)
    # 本机有 key 时用片段判"明文是否出现";拿不到 key(本机打远端)时,只能靠
    # 返回结构证明脱敏——只暴露 has_api_key 布尔位,没有任何 key 类字段。
    structural_ok = not signals
    _rec(results, "GET /config (不泄露 key)",
         None if (r.status_code != 200) else (True if not leaked and structural_ok else False),
         {"status": r.status_code, "key_leaked": leaked,
          "leak_check_active": bool(frags),
          "structural_check": "pass" if structural_ok else signals,
          "has_api_key_flags": {k: v for k, v in _has_key_flags(cfg).items()}})

    # 运行时覆盖 + 回滚(不能留下残留覆盖)
    payload = {"engines": [{"name": "multimodal", "model": "PaddleOCR-VL-1.5"}]}
    r = requests.post(f"{base}/config", json=payload, timeout=10)
    _rec(results, "POST /config", r.status_code in (200, 204), {"status": r.status_code})
    r = requests.delete(f"{base}/config", timeout=10)
    _rec(results, "DELETE /config", r.status_code in (200, 204), {"status": r.status_code})

    # POST /ocr —— multipart
    t0 = time.time()
    with image.open("rb") as fh:
        r = requests.post(f"{base}/ocr",
                          files={"file": (image.name, fh, "image/jpeg")},
                          data={"format": "json", "strategy_name": "seq"}, timeout=600)
    body = r.json() if r.status_code == 200 else {}
    pages = body.get("pages", [])
    _rec(results, "POST /ocr (multipart, strategy_name=seq)",
         r.status_code == 200 and len(pages) > 0 and len(body.get("text", "")) > 0,
         {"status": r.status_code, "pages": len(pages), "chars": len(body.get("text", "")),
          "engine": body.get("engine"), "elapsed_s": round(time.time() - t0, 1)})

    # POST /ocr/text —— image_url。取不到服务端可达的来源就 SKIP:REST 客户端
    # 无法假定服务器文件系统上有测试图片,强行发本机路径只会稳定 400。
    server_src = _server_resolvable_url(base, image)
    if server_src:
        src_kind = "url" if server_src.startswith("http") else "path"
    else:
        src_kind = ""

    # POST /ocr/text —— image_url(服务端可达的来源:本机路径或 http URL)
    if server_src:
        t0 = time.time()
        r = requests.post(f"{base}/ocr/text",
                          json={"image_url": server_src, "strategy_name": "local",
                                "format": "json"}, timeout=300)
        body = r.json() if r.status_code == 200 else {}
        _rec(results, "POST /ocr/text (image_url, local)",
             r.status_code == 200 and len(body.get("text", "")) > 0,
             {"status": r.status_code, "chars": len(body.get("text", "")),
              "engine": body.get("engine"), "source_kind": src_kind,
              "elapsed_s": round(time.time() - t0, 1)})
    else:
        _rec(results, "POST /ocr/text (image_url, local)", None,
             {"reason": "无服务端可达的图片来源;远端请设 JYKJ_OCR_REMOTE_IMAGE_URL"})

    # POST /ocr/text —— image_url 指向服务端不存在的文件 -> 400(与部署拓扑无关)
    r = requests.post(f"{base}/ocr/text",
                      json={"image_url": "/no/such/file_zz.png", "strategy_name": "local",
                            "format": "json"}, timeout=60)
    _rec(results, "POST /ocr/text 不存在的 image_url -> 400",
         r.status_code == 400,
         {"status": r.status_code, "detail": _detail_text(r)})

    # POST /ocr/text —— image_data(完整 data URI)
    uri = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    r = requests.post(f"{base}/ocr/text",
                      json={"image_data": uri, "strategy_name": "local",
                            "format": "json"}, timeout=300)
    body = r.json() if r.status_code == 200 else {}
    _rec(results, "POST /ocr/text (image_data data-URI)",
         r.status_code == 200 and len(body.get("text", "")) > 0,
         {"status": r.status_code, "chars": len(body.get("text", ""))})

    # POST /ocr/text —— image_b64
    b64 = base64.b64encode(image.read_bytes()).decode()
    r = requests.post(f"{base}/ocr/text",
                      json={"image_b64": b64, "strategy_name": "local",
                            "format": "json"}, timeout=300)
    body = r.json() if r.status_code == 200 else {}
    _rec(results, "POST /ocr/text (image_b64)",
         r.status_code == 200 and len(body.get("text", "")) > 0,
         {"status": r.status_code, "chars": len(body.get("text", ""))})

    # POST /ocr/{preset} —— 路由即策略
    with image.open("rb") as fh:
        r = requests.post(f"{base}/ocr/vl",
                          files={"file": (image.name, fh, "image/jpeg")},
                          data={"format": "json"}, timeout=600)
    body = r.json() if r.status_code == 200 else {}
    _rec(results, "POST /ocr/{preset} (preset=vl)",
         r.status_code == 200 and len(body.get("text", "")) > 0,
         {"status": r.status_code, "chars": len(body.get("text", "")),
          "engine": body.get("engine")})

    # POST /ocr/{preset}/text —— 同样依赖服务端可读的 image_url
    if server_src:
        r = requests.post(f"{base}/ocr/vl/text",
                          json={"image_url": server_src, "format": "json"}, timeout=600)
        body = r.json() if r.status_code == 200 else {}
        _rec(results, "POST /ocr/{preset}/text (preset=vl)",
             r.status_code == 200 and len(body.get("text", "")) > 0,
             {"status": r.status_code, "chars": len(body.get("text", "")),
              "engine": body.get("engine")})
    else:
        _rec(results, "POST /ocr/{preset}/text (preset=vl)", None,
             {"reason": "无服务端可达的图片来源;远端请设 JYKJ_OCR_REMOTE_IMAGE_URL"})

    # 输出格式退化:text / markdown 返回的是纯文本响应体(PlainTextResponse),
    # 不是 JSON 信封 —— server.py:509。
    for fmt in ("text", "markdown"):
        with image.open("rb") as fh:
            r = requests.post(f"{base}/ocr",
                              files={"file": (image.name, fh, "image/jpeg")},
                              data={"format": fmt, "strategy_name": "local"}, timeout=300)
        is_text = "text/plain" in r.headers.get("content-type", "")
        _rec(results, f"POST /ocr format={fmt} (纯文本退化)",
             r.status_code == 200 and is_text and len(r.text.strip()) > 0,
             {"status": r.status_code,
              "content_type": r.headers.get("content-type", ""),
              "chars": len(r.text), "is_json_envelope": False})

    # 异常路径:缺少 file
    r = requests.post(f"{base}/ocr", data={}, timeout=30)
    _rec(results, "POST /ocr 缺 file -> 4xx", 400 <= r.status_code < 500,
         {"status": r.status_code})

    # 未知预设 -> 400
    with image.open("rb") as fh:
        r = requests.post(f"{base}/ocr",
                          files={"file": (image.name, fh, "image/jpeg")},
                          data={"strategy_name": "no-such-preset-xyz"}, timeout=30)
    _rec(results, "未知 strategy_name -> 400", r.status_code == 400,
         {"status": r.status_code, "body": r.text[:200]})


def test_presets(base: str, image: Path, results: list) -> None:
    print("\n=== B. 策略预设(真实模型) ===", flush=True)
    for preset in ALL_PRESETS:
        t0 = time.time()
        try:
            with image.open("rb") as fh:
                r = requests.post(f"{base}/ocr",
                                  files={"file": (image.name, fh, "image/jpeg")},
                                  data={"strategy_name": preset, "format": "json"},
                                  timeout=900)
            dt = round(time.time() - t0, 1)
            if r.status_code != 200:
                _rec(results, f"preset={preset}", False,
                     {"status": r.status_code, "elapsed_s": dt,
                      "body": r.text[:300]})
                continue
            body = r.json()
            pages = body.get("pages", [])
            engines = sorted({(p.get("engine") or "") for p in pages})
            models = sorted({(p.get("model") or "") for p in pages})
            text = body.get("text", "")
            _rec(results, f"preset={preset}",
                 len(text) > 0 and len(pages) > 0,
                 {"status": 200, "chars": len(text), "pages": len(pages),
                  "engines": engines, "models": models,
                  "top_engine": body.get("engine"), "elapsed_s": dt})
        except Exception as exc:  # noqa: BLE001
            _rec(results, f"preset={preset}", False,
                 {"exception": f"{type(exc).__name__}: {exc}",
                  "elapsed_s": round(time.time() - t0, 1)})


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    image = Path(sys.argv[2]) if len(sys.argv) > 2 else _default_image()
    print(f"base   = {base}", flush=True)
    print(f"image  = {image.name} ({image.stat().st_size} bytes)", flush=True)

    try:
        requests.get(f"{base}/health", timeout=5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"服务不可达 {base}: {exc}")

    results: list = []
    t0 = time.time()
    test_endpoints(base, image, results)
    test_presets(base, image, results)
    total = round(time.time() - t0, 1)

    fails = [r for r in results if r["ok"] is False]
    skips = [r for r in results if r["ok"] is None]
    out = {
        "base": base,
        "image": str(image),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_s": total,
        "cases": len(results),
        "passed": sum(1 for r in results if r["ok"] is True),
        "skipped": len(skips),
        "failed": len(fails),
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}", flush=True)
    print(f"cases={len(results)} passed={out['passed']} skipped={len(skips)} "
          f"failed={len(fails)} total_s={total}", flush=True)
    for s in skips:
        print(f"  SKIP: {s['tag']} -> {json.dumps(s['detail'], ensure_ascii=False)[:160]}",
              flush=True)
    for f in fails:
        print(f"  FAIL: {f['tag']} -> {json.dumps(f['detail'], ensure_ascii=False)[:220]}",
              flush=True)
    print(f"detail -> {RESULT_PATH.name}", flush=True)
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
