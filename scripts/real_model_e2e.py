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


def _default_image() -> Path:
    for name in ("tests/兰亭序.jpeg", "test_image.png", "scan.png"):
        p = ROOT / name
        if p.exists():
            return p
    raise SystemExit("找不到测试图片")


def _rec(results: list, tag: str, ok: bool, detail: dict) -> None:
    results.append({"tag": tag, "ok": bool(ok), "detail": detail})
    mark = "PASS" if ok else "FAIL"
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
    _rec(results, "GET /config (不泄露 key)",
         r.status_code == 200 and not leaked and bool(frags),
         {"status": r.status_code, "key_leaked": leaked,
          "leak_check_active": bool(frags),
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

    # POST /ocr/text —— image_url(本地路径)
    t0 = time.time()
    r = requests.post(f"{base}/ocr/text",
                      json={"image_url": str(image), "strategy_name": "local",
                            "format": "json"}, timeout=300)
    body = r.json() if r.status_code == 200 else {}
    _rec(results, "POST /ocr/text (image_url, local)",
         r.status_code == 200 and len(body.get("text", "")) > 0,
         {"status": r.status_code, "chars": len(body.get("text", "")),
          "engine": body.get("engine"), "elapsed_s": round(time.time() - t0, 1)})

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

    # POST /ocr/{preset}/text
    r = requests.post(f"{base}/ocr/vl/text",
                      json={"image_url": str(image), "format": "json"}, timeout=600)
    body = r.json() if r.status_code == 200 else {}
    _rec(results, "POST /ocr/{preset}/text (preset=vl)",
         r.status_code == 200 and len(body.get("text", "")) > 0,
         {"status": r.status_code, "chars": len(body.get("text", "")),
          "engine": body.get("engine")})

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

    fails = [r for r in results if not r["ok"]]
    out = {
        "base": base,
        "image": str(image),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_s": total,
        "cases": len(results),
        "passed": len(results) - len(fails),
        "failed": len(fails),
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}", flush=True)
    print(f"cases={len(results)} passed={len(results) - len(fails)} failed={len(fails)} "
          f"total_s={total}", flush=True)
    for f in fails:
        print(f"  FAIL: {f['tag']} -> {json.dumps(f['detail'], ensure_ascii=False)[:220]}",
              flush=True)
    print(f"detail -> {RESULT_PATH.name}", flush=True)
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
