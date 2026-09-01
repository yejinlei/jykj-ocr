# -*- coding: utf-8 -*-
"""按接口方式端到端测试三个引擎：Python API + FastAPI HTTP。

只从环境变量读 API key（不读 .env、不写任何文件），结果只打印长度与
脱敏摘要，避免凭据落入磁盘。测试图在内存中生成，不依赖外部文件。

前置：导出 OPENAI_API_KEY / OPENAI_BASE_URL（或 SILICONFLOW_API_KEY）。
用法：  .venv/Scripts/python scripts/test_interfaces.py
"""
from __future__ import annotations

import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def _make_image() -> bytes:
    """内存中生成一张带英文文字的 PNG。"""
    img = Image.new("RGB", (640, 200), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 56)
    d.text((30, 30), "HELLO OCR", fill="black", font=font)
    d.text((30, 110), "2026 TEST", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask(text: str) -> str:
    """脱敏：只保留长度与前 8/后 4 字符。"""
    if not text:
        return "<empty>"
    if len(text) <= 12:
        return f"<{len(text)} chars>"
    return f"{text[:8]}…{text[-4:]} ({len(text)})"


def _summary(engine: str, ok: bool, text: str, extra: str = "") -> str:
    head = text.replace("\n", " ")[:60]
    return f"  [{engine}] ok={ok} len={len(text)} head={head!r} {extra}".rstrip()


def main() -> int:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    if not key:
        print("ERROR: set OPENAI_API_KEY (or SILICONFLOW_API_KEY) first")
        return 2
    print(f"key: {_mask(key)}")
    print(f"base_url env: {_mask(os.getenv('OPENAI_BASE_URL', ''))}")

    import jykj_ocr  # noqa: E402

    png = _make_image()
    # ocr() only takes a path/URL, so land the in-memory PNG in a temp file.
    import tempfile  # noqa: E402

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(png)
    tmp.close()
    img_path = tmp.name
    failures = 0

    # ---- 1) Python API: ocr() ----
    print("\n=== 1. Python API (jykj_ocr.ocr) ===")
    for eng in ("rapidocr", "siliconflow", "multimodal"):
        try:
            res = jykj_ocr.ocr(img_path, engine=eng)
            r = res[0]
            ok = bool(r.text.strip())
            print(_summary(eng, ok, r.text, f"model={r.model}"))
            if not ok:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [{eng}] ERROR: {exc!r}")
            failures += 1
    try:
        os.unlink(img_path)
    except OSError:
        pass

    # ---- 2) FastAPI HTTP: POST /ocr ----
    print("\n=== 2. HTTP POST /ocr (TestClient) ===")
    from fastapi.testclient import TestClient  # noqa: E402

    from jykj_ocr.server import app  # noqa: E402

    client = TestClient(app)
    for eng in ("rapidocr", "siliconflow", "multimodal"):
        try:
            resp = client.post(
                "/ocr",
                files={"file": ("t.png", png, "image/png")},
                data={"engine": eng, "format": "json"},
            )
            if resp.status_code != 200:
                print(f"  [{eng}] HTTP {resp.status_code}: {resp.text[:200]}")
                failures += 1
                continue
            body = resp.json()
            text = body.get("text", "")
            ok = bool(text.strip())
            print(_summary(eng, ok, text, f"http=200 engine={body.get('engine')} pages={body.get('page_count')}"))
            if not ok:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [{eng}] ERROR: {exc!r}")
            failures += 1

    # ---- 3) HTTP: GET /config 不泄露 key ----
    print("\n=== 3. GET /config (不应泄露 key 明文) ===")
    cfg = client.get("/config").json()
    leak = any(_mask(key) not in str(v) and key in str(v) for v in cfg.get("engines", []))
    has_key_flags = [e.get("name") for e in cfg.get("engines", []) if e.get("has_api_key")]
    print(f"  key_leaked={leak} engines_with_key={has_key_flags}")

    print(f"\n{'FAIL' if failures else 'PASS'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
