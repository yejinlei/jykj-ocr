# -*- coding: utf-8 -*-
"""jykj_ocr HTTP API 使用示例(通过 requests 调用本地服务)。

四种常见场景:指定引擎 / 策略链 / 一次性预设 / 只取纯文本。
依赖: pip install requests
运行:
    python scripts/demo.py <服务地址> <图片路径>
    例: python scripts/demo.py http://192.168.0.81:8000 tests/兰亭序.jpeg
    (不传参数时默认本地 127.0.0.1:8000 + tests/兰亭序.jpeg)
"""

from __future__ import annotations

import sys
from pathlib import Path


def _default_image() -> Path:
    root = Path(__file__).resolve().parent.parent
    for name in ("tests/兰亭序.jpeg", "test_image.png"):
        p = root / name
        if p.exists():
            return p
    return root / "scan.png"


def post_json(session, base: str, path: str, payload: dict) -> dict:
    r = session.post(f"{base}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def demo_single_engine(base: str, image: Path) -> None:
    """场景 1 — 指定引擎 rapidocr(multipart 上传文件,本地离线)。"""
    import requests

    print("=" * 60)
    print("[场景 1] 指定引擎 rapidocr(POST /ocr 文件上传)")
    print("=" * 60)
    with requests.Session() as s:
        with image.open("rb") as f:
            r = s.post(
                f"{base}/ocr",
                files={"file": (image.name, f, "image/jpeg")},
                data={"engine": "rapidocr", "format": "json"},
                timeout=120,
            )
            r.raise_for_status()
        d = r.json()
    # /ocr 文件上传返回 {"pages":[...],"text":...,"engine":...,"page_count":N}
    page = d["pages"][0]
    print(f"  engine   = {page.get('engine')}")
    print(f"  model    = {page.get('model') or '-'}")
    print(f"  regions  = {page.get('region_count')}")
    print(f"  text[:120]= {page.get('text', '')[:120]}...")


def demo_strategy(base: str, image: Path) -> None:
    """场景 2 — 策略链 fallback(通过 /ocr 文件上传 + strategy_name)。"""
    import requests

    print("=" * 60)
    print("[场景 2] 策略链 fallback(POST /ocr,一次性 strategy_name)")
    print("=" * 60)
    with requests.Session() as s:
        with image.open("rb") as f:
            r = s.post(
                f"{base}/ocr",
                files={"file": (image.name, f, "image/jpeg")},
                data={"strategy_name": "fallback", "format": "json"},
                timeout=120,
            )
            r.raise_for_status()
        d = r.json()
    page = d["pages"][0]
    print(f"  engine = {page.get('engine')}")
    print(f"  text   = {page.get('text', '')[:100]}...")


def demo_quality_preset(base: str, image: Path) -> None:
    """场景 3 — 一次性预设 quality(回退 + 窜行降级 + 阅读顺序重排)。"""
    import requests

    print("=" * 60)
    print("[场景 3] 一次性预设 quality(POST /ocr,engine 留空走策略链)")
    print("=" * 60)
    with requests.Session() as s:
        with image.open("rb") as f:
            r = s.post(
                f"{base}/ocr",
                files={"file": (image.name, f, "image/jpeg")},
                data={"strategy_name": "quality", "format": "json"},
                timeout=120,
            )
            r.raise_for_status()
        d = r.json()
    page = d["pages"][0]
    print(f"  engine = {page.get('engine')}")
    print(f"  text   = {page.get('text', '')[:120]}...")


def demo_text_only(base: str, image: Path) -> None:
    """场景 4 — 只取拼接好的纯文本(POST /ocr 文件上传 + format=markdown)。"""
    import requests

    print("=" * 60)
    print("[场景 4] 只取纯文本(POST /ocr + format=markdown)")
    print("=" * 60)
    with requests.Session() as s:
        with image.open("rb") as f:
            r = s.post(
                f"{base}/ocr",
                files={"file": (image.name, f, "image/jpeg")},
                data={"engine": "rapidocr", "format": "markdown"},
                timeout=120,
            )
            r.raise_for_status()
        text = r.text
    print(text[:400] if len(text) > 400 else text)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.81:8000"
    image = Path(sys.argv[2]) if len(sys.argv) > 2 else _default_image()
    if not image.exists():
        print(f"图片不存在: {image}", file=sys.stderr)
        sys.exit(1)
    print(f"service: {base}")
    print(f"source:  {image}\n")
    demo_single_engine(base, image)
    demo_strategy(base, image)
    demo_quality_preset(base, image)
    demo_text_only(base, image)