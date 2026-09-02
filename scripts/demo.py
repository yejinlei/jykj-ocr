# -*- coding: utf-8 -*-
"""jykj_ocr Python API 使用示例。

四种常见场景:指定引擎 / 策略链 / 一次性预设 / 只取纯文本。
运行: python scripts/demo.py <图片路径或URL>
       (不传参数时使用 tests/兰亭序.jpeg 或本地示例图片)
"""

from __future__ import annotations

import sys
from pathlib import Path


def _default_image() -> str:
    root = Path(__file__).resolve().parent.parent
    candidates = [root / "tests" / "兰亭序.jpeg", root / "test_image.png"]
    for p in candidates:
        if p.exists():
            return str(p)
    return "scan.png"


def demo_single_engine(image: str) -> None:
    """场景 1 — 指定引擎(离线/在线任选)。"""
    import jykj_ocr

    print("=" * 60)
    print("[场景 1] 指定引擎: rapidocr(本地离线)")
    print("=" * 60)
    results = jykj_ocr.ocr(image, engine="rapidocr")
    for r in results:
        print(f"  engine = {r.engine}")
        print(f"  model  = {r.model or '-'}")
        print(f"  regions= {len(r.regions)}")
        print(f"  text   = {r.text[:120]}...")


def demo_strategy(image: str) -> None:
    """场景 2 — 策略链(按 config 里引擎顺序依次尝试,首个通过即返回)。"""
    import jykj_ocr

    print("=" * 60)
    print("[场景 2] 策略链 fallback(默认回退,不改配置文件)")
    print("=" * 60)
    results = jykj_ocr.ocr(image, strategy_name="fallback")
    for r in results:
        print(f"  engine = {r.engine}")
        print(f"  model  = {r.model or '-'}")
        print(f"  text[:80] = {r.text[:80]}...")


def demo_quality_preset(image: str) -> None:
    """场景 3 — 一次性预设 quality(回退 + 窜行降级 + 阅读顺序重排)。"""
    import jykj_ocr

    print("=" * 60)
    print("[场景 3] 一次性预设 quality")
    print("=" * 60)
    results = jykj_ocr.ocr(image, strategy_name="quality")
    for r in results:
        print(f"  engine = {r.engine}")
        print(f"  text[:120] = {r.text[:120]}...")


def demo_text_only(image: str) -> None:
    """场景 4 — 只取拼接好的纯文本(markdown 分隔)。"""
    import jykj_ocr

    print("=" * 60)
    print("[场景 4] 只取纯文本")
    print("=" * 60)
    text = jykj_ocr.ocr_to_text(image, engine="rapidocr")
    print(text[:400] if len(text) > 400 else text)


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else _default_image()
    print(f"source: {image}\n")
    demo_single_engine(image)
    demo_strategy(image)
    demo_quality_preset(image)
    demo_text_only(image)