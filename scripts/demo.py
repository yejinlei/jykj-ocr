# -*- coding: utf-8 -*-
"""jykj_ocr HTTP API 全场景使用示例(通过 requests 调用本地服务)。

覆盖场景:
  1. 单独引擎:rapidocr / siliconflow / multimodal
  2. 顺序策略:local / vl / seq / seq-any / seq-low_conf / seq-line_overlap
  3. 最佳策略:bestof / bestof-smart / bestof-fastest / bestof-confidence /
             bestof-longest / bestof:<mode>
  4. legacy 别名:fallback / quality(验证与 seq/seq-any 等价)
  5. 输出格式:text / markdown / json

依赖:pip install requests
运行:
    python scripts/demo.py <服务地址> <图片路径>
    例: python scripts/demo.py http://192.168.0.81:8000 tests/兰亭序.jpeg
    (不传参数时默认本地 127.0.0.1:8000 + tests/兰亭序.jpeg)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _default_image() -> Path:
    root = Path(__file__).resolve().parent.parent
    for name in ("tests/兰亭序.jpeg", "test_image.png"):
        p = root / name
        if p.exists():
            return p
    return root / "scan.png"


def _section(tag: str) -> None:
    print()
    print("=" * 72)
    print(f"[{tag}]")
    print("=" * 72)


def _ocr_json(base: str, image: Path, **form) -> dict:
    """POST /ocr 文件上传,返回 JSON。"""
    import requests

    with image.open("rb") as f:
        r = requests.post(
            f"{base}/ocr",
            files={"file": (image.name, f, "image/jpeg")},
            data=form,
            timeout=180,
        )
        r.raise_for_status()
    return r.json()


def _ocr_plain(base: str, image: Path, **form) -> str:
    """POST /ocr 文件上传,返回响应体文本。"""
    import requests

    with image.open("rb") as f:
        r = requests.post(
            f"{base}/ocr",
            files={"file": (image.name, f, "image/jpeg")},
            data=form,
            timeout=180,
        )
        r.raise_for_status()
    return r.text


def _page(d: dict) -> dict:
    return (d.get("pages") or [d])[0]


def _summary(label: str, page: dict, elapsed: float, text_len: int | None = None) -> None:
    engine = page.get("engine") or "-"
    regions = page.get("region_count", len(page.get("regions") or []))
    tlen = text_len if text_len is not None else len(page.get("text") or "")
    print(f"  {label:<26} engine={engine:<12} regions={regions:<4} "
          f"chars={tlen:<5} time={elapsed:.1f}s")


# ============================================================================
# 场景 1 —— 单独引擎
# ============================================================================
def demo_single_engines(base: str, image: Path) -> None:
    _section("场景 1 · 单独引擎(rapidocr / siliconflow)")
    for engine in ("rapidocr", "siliconflow"):
        t0 = time.perf_counter()
        try:
            d = _ocr_json(base, image, engine=engine, format="json")
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {engine:<26} 失败: {exc}")
            continue
        page = _page(d)
        _summary(engine, page, elapsed)


# ============================================================================
# 场景 2 —— 顺序策略 seq*
# ============================================================================
def demo_seq_presets(base: str, image: Path) -> None:
    _section("场景 2 · 顺序策略 seq*(首个命中即返回)")
    for preset in ("local", "vl", "seq", "seq-any", "seq-low_conf", "seq-line_overlap"):
        t0 = time.perf_counter()
        try:
            d = _ocr_json(base, image, strategy_name=preset, format="json")
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {preset:<26} 失败: {exc}")
            continue
        page = _page(d)
        _summary(preset, page, elapsed)


# ============================================================================
# 场景 3 —— 最佳策略 bestof*
# ============================================================================
def demo_bestof_presets(base: str, image: Path) -> None:
    _section("场景 3 · 最佳策略 bestof*(所有引擎各跑一次,按评分选最佳)")
    for preset in ("bestof", "bestof-smart", "bestof-fastest",
                   "bestof-confidence", "bestof-longest"):
        t0 = time.perf_counter()
        try:
            d = _ocr_json(base, image, strategy_name=preset, format="json")
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {preset:<26} 失败: {exc}")
            continue
        page = _page(d)
        _summary(preset, page, elapsed)

    _section("  子场景:bestof:<mode> 语法别名(等价于 bestof-mode)")
    for preset in ("bestof:smart", "bestof:fastest"):
        t0 = time.perf_counter()
        try:
            d = _ocr_json(base, image, strategy_name=preset, format="json")
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"    {preset:<24} 失败: {exc}")
            continue
        page = _page(d)
        _summary(preset, page, elapsed)


# ============================================================================
# 场景 4 —— legacy 别名(fallback / quality)
# ============================================================================
def demo_legacy_aliases(base: str, image: Path) -> None:
    _section("场景 4 · legacy 别名(fallback == seq, quality == seq-any)")
    for preset in ("fallback", "quality"):
        t0 = time.perf_counter()
        try:
            d = _ocr_json(base, image, strategy_name=preset, format="json")
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {preset:<26} 失败: {exc}")
            continue
        page = _page(d)
        _summary(preset, page, elapsed)


# ============================================================================
# 场景 5 —— 输出格式(text / markdown / json)
# ============================================================================
def demo_output_formats(base: str, image: Path) -> None:
    _section("场景 5 · 输出格式(json / markdown / text)")
    for fmt in ("json", "markdown", "text"):
        t0 = time.perf_counter()
        try:
            if fmt == "json":
                d = _ocr_json(base, image, engine="rapidocr", format=fmt)
                text = _page(d).get("text") or ""
            else:
                text = _ocr_plain(base, image, engine="rapidocr", format=fmt)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {fmt:<10} 失败: {exc}")
            continue
        snippet = text[:100].replace("\n", " / ") if text else "(empty)"
        print(f"  {fmt:<10} time={elapsed:.1f}s  text[:100]={snippet!r}")


# ============================================================================
# 场景 6 —— 错误处理(未知预设应返回 400)
# ============================================================================
def demo_unknown_preset(base: str, image: Path) -> None:
    _section("场景 6 · 错误处理(未知预设应返回 400)")
    import requests
    with image.open("rb") as f:
        r = requests.post(
            f"{base}/ocr",
            files={"file": (image.name, f, "image/jpeg")},
            data={"strategy_name": "unknown-preset", "format": "json"},
            timeout=30,
        )
    if r.status_code == 400:
        print(f"  未知预设 → HTTP {r.status_code} (符合预期)")
    else:
        print(f"  未知预设 → HTTP {r.status_code} (期望 400)")


def main() -> None:
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://192.168.0.81:8000"
    image = Path(sys.argv[2]) if len(sys.argv) > 2 else _default_image()
    if not image.exists():
        print(f"图片不存在: {image}", file=sys.stderr)
        sys.exit(1)

    print(f"service: {base}")
    print(f"source:  {image}")
    print(f"size:    {image.stat().st_size:,} bytes")
    print()

    demo_single_engines(base, image)
    demo_seq_presets(base, image)
    demo_bestof_presets(base, image)
    demo_legacy_aliases(base, image)
    demo_output_formats(base, image)
    demo_unknown_preset(base, image)

    print()
    print("=" * 72)
    print("全部场景执行完毕")
    print("=" * 72)


if __name__ == "__main__":
    main()