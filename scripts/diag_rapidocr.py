# -*- coding: utf-8 -*-
"""Diagnose rapidocr-onnxruntime raw return shape for each input type.

Read-only probe: prints the type/len/sample of every element so we can see
whether the installed version returns (boxes, txts, scores) or a different
shape (e.g. with an `elapsed` field, or txts/scores swapped).

Usage:  .venv/Scripts/python scripts/diag_rapidocr.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from rapidocr_onnxruntime import RapidOCR  # noqa: E402

IMG = os.path.abspath(os.path.join(_HERE, "..", "test_image.png"))

ocr = RapidOCR()
img_pil = Image.open(IMG)
img_np = np.array(img_pil)


def dump(label: str, inp) -> None:
    print(f"=== input: {label} ({type(inp).__name__}) ===")
    try:
        res = ocr(inp)
    except Exception as exc:  # noqa: BLE001
        print("  ERROR:", repr(exc))
        print()
        return
    print("  type:", type(res).__name__)
    if res is None:
        print("  None")
    elif isinstance(res, (tuple, list)):
        print("  len:", len(res))
        for i, part in enumerate(res):
            n = len(part) if hasattr(part, "__len__") else None
            print(f"    [{i}] type={type(part).__name__} len={n}")
            if n:
                print(f"        sample[0]={part[0]!r}")
                if n > 1:
                    print(f"        sample[1]={part[1]!r}")
    elif isinstance(res, dict):
        print("  keys:", list(res.keys()))
        for k, v in res.items():
            n = len(v) if hasattr(v, "__len__") else None
            preview = (v[0] if n else v) if n is not None else v
            print(f"    {k}: type={type(v).__name__} len={n} sample={preview!r}")
    else:
        print("  value:", repr(res))
    print()


dump("filepath", IMG)
dump("numpy", img_np)
dump("PIL", img_pil)
