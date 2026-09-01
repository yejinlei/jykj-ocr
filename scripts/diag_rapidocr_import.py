# -*- coding: utf-8 -*-
"""在服务器上诊断 rapidocr 引擎报 "not installed" 的真实原因。

jykj_ocr 的 _load_rapidocr_class() 只捕获 ImportError,任何传递依赖
(cv2 缺 libGL、onnxruntime/numpy 不兼容等)在 import 过程中抛出的
ImportError 都会被吞掉,统一显示成误导性的 "RapidOCR is not installed"。

本脚本绕过这层封装,直接逐个 import 并打印【真实】异常,一行命令定位根因。

用法(必须用运行服务的那个解释器):
    /root/src/jykj-ocr/.venv/bin/python scripts/diag_rapidocr_import.py
"""
from __future__ import annotations

import platform
import sys
import traceback


def probe(label: str, fn) -> bool:
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
        return False
    print(f"[ OK ] {label}: {value}")
    return True


print(f"python     : {sys.version.splitlines()[0]}")
print(f"executable : {sys.executable}")
print(f"platform   : {platform.platform()}")
try:
    import numpy

    print(f"numpy      : {numpy.__version__}")
except Exception as exc:  # noqa: BLE001
    print(f"numpy      : FAILED to import -> {exc}")
print("-" * 60)

ok_cv2 = probe("import cv2", lambda: __import__("cv2").__version__)
ok_ort = probe("import onnxruntime", lambda: __import__("onnxruntime").__version__)
ok_pkg = probe("import rapidocr_onnxruntime", lambda: getattr(
    __import__("rapidocr_onnxruntime"), "__version__", "?"))
cls = probe("from rapidocr_onnxruntime import RapidOCR",
            lambda: getattr(__import__("rapidocr_onnxruntime", fromlist=["RapidOCR"]), "RapidOCR"))

if ok_cv2 and ok_ort and ok_pkg and cls:
    print("-" * 60)
    probe("构造 RapidOCR() 实例", lambda: __import__(
        "rapidocr_onnxruntime", fromlist=["RapidOCR"]).RapidOCR())
    print("结论: 该解释器下 rapidocr 完全可用 —— 若服务仍报 not installed,"
          "说明 uvicorn 跑的不是这个解释器,用 ps aux | grep uvicorn 确认。")
