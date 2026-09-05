# -*- coding: utf-8 -*-
"""一次性联调:用 PIL 造图,直打某个 OpenAI 兼容平台的 /chat/completions 验证余额+模型。

凭据走通用变量(与 config.py 同一套解析顺序),硅基流动只是本脚本默认的平台:

    OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.siliconflow.cn/v1 \
        python scripts/probe_siliconflow.py

需要探别的平台时覆盖 URL 与模型:

    PROBE_URL=https://api.moark.com/v1/chat/completions \
    PROBE_MODEL=PaddleOCR-VL-1.5 python scripts/probe_siliconflow.py
"""
import base64, io, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) or ".", "..", "src")))
from jykj_ocr.cli import _load_dotenv
_load_dotenv()
import requests
from PIL import Image, ImageDraw, ImageFont

KEY = (os.getenv("OPENAI_API_KEY")
       or os.getenv("JYKJ_OCR_MULTIMODAL_API_KEY")
       or os.getenv("MULTIMODAL_API_KEY")
       or "")
if not KEY:
    print("OPENAI_API_KEY not set"); sys.exit(2)
# 只打长度与是否非空——不要打印 key 的任何片段,日志会进版本库
print(f"[key] len={len(KEY)}")

# 端点与模型都可覆盖:不传 PROBE_URL 时跟随 OPENAI_BASE_URL(与 config.py 一致),
# 再没有就退回硅基流动内置端点——只是默认值,不代表哪个平台被特殊对待。
_base = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
URL = os.getenv("PROBE_URL", f"{_base}/chat/completions")
MODEL = os.getenv("PROBE_MODEL", "PaddlePaddle/PaddleOCR-VL-1.5")

# 优先用系统里支持中文的字体,字号做大,避免小字被模型漏识别
font = None
for path in (
    r"C:\Windows\Fonts\msyh.ttc",       # Microsoft YaHei
    r"C:\Windows\Fonts\simsun.ttc",     # SimSun
    r"C:\Windows\Fonts\simhei.ttf",     # SimHei
):
    if os.path.isfile(path):
        font = ImageFont.truetype(path, 56)
        print(f"[font] {path}")
        break
if font is None:
    font = ImageFont.load_default()
    print("[font] default (no CJK font found)")

canvas_w, canvas_h = 1280, 400
img = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(img)
# 分两行,减少每行长度
lines = ["Hello 世界", "2026 OCR 测试"]
y = 40
for line in lines:
    bbox = draw.textbbox((0, 0), line, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, y), line, fill="black", font=font)
    y += 150
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "请识别图片中的全部文字，只输出识别到的文字。"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}],
    "temperature": 0,
}
print(f"[req] POST {URL} model={MODEL}")
r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, json=payload, timeout=180)
print(f"[resp] HTTP {r.status_code}")
print(f"[body] {r.text[:3000]}")
try:
    data = r.json()
except ValueError:
    sys.exit(0)
choices = data.get("choices") or []
if choices:
    msg = ((choices[0].get("message") or {}).get("content") or "").strip()
    print("\n=== recognised ===")
    print(msg)
    print("=== tokens ===")
    for t in ("Hello", "世界", "OCR", "2026"):
        print(f"  '{t}': {t in msg}")
