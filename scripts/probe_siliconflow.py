# -*- coding: utf-8 -*-
"""一次性联调:用 PIL 造图,直打 SiliconFlow /chat/completions 验证余额+模型。"""
import base64, io, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) or ".", "..", "src")))
from jykj_ocr.cli import _load_dotenv
_load_dotenv()
import requests
from PIL import Image, ImageDraw, ImageFont

KEY = os.getenv("SILICONFLOW_API_KEY", "")
if not KEY:
    print("SILICONFLOW_API_KEY not set"); sys.exit(2)
print(f"[key] len={len(KEY)} tail={KEY[-6:]}")

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
    "model": "PaddlePaddle/PaddleOCR-VL-1.5",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "请识别图片中的全部文字，只输出识别到的文字。"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}],
    "temperature": 0,
}
url = "https://api.siliconflow.cn/v1/chat/completions"
print(f"[req] POST {url}")
r = requests.post(url, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, json=payload, timeout=180)
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
