# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# jykj_ocr 服务镜像
#
#   docker build -t jykj_ocr .
#   docker run --rm -p 8000:8000 --env-file .env jykj_ocr
#
#   或使用 compose：
#   docker compose up --build
#
# 本地一次性任务（不进容器服务）：
#   docker run --rm --env-file .env -v "$PWD:/data" jykj_ocr \
#       python -m jykj_ocr /data/image.png --engine siliconflow
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JYKJ_OCR_CONFIG=/app/config/config.yaml \
    JYKJ_OCR_PORT=8000

WORKDIR /app

# ---------------------------------------------------------------------------
# 依赖层 —— 只有 requirements.txt 变化才会重建，其余命中缓存
#
# 无 apt 系统依赖层:rapidocr 的传递依赖已用 opencv-python-headless 顶替
# GUI 版(见 requirements.txt),import cv2 不再链接 libGL.so.1,slim 镜像
# 开箱即用。若换回 opencv-python 需补装 libgl1 libglib2.0-0。
# ---------------------------------------------------------------------------
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# RapidOCR 权重缓存：首次运行自动下载，可用 volume 预置。
# 然后在 config 中设置 engines.rapidocr.model_dir=/opt/models
ARG RAPIDOCR_HOME=/opt/models
RUN mkdir -p "$RAPIDOCR_HOME"
ENV RAPIDOCR_HOME=$RAPIDOCR_HOME

# ---------------------------------------------------------------------------
# 应用层
# ---------------------------------------------------------------------------
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

# 非 root 用户运行
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app "$RAPIDOCR_HOME"
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "jykj_ocr.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
