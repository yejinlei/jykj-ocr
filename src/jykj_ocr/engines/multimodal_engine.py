# -*- coding: utf-8 -*-
"""OpenAI-compatible multimodal OCR engine.

Works with SiliconFlow and any provider exposing
``POST /chat/completions`` with ``image_url`` content parts. The SiliconFlow
engine subclasses this one and only pins defaults.

Only ``requests`` is required — no ``openai`` SDK dependency, which keeps the
image small and avoids SDK churn.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List

from ..config import EngineConfig, load_prompt, normalise_engine
from ..engine.base import (
    MAX_IMAGE_BYTES,
    BaseEngine,
    EngineNotAvailable,
    PageImage,
    register,
)
from ..models import BoundingBox, OCRResult, TextRegion

LOGGER = logging.getLogger(__name__)

#: Default prompt for document OCR. Kept terse to avoid filler text.
DEFAULT_PROMPT = (
    "请识别图片中的全部文字内容，按原文的版面顺序输出。"
    "只输出识别到的文字，不要翻译、不要解释、不要添加任何多余内容。"
    "如果是表格，请用 Markdown 表格输出。"
)

DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_IMAGE_FORMAT = "png"


def _b64_to_image_url(data: bytes, image_format: str = DEFAULT_IMAGE_FORMAT) -> str:
    return f"data:image/{image_format};base64," + base64.b64encode(data).decode("ascii")


def _extract_text(payload: Dict[str, Any]) -> str:
    """Pull the assistant message text out of an OpenAI-style response."""
    try:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        first = choices[0] or {}
        message = first.get("message") or first.get("delta") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Some providers return content as a list of parts.
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts)
    except Exception:
        return ""
    return ""


class MultimodalEngine(BaseEngine):
    """OCR via a multimodal LLM over an OpenAI-compatible HTTP API."""

    name = "multimodal"

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        # Set before anything can call engine_id(): SiliconFlowEngine subclasses
        # inherit this __init__ and rely on it to identify themselves.
        self._engine_id = normalise_engine(config.name)
        self.model_name = config.resolved_model or DEFAULT_MODEL
        # ``resolved_base_url`` already checks config -> OPENAI_BASE_URL. A bare
        # ``multimodal`` engine with neither must not silently fall back to
        # SiliconFlow (that would route a different provider's key to the wrong
        # endpoint); require an explicit URL instead.
        base = config.resolved_base_url
        if not base and self.engine_id() != "siliconflow":
            raise EngineNotAvailable(
                "no base URL for multimodal engine. Set OPENAI_BASE_URL "
                "(or base_url in the engine config)."
            )
        self.base_url = (base or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = config.resolved_api_key
        self.temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        self.timeout = float(getattr(config, "timeout", 120.0) or 120.0)
        self.max_tokens = getattr(config, "max_tokens", None)
        self.prompt = self._resolve_prompt(config)

    def _resolve_prompt(self, config: EngineConfig) -> str:
        prompt = getattr(config, "prompt", "") or ""
        path = (config.merged_extra() or {}).get("prompt_file")
        if path and not prompt:
            try:
                prompt = load_prompt(str(path)).strip()
            except OSError as exc:
                LOGGER.warning("could not read prompt_file %s: %s", path, exc)
        return prompt or DEFAULT_PROMPT

    def engine_id(self) -> str:
        return self._engine_id or self.name

    # -- implementation -----------------------------------------------------
    def _recognise_impl(self, page: PageImage) -> Dict[str, Any]:
        if not page.has_image:
            raise EngineNotAvailable(
                "multimodal engine needs a decoded image; decode the page first"
            )
        if not self.api_key:
            raise EngineNotAvailable(
                f"no API key for '{self.engine_id()}'. Set SILICONFLOW_API_KEY, "
                "OPENAI_API_KEY, or api_key in the engine config."
            )
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise EngineNotAvailable(
                "requests is required. Install with: pip install requests"
            ) from exc

        image_format = self.config.extra.get("image_format", DEFAULT_IMAGE_FORMAT)
        payload_bytes = self._image_bytes(page, image_format)
        if len(payload_bytes) > MAX_IMAGE_BYTES:
            payload_bytes = self._shrink(payload_bytes)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _b64_to_image_url(payload_bytes, image_format)},
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = int(self.max_tokens)

        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"request to {url} failed: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"{self.engine_id()} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"non-JSON response from {url}") from exc
        return {"text": _extract_text(data), "raw": data}

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _image_bytes(page: PageImage, image_format: str) -> bytes:
        """Encode ``page`` in the preferred format, falling back to PNG.

        JPEG is dramatically smaller than PNG for photographic scans, which
        matters because we base64 the payload inline in the request body.
        """
        fmt = (image_format or "png").lower()
        if fmt == "png":
            return page.to_png_bytes()
        if fmt in ("jpg", "jpeg"):
            try:
                import io

                buffer = io.BytesIO()
                page.pil_image.convert("RGB").save(buffer, format="JPEG", quality=85)
                return buffer.getvalue()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("jpeg encoding failed (%s); using PNG", exc)
                return page.to_png_bytes()
        return page.to_png_bytes()

    def _shrink(self, png: bytes) -> bytes:
        """Downscale an oversized image before uploading."""
        import io

        try:
            from PIL import Image
        except ImportError:
            LOGGER.warning("image too large (%d bytes) and Pillow is unavailable", len(png))
            return png
        with Image.open(io.BytesIO(png)) as image:
            width, height = image.size
            scale = min(1.0, ((MAX_IMAGE_BYTES / len(png)) ** 0.5) * 0.9)
            if scale <= 0.0:
                return png
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            resized = image.resize(new_size)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG", optimize=True)
            out = buffer.getvalue()
        if out and len(out) < len(png):
            LOGGER.info("shrunk image from %d to %d bytes", len(png), len(out))
            return out
        return png

    def _wrap(self, page: PageImage, raw: Dict[str, Any]) -> OCRResult:
        text = str(raw.get("text", "") or "")
        regions: List[TextRegion] = []
        if text.strip():
            regions.append(
                TextRegion(
                    text=text,
                    bbox=BoundingBox(
                        x1=0.0, y1=0.0, x2=float(page.width), y2=float(page.height)
                    ),
                    confidence=1.0,
                    engine=self.engine_id(),
                )
            )
        return OCRResult(
            text=text,
            regions=regions,
            engine=self.engine_id(),
            model=self.model_name,
            width=page.width,
            height=page.height,
        )


@register("multimodal")
def _multimodal_factory(config: EngineConfig) -> MultimodalEngine:
    return MultimodalEngine(config)


def parse_text_to_regions(text: str) -> List[TextRegion]:
    """Split plain text into line regions (no coordinates available)."""
    regions: List[TextRegion] = []
    for line in (text or "").splitlines():
        if line.strip():
            regions.append(TextRegion(text=line, engine="multimodal"))
    return regions


__all__ = ["DEFAULT_PROMPT", "MultimodalEngine", "parse_text_to_regions"]
