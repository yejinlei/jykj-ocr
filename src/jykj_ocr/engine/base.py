# -*- coding: utf-8 -*-
"""Engine contract, registry, and input loading.

An "engine" is anything exposing ``recognise(page) -> OCRResult`` where ``page``
is a :class:`PageImage`.

Engines are registered lazily: importing :mod:`jykj_ocr` never imports PIL,
OpenCV, ``rapidocr`` or ``openai``. They are imported only when a caller
actually requests an engine that needs them.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..config import EngineConfig
from ..models import OCRResult

LOGGER = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # safety cap for base64 payloads to APIs

Registry = Dict[str, Callable[[EngineConfig], "BaseEngine"]]
_REGISTRY: Registry = {}
_LAZY_IMPORTS: Dict[str, Callable[[], None]] = {}


@dataclass
class PageImage:
    """One page of a document, kept in the cheapest usable form."""

    pil_image: Any = None
    data: Optional[bytes] = None
    page: int = 0
    width: int = 0
    height: int = 0
    format: str = ""

    def __post_init__(self) -> None:
        if self.pil_image is not None and not self.width:
            try:
                self.width, self.height = self.pil_image.size
            except Exception:  # pragma: no cover
                pass

    @property
    def has_image(self) -> bool:
        return self.pil_image is not None

    def to_png_bytes(self) -> bytes:
        """Encode the page as PNG (the format APIs handle most reliably)."""
        import io

        if self.pil_image is None:
            if self.data:
                return self.data
            raise ValueError("PageImage has no image data")
        buffer = io.BytesIO()
        self.pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def encoded_png(self) -> str:
        return base64.b64encode(self.to_png_bytes()).decode("ascii")


class EngineNotAvailable(Exception):
    """Raised when an engine cannot be imported or resolved."""


class BaseEngine:
    """Abstract base for all OCR engines.

    Subclasses implement :meth:`_recognise_impl` and return a raw provider
    payload; this base class wraps the call with engine/model tagging and page
    dimensions so every engine emits a uniform :class:`OCRResult`.
    """

    name: str = "base"

    def __init__(self, config: EngineConfig) -> None:
        if config is None:
            raise ValueError("engine requires a config")
        self.config = config

    # -- public API ---------------------------------------------------------
    def recognise(self, page: PageImage) -> OCRResult:
        """Run recognition on ``page`` and return a normalised result."""
        try:
            raw = self._recognise_impl(page)
        except EngineNotAvailable:
            raise
        except Exception as exc:
            raise EngineError(
                f"{self.engine_id()} failed: {exc}", engine=self.engine_id()
            ) from exc
        return self._wrap(page, raw)

    def supports(self, page: PageImage) -> bool:
        """Whether this engine can handle ``page`` at all."""
        return page.has_image

    def engine_id(self) -> str:
        return self.name

    # -- overrides ----------------------------------------------------------
    def _recognise_impl(self, page: PageImage) -> Dict[str, Any]:
        raise NotImplementedError

    def _wrap(self, page: PageImage, raw: Dict[str, Any]) -> OCRResult:
        """Convert a raw provider payload into an :class:`OCRResult`.

        Subclasses that produce structured regions override this.
        """
        text = str(raw.get("text", "") or "")
        return OCRResult(
            text=text,
            regions=[],
            engine=self.engine_id(),
            model=getattr(self, "model_name", ""),
            width=page.width,
            height=page.height,
        )


class EngineError(Exception):
    """Raised when an engine call fails."""

    def __init__(self, message: str, *, engine: str = "") -> None:
        super().__init__(message)
        self.engine = engine


# -- registry ---------------------------------------------------------------
def register(name: str) -> Callable[[Callable[[EngineConfig], BaseEngine]], Callable[[EngineConfig], BaseEngine]]:
    """Register an engine factory by ``name``.

    Used as a parameterised decorator::

        @register("multimodal")
        def _multimodal_factory(config: EngineConfig) -> BaseEngine:
            ...
    """

    def decorator(
        fn: Callable[[EngineConfig], BaseEngine]
    ) -> Callable[[EngineConfig], BaseEngine]:
        _REGISTRY[name] = fn
        return fn

    return decorator


def register_lazy(name: str, importer: Callable[[], None]) -> None:
    """Record a deferred module import for ``name``."""
    _LAZY_IMPORTS[name] = importer


def available_engines() -> List[str]:
    """Engines known to the registry (registered plus deferred)."""
    return sorted(set(_REGISTRY) | set(_LAZY_IMPORTS))


def create_engine(name: str, config: Optional[EngineConfig] = None) -> BaseEngine:
    """Instantiate an engine by name, importing it lazily if needed."""
    key = (name or "").strip().lower()
    if not key:
        raise EngineNotAvailable("no engine name given")

    if key not in _REGISTRY and key in _LAZY_IMPORTS:
        LOGGER.debug("importing engine module for %s", key)
        _LAZY_IMPORTS[key]()

    factory = _REGISTRY.get(key)
    if factory is None:
        raise EngineNotAvailable(
            f"unknown engine '{name}'. Available: "
            f"{', '.join(available_engines()) or 'none'}"
        )
    return factory(config or EngineConfig(name=key))


def describe_engines() -> Dict[str, str]:
    """Human-readable engine descriptions for ``--list-engines``."""
    return {
        "rapidocr": "本地 RapidOCR (ONNX)，无需 API key，离线可用",
        "siliconflow": "硅基流动多模态 OCR 大模型 (DeepSeek-OCR)",
        "multimodal": "OpenAI 兼容多模态端点（其他平台通用）",
    }


# -- input loading ----------------------------------------------------------
class InputError(Exception):
    """Raised when an input path cannot be loaded."""


def load_image(source: str) -> List[PageImage]:
    """Load one or more pages from an image, PDF, or URL."""
    from . import inputs

    return inputs.load(source)


def content_type_for(name: str, data: Optional[bytes] = None) -> str:
    """Best-effort MIME type detection."""
    if data:
        if data[:4] == b"%PDF":
            return "application/pdf"
    guess, _ = mimetypes.guess_type(name or "")
    return guess or "application/octet-stream"


__all__ = [
    "BaseEngine",
    "EngineError",
    "EngineNotAvailable",
    "InputError",
    "MAX_IMAGE_BYTES",
    "PageImage",
    "available_engines",
    "content_type_for",
    "create_engine",
    "describe_engines",
    "load_image",
    "register",
    "register_lazy",
]
