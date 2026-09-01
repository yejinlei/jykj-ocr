# -*- coding: utf-8 -*-
"""Local RapidOCR engine (ONNX runtime) — offline, no API key required.

Supports both RapidOCR 1.x and 2.x return shapes:

- 2.x: ``engine(img)`` -> dict with ``txts`` / ``boxes`` / ``scores``.
- 1.x: ``engine(img)`` -> ``(boxes, txts, scores)`` or ``None``.

Both are normalised into an :class:`OCRResult`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import EngineConfig
from ..engine.base import BaseEngine, EngineNotAvailable, PageImage, register
from ..models import OCRResult, TextRegion

LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGS = ("ch", "en", "chinese", "korean", "japanese")

_ENGINE_CACHE: Dict[Tuple[str, str], Any] = {}


def _load_rapidocr_class() -> Tuple[Any, str]:
    """Import a RapidOCR class from whichever 1.x/2.x variant is installed."""
    for module_name, class_name in (
        ("rapidocr_onnxruntime", "RapidOCR"),
        ("rapidocr", "RapidOCR"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name), module_name
        except ImportError:
            continue
    raise EngineNotAvailable(
        "RapidOCR is not installed. Install with: pip install rapidocr-onnxruntime"
    )


def _normalise_lang(lang: str) -> str:
    value = (lang or "ch").strip().lower()
    if value in ("ch", "chinese", "zh"):
        return "ch"
    if value in ("en", "english"):
        return "en"
    if value in ("korean", "ko"):
        return "korean"
    if value in ("japanese", "ja"):
        return "japanese"
    return value


def _get_engine(config: EngineConfig) -> Any:
    """Instantiate (or reuse) a RapidOCR engine for this config."""
    lang = _normalise_lang(config.lang)
    cache_key = (lang, str((config.extra or {}).get("model_dir", "")))
    if cache_key in _ENGINE_CACHE:
        return _ENGINE_CACHE[cache_key]

    cls, module_name = _load_rapidocr_class()
    extra = dict(config.extra or {})
    model_name = extra.get("model_name") or ("ch" if lang == "ch" else lang)

    LOGGER.info("loading RapidOCR from %s (model=%s)", module_name, model_name)
    try:
        instance = cls(model_name_or_path=model_name)
    except TypeError:
        # 1.x style: default construction.
        instance = cls()
    except Exception as exc:
        raise EngineNotAvailable(
            f"failed to initialise RapidOCR ({module_name}): {exc}"
        ) from exc

    _ENGINE_CACHE[cache_key] = instance
    return instance


def _run(
    engine: Any, img: Any
) -> Tuple[Optional[Sequence[Any]], Optional[Sequence[str]], Optional[Sequence[float]]]:
    """Call RapidOCR and return ``(boxes, texts, scores)`` regardless of version.

    RapidOCR releases return several shapes; we normalise them all:

    - ``1.4.x`` (``rapidocr-onnxruntime``): ``(results, elapsed)`` — a 2-tuple
      where ``results`` is a list of ``[box, text, score]`` triples and
      ``elapsed`` is a list of timing floats. The timing element is what gets
      mistaken for text if one naively indexes ``[1]``.
    - ``1.x`` tuple ``(boxes, txts, scores)`` or ``None``.
    - ``2.x`` dict with ``txts`` / ``boxes`` / ``scores``.
    """
    result = engine(img)
    if result is None:
        return None, None, None

    # 2.x dict shape.
    if isinstance(result, dict):
        texts = result.get("txts") or []
        boxes = result.get("boxes") or result.get("boxes_rot") or []
        scores = list(result.get("scores") or [])
        if len(scores) < len(texts):
            scores.extend([1.0] * (len(texts) - len(scores)))
        return boxes, texts, scores

    # Tuple/list shape. Distinguish 1.4.x (results, elapsed) from
    # 1.x (boxes, txts, scores) by inspecting the first element: the
    # results list holds ``[box, text, score]`` triples; a bare boxes list
    # holds ``[[x,y],...]`` corner arrays.
    if isinstance(result, (tuple, list)):
        first = result[0] if len(result) > 0 else None
        if _looks_like_results_list(first):
            triples = first or []
            boxes = [t[0] for t in triples]
            texts = [t[1] for t in triples]
            scores = [t[2] if len(t) > 2 else 1.0 for t in triples]
            return boxes, texts, scores

        # 1.x three-element (boxes, txts, scores) shape.
        boxes = result[0]
        texts = result[1]
        scores = result[2] if len(result) > 2 else [1.0] * len(texts or [])
        return boxes, texts, scores

    return None, None, None


def _looks_like_results_list(part: Any) -> bool:
    """True when ``part`` is a list of ``[box, text, score]`` triples.

    The 1.4.x rapidocr-onnxruntime return wraps real results this way, with the
    *second* top-level element being timing floats — so a length-2 result that
    passes through the 1.x branch would index the timing list as ``txts``.
    """
    if not isinstance(part, (list, tuple)) or not part:
        return False
    first = part[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return False
    # A real result triple is [box, text, score]; element 1 is the text string.
    return isinstance(first[1], str)


class RapidOCREngine(BaseEngine):
    """Local offline OCR using RapidOCR (ONNX runtime)."""

    name = "rapidocr"
    model_name = "rapidocr-onnxruntime"

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self.lang = _normalise_lang(config.lang)

    def engine_id(self) -> str:
        return "rapidocr"

    def _recognise_impl(self, page: PageImage) -> Dict[str, Any]:
        if not page.has_image:
            raise EngineNotAvailable("rapidocr engine needs a decoded image")
        instance = _get_engine(self.config)

        image = page.pil_image
        if (self.config.extra or {}).get("use_numpy"):
            import numpy as np  # type: ignore

            image = np.array(page.pil_image)

        boxes, texts, scores = _run(instance, image)
        if not texts:
            return {"text": "", "regions": []}

        region_list: List[Dict[str, Any]] = []
        for index, text in enumerate(texts):
            value = str(text or "")
            if not value.strip():
                continue
            box: Any = boxes[index] if boxes and index < len(boxes) else None
            score = scores[index] if scores and index < len(scores) else 1.0
            region_list.append(
                {
                    "text": value,
                    "bbox": _bbox_from_corners(box),
                    "confidence": float(score) if score is not None else 1.0,
                }
            )

        joined = "\n".join(r["text"] for r in region_list if r["text"])
        return {"text": joined, "regions": region_list}

    def _wrap(self, page: PageImage, raw: Dict[str, Any]) -> OCRResult:
        regions: List[TextRegion] = []
        for item in raw.get("regions") or []:
            regions.append(
                TextRegion.from_parts(
                    item.get("text", ""),
                    bbox=item.get("bbox"),
                    confidence=item.get("confidence", 1.0),
                    engine=self.engine_id(),
                )
            )
        text = str(raw.get("text", "") or "")
        if not text and regions:
            text = "\n".join(r.text for r in regions if r.text)
        return OCRResult(
            text=text,
            regions=regions,
            engine=self.engine_id(),
            model=self.model_name,
            width=page.width,
            height=page.height,
        )


def _bbox_from_corners(corners: Any) -> Optional[Dict[str, float]]:
    """Convert 4 ``[x, y]`` corners into an axis-aligned box dict."""
    if not isinstance(corners, (list, tuple)) or len(corners) < 4:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for corner in corners:
        try:
            xs.append(float(corner[0]))
            ys.append(float(corner[1]))
        except (TypeError, ValueError, IndexError):
            continue
    if len(xs) < 4 or len(ys) < 4:
        return None
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


@register("rapidocr")
def _rapidocr_factory(config: EngineConfig) -> RapidOCREngine:
    return RapidOCREngine(config or EngineConfig(name="rapidocr"))


def clear_cache() -> None:
    """Drop cached RapidOCR instances (useful for tests)."""
    _ENGINE_CACHE.clear()


__all__ = ["RapidOCREngine", "SUPPORTED_LANGS", "clear_cache"]
