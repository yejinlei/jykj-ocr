# -*- coding: utf-8 -*-
"""Core data models for OCR results.

Kept dependency-free: ``pydantic`` is optional but recommended. When it is
absent, we fall back to a tiny stdlib-only ``_PydanticBase`` so the module always
imports (useful for the offline container image).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - exercised implicitly by whichever path is taken
    from pydantic import BaseModel as _PydanticBase

    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    _HAS_PYDANTIC = False

    class _PydanticBase:  # type: ignore[no-redef]
        """Minimal stand-in supporting the subset of pydantic we use."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self, **_: Any) -> Dict[str, Any]:
            return dict(self.__dict__)

        def dict(self, **_: Any) -> Dict[str, Any]:  # noqa: D401
            return self.model_dump()


class Point(_PydanticBase):
    """A point in image pixel coordinates."""

    x: float = 0.0
    y: float = 0.0


class BoundingBox(_PydanticBase):
    """Axis-aligned box with a confidence score in ``[0, 1]``."""

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    def area(self) -> float:
        return self.width * self.height

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
        }


def _norm_confidence(value: Any) -> float:
    """Coerce a confidence value into ``[0, 1]`` regardless of source scale."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(conf):
        return 0.0
    if conf < 0:
        return 0.0
    if conf > 1.0:
        # Some engines report 0-100 percentages.
        if conf <= 100.0:
            return conf / 100.0
        return 1.0
    return conf


#: Sentinel distinguishing "caller did not pass a confidence" from a real 1.0,
#: so that a bbox carrying its own score is not silently overwritten.
_UNSET: Any = object()


class TextRegion(_PydanticBase):
    """A single recognised text block."""

    text: str = ""
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    engine: str = ""

    @classmethod
    def from_parts(
        cls,
        text: str,
        bbox: Any = None,
        confidence: Any = _UNSET,
        engine: str = "",
    ) -> "TextRegion":
        """Build a region from loosely-typed engine output.

        ``bbox`` may be a dict, a list of 4 corner points ``[[x, y], ...]``, or a
        flat tuple of coordinates. Anything unrecognisable yields a region with no
        box rather than raising — OCR must not fail the whole document because one
        engine returned a weird shape.

        A confidence supplied by the caller wins; otherwise the box keeps the
        score the engine already attached to it (no silent upgrade to 1.0).
        """
        box: Optional[BoundingBox] = None
        if bbox is not None:
            if isinstance(bbox, dict):
                try:
                    box = BoundingBox(
                        x1=float(bbox.get("x1", bbox.get("xmin", 0))),
                        y1=float(bbox.get("y1", bbox.get("ymin", 0))),
                        x2=float(bbox.get("x2", bbox.get("xmax", 0))),
                        y2=float(bbox.get("y2", bbox.get("ymax", 0))),
                        confidence=_norm_confidence(
                            bbox.get("confidence", bbox.get("score", 1.0))
                        ),
                    )
                except (TypeError, ValueError):
                    box = None
            elif isinstance(bbox, (list, tuple)):
                nums: List[float] = []
                for item in bbox:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            nums.extend([float(item[0]), float(item[1])])
                        except (TypeError, ValueError):
                            continue
                    elif isinstance(item, (int, float)):
                        nums.append(float(item))
                if len(nums) >= 8:
                    xs = nums[0::2]
                    ys = nums[1::2]
                    box = BoundingBox(
                        x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)
                    )
                elif len(nums) == 4:
                    box = BoundingBox(x1=nums[0], y1=nums[1], x2=nums[2], y2=nums[3])
        if box is not None and confidence is not _UNSET:
            box.confidence = _norm_confidence(confidence)
        if confidence is _UNSET:
            effective = box.confidence if box is not None else 1.0
        else:
            effective = _norm_confidence(confidence)
        return cls(
            text=str(text or ""),
            bbox=box,
            confidence=effective,
            engine=engine,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.as_dict() if self.bbox else None,
            "confidence": round(self.confidence, 4),
            "engine": self.engine,
        }


class OCRResult(_PydanticBase):
    """Aggregated output of one ``recognise()`` call."""

    text: str = ""
    regions: List[TextRegion] = []
    engine: str = ""
    model: str = ""
    elapsed_ms: int = 0
    width: int = 0
    height: int = 0

    @property
    def ok(self) -> bool:
        """True when recognition produced any usable text."""
        return bool(self.text.strip()) or bool(self.regions)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "engine": self.engine,
            "model": self.model,
            "elapsed_ms": self.elapsed_ms,
            "width": self.width,
            "height": self.height,
            "region_count": len(self.regions),
            "regions": [r.as_dict() for r in self.regions],
        }

    def to_markdown(self) -> str:
        """Render regions sorted top-to-bottom, left-to-right."""
        ordered = sorted(
            (r for r in self.regions if r.text.strip()),
            key=lambda r: (
                r.bbox.y1 if r.bbox else float("inf"),
                r.bbox.x1 if r.bbox else float("inf"),
            ),
        )
        lines = [r.text for r in ordered if r.text.strip()]
        if lines:
            return "\n".join(lines)
        return self.text


class EmptyResultError(Exception):
    """Raised when an engine returned no text and ``strict`` mode is enabled."""


__all__ = [
    "BoundingBox",
    "EmptyResultError",
    "OCRResult",
    "Point",
    "TextRegion",
]
