# -*- coding: utf-8 -*-
"""Tests for the core data models."""

from __future__ import annotations

import math

import pytest

from jykj_ocr.models import BoundingBox, OCRResult, TextRegion, _norm_confidence


class TestNormConfidence:
    def test_none_and_garbage(self):
        assert _norm_confidence(None) == 0.0
        assert _norm_confidence("abc") == 0.0
        assert _norm_confidence(math.nan) == 0.0

    def test_negative_clamps(self):
        assert _norm_confidence(-0.5) == 0.0

    def test_unit_scale_passthrough(self):
        assert _norm_confidence(0.5) == pytest.approx(0.5)
        assert _norm_confidence(1.0) == 1.0

    def test_percent_scale_is_divided(self):
        assert _norm_confidence(95) == pytest.approx(0.95)
        assert _norm_confidence(0.0) == 0.0

    def test_above_hundred_clamps_to_one(self):
        assert _norm_confidence(150) == 1.0


class TestTextRegion:
    def test_from_dict_bbox(self):
        r = TextRegion.from_parts(
            "hi", bbox={"x1": 1, "y1": 2, "x2": 10, "y2": 9, "score": 0.88}, engine="e"
        )
        assert r.bbox is not None
        assert (r.bbox.x1, r.bbox.x2) == (1.0, 10.0)
        assert r.confidence == pytest.approx(0.88)

    def test_from_four_corners(self):
        corners = [[0, 0], [50, 0], [50, 20], [0, 20]]
        r = TextRegion.from_parts("x", bbox=corners, confidence=0.7)
        assert r.bbox.as_dict()["x1"] == 0.0
        assert r.bbox.as_dict()["y2"] == 20.0

    def test_garbage_bbox_does_not_raise(self):
        for bad in [object(), "not-a-box", [1, 2, 3], [[], [], [], []]]:
            r = TextRegion.from_parts("text", bbox=bad)
            assert r.text == "text"

    def test_no_bbox(self):
        r = TextRegion.from_parts("text", bbox=None)
        assert r.bbox is None
        assert r.text == "text"

    def test_flat_eight_numbers(self):
        r = TextRegion.from_parts("t", bbox=(1, 2, 9, 2, 9, 8, 1, 8))
        assert r.bbox is not None
        assert r.bbox.width == 8.0 and r.bbox.height == 6.0


class TestOCRResult:
    def test_ok_flag(self):
        assert OCRResult(text="", regions=[]).ok is False
        assert OCRResult(text="  ").ok is False
        assert OCRResult(text="hi").ok is True
        assert OCRResult(text="", regions=[TextRegion(text="a")]).ok is True

    def test_to_markdown_sorts_top_left_first(self):
        result = OCRResult(
            text="",
            regions=[
                TextRegion(text="bottom", bbox=BoundingBox(x1=0, y1=100, x2=50, y2=110)),
                TextRegion(text="right-top", bbox=BoundingBox(x1=50, y1=0, x2=100, y2=10)),
                TextRegion(text="left-top", bbox=BoundingBox(x1=0, y1=0, x2=40, y2=10)),
                TextRegion(text="", bbox=BoundingBox(x1=0, y1=50, x2=10, y2=60)),
            ],
        )
        assert result.to_markdown() == "left-top\nright-top\nbottom"

    def test_to_markdown_falls_back_to_text(self):
        assert OCRResult(text="plain").to_markdown() == "plain"

    def test_as_dict_round_trip(self):
        d = OCRResult(
            text="t",
            engine="rapidocr",
            model="m",
            width=10,
            height=20,
            regions=[TextRegion(text="a", confidence=0.5)],
        ).as_dict()
        assert d["engine"] == "rapidocr"
        assert d["region_count"] == 1
        assert d["regions"][0]["confidence"] == 0.5
        assert isinstance(d, dict)

    def test_bounding_box_properties(self):
        box = BoundingBox(x1=1, y1=2, x2=6, y2=8)
        assert box.width == 5 and box.height == 6
        assert box.area() == 30
        assert box.center.x == 3.5 and box.center.y == 5.0
        assert BoundingBox(x1=6, y1=2, x2=1, y2=8).width == 0.0
