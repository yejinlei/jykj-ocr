# -*- coding: utf-8 -*-
"""Tests for line-overlap detection and reading-order rebuild (窜行)."""

from __future__ import annotations

from jykj_ocr.models import (
    BoundingBox,
    OCRResult,
    TextRegion,
    detect_line_overlap,
    rebuild_text_from_regions,
    sort_regions_by_position,
)


def _region(text: str, x1: float, y1: float, x2: float, y2: float) -> TextRegion:
    return TextRegion(text=text, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))


def _result(*regions: TextRegion) -> OCRResult:
    return OCRResult(text=" ".join(r.text for r in regions), regions=list(regions))


class TestSortRegionsByPosition:
    def test_out_of_order_regions_are_reordered(self):
        regions = [
            _region("second", 0, 30, 60, 45),
            _region("first", 0, 0, 60, 15),
        ]
        assert [r.text for r in sort_regions_by_position(regions)] == ["first", "second"]

    def test_within_line_sorted_left_to_right(self):
        regions = [
            _region("bar", 80, 0, 140, 15),
            _region("foo", 10, 0, 70, 15),
        ]
        assert [r.text for r in sort_regions_by_position(regions)] == ["foo", "bar"]

    def test_bbox_less_regions_appended_last(self):
        loose = TextRegion(text="loose")
        boxed = _region("boxed", 0, 5, 20, 20)
        ordered = sort_regions_by_position([loose, boxed])
        assert [r.text for r in ordered] == ["boxed", "loose"]


class TestRebuildTextFromRegions:
    def test_joins_in_reading_order(self):
        result = _result(
            _region("B2", 0, 30, 50, 45),
            _region("A2", 60, 0, 110, 15),
            _region("A1", 0, 0, 50, 15),
        )
        assert rebuild_text_from_regions(result) == "A1\nA2\nB2"

    def test_single_region_keeps_original_text(self):
        result = OCRResult(text="original blob", regions=[_region("only", 0, 0, 10, 10)])
        assert rebuild_text_from_regions(result) == "original blob"

    def test_no_regions_keeps_text(self):
        assert rebuild_text_from_regions(OCRResult(text="plain")) == "plain"


class TestDetectLineOverlap:
    def test_clean_layout_is_not_garbled(self):
        result = _result(
            _region("line one", 0, 0, 100, 12),
            _region("line two", 0, 20, 90, 32),
            _region("line three", 0, 40, 95, 52),
        )
        assert detect_line_overlap(result) is False

    def test_merged_row_detected_by_aspect_ratio(self):
        # A box spanning several text rows: aspect ratio > 12 AND width > 8x median height.
        result = _result(
            _region("short", 0, 0, 40, 10),
            _region("merged", 0, 0, 300, 10),
        )
        assert detect_line_overlap(result) is True

    def test_colliding_boxes_detected(self):
        a = _region("a", 0, 0, 50, 20)
        b = _region("b", 10, 5, 60, 25)  # overlaps a on both axes
        assert detect_line_overlap(_result(a, b)) is True

    def test_side_by_side_boxes_not_flagged(self):
        a = _region("a", 0, 0, 50, 20)
        b = _region("b", 60, 0, 110, 20)  # same line, no x overlap
        assert detect_line_overlap(_result(a, b)) is False

    def test_region_less_result_never_flagged(self):
        assert detect_line_overlap(OCRResult(text="vl output")) is False

    def test_single_box_not_flagged(self):
        assert detect_line_overlap(_result(_region("one", 0, 0, 10, 10))) is False
