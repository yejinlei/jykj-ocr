# -*- coding: utf-8 -*-
"""Tests for engine/inputs.py — URL downloads with extension-less temp files."""

from __future__ import annotations

import io

from jykj_ocr.engine import inputs


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(buf, format="JPEG")
    return buf.getvalue()


class TestMagicSniffing:
    def test_png_without_known_extension_is_recognised(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(_png_bytes())
        assert inputs._magic_bytes(str(path)) == ".png"
        assert inputs._looks_like_image(str(path)) is True

    def test_jpeg_without_known_extension_is_recognised(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(_jpeg_bytes())
        assert inputs._magic_bytes(str(path)) == ".jpg"
        assert inputs._looks_like_image(str(path)) is True

    def test_riff_webp_recognised_but_plain_riff_is_not(self, tmp_path):
        webp = tmp_path / "a.bin"
        webp.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
        assert inputs._magic_bytes(str(webp)) == ".webp"
        wav = tmp_path / "b.bin"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        assert inputs._magic_bytes(str(wav)) is None

    def test_unknown_bytes_are_not_an_image(self, tmp_path):
        path = tmp_path / "junk.bin"
        path.write_bytes(b"not an image at all")
        assert inputs._magic_bytes(str(path)) is None
        assert inputs._looks_like_image(str(path)) is False

    def test_known_extension_still_passes_without_magic(self, tmp_path):
        # Legacy behaviour: a .png-named file is accepted by extension alone.
        path = tmp_path / "plain.png"
        path.write_bytes(b"garbage")
        assert inputs._looks_like_image(str(path)) is True

    def test_missing_file_is_not_an_image(self, tmp_path):
        assert inputs._looks_like_image(str(tmp_path / "nope.bin")) is False


class TestLoadFromDownloadedUrl:
    def test_url_download_with_bin_suffix_loads_as_image(self, monkeypatch):
        # Simulate POST /ocr/text with a URL whose path has no image extension:
        # _download() lands on a .bin temp file; load() must sniff the content.
        called = {}

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout=None):
            called["url"] = url
            return _Resp(_png_bytes())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        pages = inputs.load("https://example.com/img/u=12345&fm=3074&f=PNG")
        assert called["url"].startswith("https://example.com/")
        assert len(pages) == 1
        assert pages[0].width == 8 and pages[0].height == 6
