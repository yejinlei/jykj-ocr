# -*- coding: utf-8 -*-
"""Input loading: images, PDFs and URLs into :class:`PageImage` objects.

PDF rendering is tried in order of preference:
``pymupdf`` (fast, no system deps) -> ``pdf2image`` (needs Poppler) -> Pillow's
built-in PDF reader (needs Ghostscript). Failing fast with a clear message is
better than silently returning one blank page.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, List, Optional

from .base import InputError, PageImage

LOGGER = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
_PDF_EXTENSIONS = {".pdf"}
_URL_PREFIXES = ("http://", "https://")
_DATA_PREFIX = "data:"


def load(source: str, *, max_pages: Optional[int] = None, dpi: int = 200) -> List[PageImage]:
    """Load ``source`` into pages.

    ``source`` can be a local file path, an ``http(s)://`` URL, or a
    ``data:`` URI (e.g. ``data:image/png;base64,...``).
    """
    if not source or not str(source).strip():
        raise InputError("empty input path")
    source = str(source).strip()

    if source.lower().startswith(_URL_PREFIXES):
        source = _download(source)
    elif source.lower().startswith(_DATA_PREFIX):
        source = _decode_data_uri(source)

    if not os.path.isfile(source):
        raise InputError(f"input not found: {source}")

    if source.lower().endswith(tuple(_PDF_EXTENSIONS)) or _looks_like_pdf(source):
        return _load_pdf(source, max_pages=max_pages, dpi=dpi)
    if _looks_like_image(source):
        return [_load_image_file(source)]
    raise InputError(f"unsupported input type: {source}")


def _looks_like_pdf(path: str) -> bool:
    with open(path, "rb") as handle:
        return handle.read(4) == b"%PDF"


def _looks_like_image(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    return ext in _IMAGE_EXTENSIONS or _magic_bytes(path) is not None


_MAGIC_IMAGE_TYPES = {
    b"\x89PNG": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"GIF8": ".gif",
    b"BM": ".bmp",
}


def _magic_bytes(path: str) -> Optional[str]:
    """Sniff the real image type from file contents (extension may be missing)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(12)
    except OSError:
        return None
    if head.startswith(b"RIFF"):
        return ".webp" if head[8:12] == b"WEBP" else None
    for magic, ext in _MAGIC_IMAGE_TYPES.items():
        if head.startswith(magic):
            return ext
    return None


def _decode_data_uri(uri: str) -> str:
    """Decode a ``data:`` URI into a temp file.

    Accepts ``data:<type>[;base64],<payload>``. The extension is derived from
    the ``<type>`` (e.g. ``image/png`` -> ``.png``); if it is unrecognised the
    raw bytes are written to a ``.bin`` file so the caller can sniff by magic.
    """
    import base64

    try:
        header, payload = uri.split(",", 1)
    except ValueError:
        raise InputError("invalid data URI: missing payload") from None

    parts = header.split(";")
    media = parts[0].lower() if parts else ""  # e.g. "data:image/png"
    if not media.startswith(_DATA_PREFIX):
        raise InputError(f"invalid data URI header: {media!r}")

    # Derive extension from the media type.
    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
        "application/pdf": ".pdf",
    }
    mime = media[len(_DATA_PREFIX):] if media.startswith(_DATA_PREFIX) else ""
    suffix = ext_map.get(mime.lower()) or ".bin"

    is_base64 = any(p.strip().lower() == "base64" for p in parts[1:])
    try:
        raw = base64.b64decode(payload) if is_base64 else payload.encode("utf-8")
    except Exception as exc:
        raise InputError(f"failed to decode data URI: {exc}") from exc

    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(raw)
        handle.close()
        return handle.name
    except Exception as exc:
        raise InputError(f"failed to write data URI payload: {exc}") from exc


def _download(url: str) -> str:
    """Fetch ``url`` into a temp file, preserving the extension."""
    import urllib.parse
    import urllib.request

    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".bin"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            handle.write(response.read())
        handle.close()
        return handle.name
    except Exception as exc:
        raise InputError(f"failed to download {url}: {exc}") from exc


def _load_image_file(path: str) -> PageImage:
    """Decode an image file with Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise InputError(
            "Pillow is required to read images. Install with: pip install pillow"
        ) from exc
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.copy() if image.mode in ("RGB", "L") else image.convert("RGB")
        width, height = rgb.size
    except Exception as exc:
        raise InputError(f"failed to decode image {path}: {exc}") from exc
    return PageImage(pil_image=rgb, page=0, width=width, height=height, format="image")


def _load_pdf(path: str, *, max_pages: Optional[int], dpi: int) -> List[PageImage]:
    for renderer in (_render_with_pymupdf, _render_with_pdf2image, _render_with_pillow):
        try:
            pages = renderer(path, max_pages=max_pages, dpi=dpi)
        except ImportError:
            continue
        except Exception as exc:  # try the next renderer
            LOGGER.debug("%s failed on %s: %s", renderer.__name__, path, exc)
            continue
        if pages:
            return pages
    raise InputError(
        f"could not render PDF {path}. Install one of: pymupdf, "
        "pdf2image (with Poppler), or Pillow with Ghostscript support."
    )


def _render_with_pymupdf(path: str, *, max_pages: Optional[int], dpi: int) -> List[PageImage]:
    import fitz  # type: ignore

    pages: List[PageImage] = []
    with fitz.open(path) as doc:
        count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for index in range(count):
            pixmap = doc[index].get_pixmap(matrix=matrix)
            pages.append(
                PageImage(
                    data=pixmap.tobytes("png"),
                    page=index,
                    width=pixmap.width,
                    height=pixmap.height,
                    format="pdf",
                )
            )
    return pages


def _render_with_pdf2image(
    path: str, *, max_pages: Optional[int], dpi: int
) -> List[PageImage]:
    from pdf2image import convert_from_path  # type: ignore

    kwargs: dict = {"dpi": dpi}
    if max_pages is not None:
        kwargs["first_page"] = 1
        kwargs["last_page"] = max_pages
    images = convert_from_path(path, **kwargs)
    return [
        PageImage(pil_image=img, page=i, width=img.width, height=img.height, format="pdf")
        for i, img in enumerate(images)
    ]


def _render_with_pillow(
    path: str, *, max_pages: Optional[int], dpi: int
) -> List[PageImage]:
    from PIL import Image  # type: ignore

    pages: List[PageImage] = []
    with Image.open(path) as image:
        count = getattr(image, "n_frames", 1)
        if max_pages is not None:
            count = min(count, max_pages)
        for index in range(count):
            image.seek(index)
            rgb = image.copy() if image.mode in ("RGB", "L") else image.convert("RGB")
            rgb = rgb.resize((int(rgb.width * dpi / 72.0), int(rgb.height * dpi / 72.0)))
            pages.append(
                PageImage(
                    pil_image=rgb,
                    page=index,
                    width=rgb.width,
                    height=rgb.height,
                    format="pdf",
                )
            )
    return pages


def attach_pil(page: PageImage) -> PageImage:
    """Ensure ``page`` carries a PIL image, decoding ``data`` if necessary."""
    if page.pil_image is not None:
        return page
    if page.data is None:
        raise InputError("page has neither a PIL image nor raw data")
    try:
        import io

        from PIL import Image
    except ImportError as exc:
        raise InputError(
            "Pillow is required to decode page images. Install with: pip install pillow"
        ) from exc
    with Image.open(io.BytesIO(page.data)) as image:
        rgb = image.copy() if image.mode in ("RGB", "L") else image.convert("RGB")
    return PageImage(
        pil_image=rgb,
        data=page.data,
        page=page.page,
        width=rgb.width,
        height=rgb.height,
        format=page.format,
    )


__all__ = ["attach_pil", "load"]
