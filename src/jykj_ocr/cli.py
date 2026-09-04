# -*- coding: utf-8 -*-
"""Command-line entry point.

    python -m jykj_ocr image.png --engine siliconflow
    python -m jykj_ocr doc.pdf --engine rapidocr --format markdown -o out.md
    python -m jykj_ocr --list-engines
    python -m jykj_ocr serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .engine import describe_engines
from .engine.registry import STRATEGY_PRESETS

LOG = logging.getLogger("jykj_ocr.cli")


def _load_dotenv(path: str = ".env") -> None:
    """Load a ``.env`` file into the environment (best-effort, stdlib only).

    Existing environment variables always win, so this never overrides an
    explicit ``SILICONFLOW_API_KEY`` export.
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        LOG.warning("could not read %s: %s", path, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jykj_ocr",
        description="多引擎 OCR：本地 RapidOCR 与多模态 OCR 大模型（硅基流动等）",
    )
    parser.add_argument("source", nargs="?", help="图片或 PDF 路径，或 http(s) URL")
    parser.add_argument("-c", "--config", help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument(
        "--engine",
        help="指定引擎，忽略配置中的策略链；如 rapidocr / siliconflow / multimodal",
    )
    parser.add_argument(
        "--strategy-name",
        choices=STRATEGY_PRESETS,
        help=(
            "按命名预设整体切换策略（仅本次运行生效）："
            "local/vl 仅本地/仅 VL;seq* 顺序回退;cascade* 无重试直接降级;bestof* 多引擎择优"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="输出格式（默认 text）",
    )
    parser.add_argument("-o", "--output", help="输出文件；缺省打印到 stdout")
    parser.add_argument("--max-pages", type=int, help="最多处理的 PDF 页数")
    parser.add_argument("--dpi", type=int, default=200, help="PDF 渲染 DPI（默认 200）")
    parser.add_argument("--list-engines", action="store_true", help="列出可用引擎后退出")
    parser.add_argument("-v", "--verbose", action="store_true", help="开启调试日志")

    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.getenv("JYKJ_OCR_PORT", "8000")))
    serve.add_argument("--config", help="配置文件路径")
    return parser


def _write_output(text: str, output: Optional[str]) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)
        LOG.info("wrote %s (%d chars)", output, len(text))
        return
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


def _render(result: Any, fmt: str) -> str:
    """Serialise one OCRResult according to ``fmt``."""
    if fmt == "json":
        return json.dumps(result.as_dict(), ensure_ascii=False, indent=2)
    return result.to_markdown()


def _serve(args: argparse.Namespace) -> int:
    if args.config:
        os.environ["JYKJ_OCR_CONFIG"] = args.config
    elif "JYKJ_OCR_CONFIG" not in os.environ:
        os.environ["JYKJ_OCR_CONFIG"] = os.path.join("config", "config.yaml")
    try:
        import uvicorn
    except ImportError:
        print(
            "缺少 FastAPI 依赖。请运行: pip install fastapi uvicorn python-multipart",
            file=sys.stderr,
        )
        return 2
    uvicorn.run("jykj_ocr.server:app", host=args.host, port=args.port)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_engines:
        for name, description in describe_engines().items():
            print(f"{name:14s} {description}")
        return 0

    if args.command == "serve":
        return _serve(args)

    if not args.source:
        build_parser().print_help()
        return 2

    from . import ocr

    results = ocr(
        args.source,
        engine=args.engine,
        config_path=args.config,
        max_pages=args.max_pages,
        dpi=args.dpi,
        strategy_name=args.strategy_name,
    )
    if not results:
        print("未识别到任何内容", file=sys.stderr)
        return 1

    text = "\n\n".join(
        chunk for chunk in (_render(r, args.format) for r in results if r.ok) if chunk
    )
    if not text:
        print("未识别到任何内容", file=sys.stderr)
        return 1

    _write_output(text, args.output)
    LOG.info(
        "完成：%d 页，引擎 %s，耗时 %dms",
        len(results),
        results[0].engine,
        sum(r.elapsed_ms for r in results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
