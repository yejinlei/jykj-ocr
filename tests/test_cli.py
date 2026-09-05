# -*- coding: utf-8 -*-
"""CLI 解析回归测试。

cli.py 曾经完全零测试覆盖,导致一个让 CLI 彻底不可用的 bug 存活了很久:
``build_parser()`` 同时定义了位置参数 ``source``(``nargs="?"``)和名为 ``serve``
的子命令(argparse ``add_subparsers``)。两者争抢同一份 token 流,于是
``jykj_ocr image.png --engine rapidocr`` 直接退出码 2:

    error: argument source: invalid choice: 'image.png' (choose from 'serve')

修法是预处理 argv 把 ``serve`` 关键字摘出来。这里锁住该行为,防止有人为了"更像
子命令"而改回 argparse 子解析。
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from jykj_ocr.cli import _split_serve_command, build_parser, main

# 离线用例:OCR 一律打桩,不发网络请求,不加载任何真实引擎。
IMAGE = "tests/兰亭序.jpeg"
FAKE_RESULT = {"text": "识别到的文本", "engine": "rapidocr", "regions": []}


class _FakeResult:
    ok = True
    elapsed_ms = 12
    engine = FAKE_RESULT["engine"]

    def to_markdown(self):
        return FAKE_RESULT["text"]

    def as_dict(self):
        return FAKE_RESULT


@pytest.fixture
def fake_ocr(monkeypatch):
    """把顶层 ocr() 换成记录参数的桩,返回一条假结果。"""
    import jykj_ocr

    calls = []

    def _fake_ocr(source, **kwargs):
        calls.append({"source": source, **kwargs})
        return [_FakeResult()]

    monkeypatch.setattr(jykj_ocr, "ocr", _fake_ocr)
    return calls


@pytest.fixture
def fake_serve(monkeypatch):
    """拦住真正的 uvicorn.run,只记录解析出来的参数。"""
    captured = []

    def _fake_serve(args):
        captured.append(args)
        return 0

    monkeypatch.setattr("jykj_ocr.cli._serve", _fake_serve)
    return captured


# ---------------------------------------------------------------- serve 关键字


def test_split_serve_command_flags_serve():
    serving, argv = _split_serve_command(["serve", "--port", "8010"])
    assert serving is True
    assert argv == ["--port", "8010"]


def test_split_serve_command_ignores_ocr_argv():
    serving, argv = _split_serve_command([IMAGE, "--engine", "rapidocr"])
    assert serving is False
    assert argv == [IMAGE, "--engine", "rapidocr"]


def test_split_serve_command_none_means_sys_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "serve", "--port", "9000"])
    serving, argv = _split_serve_command(None)
    assert serving is True
    assert argv == ["--port", "9000"]


def test_serve_keyword_works_from_any_position(fake_serve):
    """serve 放任意位置都要被识别,OCR 参数不受影响。"""
    for argv in (
        ["serve", "--port", "8010"],
        ["--port", "8010", "serve"],
        ["serve", "--port", "8010", "--host", "127.0.0.1"],
    ):
        fake_serve.clear()
        assert main(argv) == 0
        args = fake_serve[0]
        assert args.port == 8010
        assert args.source is None


def test_serve_sets_env_config(monkeypatch):
    """serve 分支要在启动 uvicorn 前设好 JYKJ_OCR_CONFIG,不能留给 uvicorn 猜。"""
    monkeypatch.delenv("JYKJ_OCR_CONFIG", raising=False)
    monkeypatch.setitem(
        sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **kw: None))
    assert main(["serve", "-c", "config/config.yaml"]) == 0
    assert os.environ["JYKJ_OCR_CONFIG"] == "config/config.yaml"


def test_serve_defaults_config_env_when_absent(monkeypatch):
    """既没传 -c 也没设环境变量时,回退到 config/config.yaml。"""
    monkeypatch.delenv("JYKJ_OCR_CONFIG", raising=False)
    monkeypatch.setitem(
        sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **kw: None))
    assert main(["serve"]) == 0
    assert os.environ["JYKJ_OCR_CONFIG"] == os.path.join("config", "config.yaml")


# --------------------------------------------------- OCR 路径(原 bug 回归点)


def test_ocr_positional_source_does_not_hit_invalid_choice(fake_ocr):
    """原 bug:`image.png` 被当成子命令选择,报 invalid choice 后退出 2。"""
    assert main([IMAGE, "--engine", "rapidocr"]) == 0
    assert fake_ocr[0]["source"] == IMAGE
    assert fake_ocr[0]["engine"] == "rapidocr"


def test_ocr_options_before_source(fake_ocr):
    assert main(["--engine", "rapidocr", IMAGE]) == 0
    assert fake_ocr[0]["source"] == IMAGE


def test_ocr_json_format_written_to_output_file(fake_ocr, tmp_path):
    out = tmp_path / "out.json"
    assert main([IMAGE, "--engine", "rapidocr", "--format", "json", "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["text"] == FAKE_RESULT["text"]


def test_ocr_strategy_name_is_forwarded(fake_ocr):
    assert main([IMAGE, "--strategy-name", "seq-low_conf"]) == 0
    assert fake_ocr[0]["strategy_name"] == "seq-low_conf"


def test_ocr_pdf_dpi_and_max_pages_forwarded(fake_ocr):
    assert main(["doc.pdf", "--dpi", "300", "--max-pages", "2"]) == 0
    assert fake_ocr[0]["dpi"] == 300
    assert fake_ocr[0]["max_pages"] == 2


def test_unknown_strategy_name_is_a_parser_error(fake_ocr, capsys):
    """未知预设必须在 argparse 层报错(退出 2),不能静默走默认策略。"""
    with pytest.raises(SystemExit) as exc:
        main([IMAGE, "--strategy-name", "no-such-preset"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_no_source_prints_help_and_exits_2(fake_ocr, capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out


def test_unknown_option_is_a_parser_error(fake_ocr, capsys):
    with pytest.raises(SystemExit) as exc:
        main([IMAGE, "--engine-x", "rapidocr"])
    assert exc.value.code == 2
    assert "invalid choice" not in capsys.readouterr().err


def test_list_engines_exits_without_ocr(fake_ocr):
    assert main(["--list-engines"]) == 0
    assert fake_ocr == []


# ---------------------------------------------------------------- 解析器形状


def test_parser_has_no_subcommands():
    """serve 是关键字,不是子命令——解析器里不应出现名为 command 的 subparser。"""
    parser = build_parser()
    subs = [a for a in parser._actions
            if getattr(a, "choices", None) is not None and a.dest == "command"]
    assert subs == []


def test_parser_default_port_comes_from_env(monkeypatch):
    monkeypatch.setenv("JYKJ_OCR_PORT", "9090")
    assert build_parser().parse_args([]).port == 9090
    monkeypatch.delenv("JYKJ_OCR_PORT", raising=False)
    assert build_parser().parse_args([]).port == 8000
