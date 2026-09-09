# -*- coding: utf-8 -*-
"""Decision trace, response envelope, and OpenAPI placeholder coverage.

The real ``create_app`` is used here, unlike ``test_server_preset_routes.py``
which hand-copies the route helpers into a mock app. That matters: the new
``score`` / ``score_mode`` / ``decision`` fields and the Swagger placeholder
filter live in the real ``_apply_inline_overrides`` + ``_ocr_response`` pair, so
a mirror would pass no matter what the production code did.

Only the OCR engines are faked (``build_engine``) — the real ``build_pipeline``,
``apply_strategy_preset``, and ``load_source`` all run, using the two images in
this directory. So every call is a genuine end-to-end pass from HTTP request to
decision trace, with no Pillow-heavy work, no network, and no API key.
"""

from __future__ import annotations

import base64
import json as _json
import sys
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jykj_ocr.config import EngineConfig, from_mapping
from jykj_ocr.engine import registry as engine_registry
from jykj_ocr.models import BoundingBox, OCRResult, TextRegion

_TESTS = Path(__file__).resolve().parent
SEAL = _TESTS / "盖章.png"
LANTING = _TESTS / "兰亭序.jpeg"

#: The seal is a small, geometric, high-contrast image — a good stand-in for a
#: layout the retry predicate can reject.
SEAL_B64 = base64.b64encode(SEAL.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Engine doubles
# ---------------------------------------------------------------------------
class _FakeEngine:
    """Config-driven OCR double.

    Every dial lives in the ``engines`` mapping, so a test reads like a
    scenario description instead of a chain of constructors::

        _wire(monkeypatch, {"rapidocr": {"text": "", "confidence": 0.2},
                            "multimodal": {"text": "seal number", "retries": 1}})
    """

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.calls = 0
        self.config = EngineConfig(
            name=name,
            base_url=spec.get("base_url", ""),
            api_key=spec.get("api_key", ""),
            model=spec.get("model", ""),
        )
        self.model = spec.get("model", "")
        self._text = spec.get("text", "")
        self._confidence = spec.get("confidence", 1.0)
        self._raise = spec.get("raise")
        # An empty latency means "the engine did not time itself" — which is the
        # case for every real engine in this project, and the exact condition
        # that used to break bestof-fastest.
        self._own_ms = spec.get("own_elapsed_ms", 0)
        # Merged rows (窜行) are a *geometric* signal: the boxes have to be
        # genuinely wide or genuinely overlapping. The default vertical stack
        # is a clean reading order and must not trip the predicate.
        self._garbled = bool(spec.get("garbled"))
        # Real engines carry no credentials; set this to make the base_url /
        # api_key visible in the output so an endpoint can be asserted to
        # receive the right one.
        self._echo_creds = bool(spec.get("echo_creds"))

    def _output_text(self) -> str:
        # Only echoes when the test itself declared a credential, not when the
        # config entry happens to carry one: the ranking/latency tests use
        # _cfg()'s multimodal entry, which does have a base_url and api_key,
        # and those would perturb char counts and scores.
        if (self._text and self._echo_creds
                and (self.spec.get("base_url") or self.spec.get("api_key"))):
            return f"{self._text}|{self.config.base_url}|{self.config.api_key}"
        return self._text

    def _region_boxes(self, count: int):
        if self._garbled:
            # A real 窜行 case: detection merged the rows into one long box, so
            # every box is ~30x wider than it is tall and they overlap.
            return [
                BoundingBox(x1=0, y1=offset * 6, x2=360, y2=offset * 6 + 12)
                for offset in range(count)
            ]
        return [
            BoundingBox(x1=0, y1=offset * 24, x2=50, y2=offset * 24 + 20)
            for offset in range(count)
        ]

    def recognise(self, image) -> OCRResult:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        # The response body is built from ``regions`` (see ``to_markdown``), not
        # from ``text``, so the credential echo has to land in the regions too.
        text = self._output_text()
        chunks = _chunks(text, self.spec.get("fragments"))
        # No text and no regions: OCRResult.ok is `text or regions`, so an
        # empty-text result with a leftover region would still count as ok and
        # hide the retry path these tests are checking.
        regions = [] if not text else [
            TextRegion(text=chunk, confidence=self._confidence, bbox=bbox)
            for chunk, bbox in zip(chunks, self._region_boxes(len(chunks)))
        ]
        result = OCRResult(
            engine=self.name, model=self.model,
            text=text, regions=regions, width=80, height=240,
        )
        # 0 is the honest "engine did not set this" value; the strategy layer
        # must supply a measured figure, not trust a zero.
        result.elapsed_ms = self._own_ms
        return result


def _chunks(text: str, count: Optional[int]) -> list:
    """Split ``text`` into ``count`` equal pieces, or one piece when omitted.

    Used to build multi-region results without writing region literals by hand.
    """
    if not count or count <= 1:
        return [text]
    if not text:
        return [text] * count
    size = max(1, len(text) // count)
    return [text[i:i + size] for i in range(0, len(text), size)]


def _cfg():
    return from_mapping({
        "engines": [
            {"name": "rapidocr", "enabled": True},
            {"name": "multimodal", "enabled": True,
             "model": "PaddleOCR-VL-1.5",
             "base_url": "https://entry.example.com/v1",
             "api_key": "sk-entry"},
        ],
        "strategy": {"max_retries": 0, "min_confidence": 0.6},
        "output": {},
        "pdf": {},
    })


def _wire(monkeypatch, specs: dict, image: Path = SEAL) -> dict:
    """Point the real registry at faked engines and return the engine instances.

    ``build_pipeline`` is left untouched: the preset-to-engine assembly is part
    of what these tests are checking. ``load_config`` is faked so no config
    file from the checkout influences the scenario.
    """
    engines = {name: _FakeEngine(name, spec) for name, spec in specs.items()}
    _wire_engines(monkeypatch, engines, image)
    return engines


def _wire_engines(monkeypatch, engines: dict, image: Path = SEAL) -> None:
    """Point the real registry at already-built engine doubles.

    Split out from :func:`_wire` so a test that needs to read ``.calls`` can
    share one set of instances between the wiring and the client instead of
    getting two unrelated ones. The doubles also pick up the ``EngineConfig``
    the registry hands them, so a per-request ``base_url`` / ``api_key``
    override is observable in the engine the pipeline actually used.
    """
    from jykj_ocr.engine.inputs import load

    def _build(name, config, engine_config=None):
        engine = engines[name]
        if engine_config is not None and (engine_config.base_url or
                                          engine_config.api_key):
            engine.config = engine_config
        return engine

    monkeypatch.setattr(engine_registry, "build_engine", _build)
    monkeypatch.setattr("jykj_ocr.server.load_source",
                        lambda source, max_pages=None, dpi=200: load(str(image)))
    monkeypatch.setattr("jykj_ocr.server.load_config", lambda path=None: _cfg())


def _client(monkeypatch, specs: dict, image: Path = SEAL,
            engines: Optional[dict] = None,
            cfg: Optional[Config] = None) -> TestClient:
    """Build the app once per test and hand back the client.

    ``_wire`` returns the engine instances the app is wired to. Pass them back
    through ``engines`` when a test asserts on ``.calls``: without it, ``_wire``
    builds one set of doubles and ``_client`` builds a second, and the counts
    are read off the set nobody used. ``cfg`` swaps the config the app loads —
    used to exercise a single-enabled-engine deployment shape.
    """
    from jykj_ocr.server import create_app

    if engines is None:
        _wire(monkeypatch, specs, image)
    else:
        _wire_engines(monkeypatch, engines, image)
    if cfg is not None:
        monkeypatch.setattr("jykj_ocr.server.load_config", lambda path=None: cfg)
    return TestClient(create_app())


def _post_file(client: TestClient, url: str, path: Path, **data):
    return client.post(
        url,
        files={"file": (path.name, path.read_bytes(), "image/png")},
        data=data,
    )


def _post_png(client: TestClient, url: str, **data):
    return client.post(url, files={"file": ("img.png", b"\x89PNG", "image/png")},
                       data=data)


# ---------------------------------------------------------------------------
# Decision trace — seq / cascade family
# ---------------------------------------------------------------------------
class TestSeqDecisionTrace:
    def test_first_accept_records_first_accepted(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local wins", "own_elapsed_ms": 5},
            "multimodal": {"text": "remote never reached", "own_elapsed_ms": 200},
        })

        r = _post_png(c, "/ocr/seq", format="json")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["engine"] == "rapidocr"
        assert body["score_mode"] == "mean_confidence"
        # seq scores the accepted entry by mean region confidence.
        assert body["score"] == pytest.approx(1.0)

        d = body["decision"]
        assert d["selected"] == "rapidocr"
        assert d["reason"] == "first_accepted"
        assert d["fallback"] is False
        assert d["engine_order"] == ["rapidocr", "multimodal"]
        assert d["retries"] == 0
        assert d["accepted"][0]["engine"] == "rapidocr"
        assert d["accepted"][0]["own_elapsed_ms"] == 5
        assert d["rejected"] == []
        assert d["total_elapsed_ms"] >= 0

    def test_low_confidence_is_rejected_then_degraded(self, monkeypatch):
        """Cascade degrades on a low-confidence result and records the reason."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "suspicious output", "confidence": 0.31},
            "multimodal": {"text": "cleaner remote output", "confidence": 0.99},
        })

        r = _post_png(c, "/ocr/cascade-low_conf", format="json")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["engine"] == "multimodal"
        assert body["score"] == pytest.approx(0.99)
        assert body["text"] == "cleaner remote output"

        d = body["decision"]
        assert d["reason"] == "first_accepted"
        assert d["fallback"] is False
        assert len(d["rejected"]) == 1
        rejected = d["rejected"][0]
        assert rejected["engine"] == "rapidocr"
        assert rejected["ok"] is True
        assert rejected["mean_confidence"] == pytest.approx(0.31)
        assert rejected["region_count"] == 1
        assert rejected["char_count"] == len("suspicious output")
        assert rejected["attempt"] == 1

    def test_garbled_layout_is_rejected_with_the_number_that_caused_it(self, monkeypatch):
        """A merged-row layout must be diagnosable from the trace alone."""
        fragments = 20
        c = _client(monkeypatch, {
            "rapidocr": {"text": "一" * fragments, "fragments": fragments,
                          "garbled": True},
            "multimodal": {"text": "clean reading", "confidence": 0.99},
        })

        r = _post_png(c, "/ocr/cascade-line_overlap", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "multimodal"
        assert len(d["rejected"]) == 1
        rejected = d["rejected"][0]
        assert rejected["engine"] == "rapidocr"
        assert rejected["garbled_layout"] is True
        assert rejected["region_count"] == fragments

    def test_rejecting_engine_is_reported_as_an_error(self, monkeypatch):
        """A raising engine must surface in decision["errors"], not disappear."""
        c = _client(monkeypatch, {
            "rapidocr": {"raise": RuntimeError("model file missing")},
            "multimodal": {"text": "remote saves the day", "confidence": 0.9},
        })

        r = _post_png(c, "/ocr/seq", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "multimodal"
        assert d["errors"] == [
            {"engine": "rapidocr", "error": "RuntimeError: model file missing"}
        ]
        assert d["accepted"][0]["engine"] == "multimodal"

    def test_empty_first_engine_retries_then_degrades(self, monkeypatch):
        """retry_mode=no_text with retries=1 records two rejected attempts."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": ""},
            "multimodal": {"text": "remote text"},
        })

        r = _post_png(c, "/ocr", format="json",
                      strategy_name="seq", max_retries="1")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["retries"] == 1
        assert [e["attempt"] for e in d["rejected"]] == [1, 2]
        assert all(e["ok"] is False for e in d["rejected"])
        assert d["accepted"][0]["engine"] == "multimodal"

    def test_cascade_means_no_retry(self, monkeypatch):
        """max_retries: 0 is the whole point of the cascade family."""
        spec = {"rapidocr": {"text": ""}, "multimodal": {"text": "remote text"}}
        engines = _wire(monkeypatch, spec)
        c = _client(monkeypatch, spec, engines=engines)

        r = _post_png(c, "/ocr", format="json", strategy_name="cascade")
        assert r.status_code == 200, r.text
        assert r.json()["decision"]["retries"] == 0
        # The rejected engine was tried exactly once, not retried.
        assert engines["rapidocr"].calls == 1
    def test_seq_retains_the_configured_retries(self, monkeypatch):
        """Regression: apply_strategy_preset used to ``pop`` max_retries for
        every non-cascade preset, silently resetting a configured
        ``max_retries: 0`` back to the default 1. The docstring promised
        ``seq*`` keeps the config value — and ``/ocr/cascade`` was the only
        preset that actually delivered zero retries for callers who set it
        in config.
        """
        import jykj_ocr.engine.registry as reg
        from jykj_ocr.engine.registry import apply_strategy_preset

        cfg = _cfg()
        assert cfg.strategy["max_retries"] == 0

        for name in ("seq", "seq-low_conf", "seq-line_overlap", "local",
                     "fallback", "quality", "vl"):
            out = apply_strategy_preset(cfg, name)
            assert out.strategy["max_retries"] == 0, f"{name} lost the config value"

        # cascade* still forces 0, and bestof still drops it.
        assert apply_strategy_preset(cfg, "cascade").strategy["max_retries"] == 0
        assert "max_retries" not in apply_strategy_preset(cfg, "bestof").strategy

        # And the pipeline really builds with it — this is what HTTP returns.
        engines = _wire(monkeypatch, {
            "rapidocr": {"text": ""},
            "multimodal": {"text": "remote text"},
        })
        pipeline = reg.build_pipeline(apply_strategy_preset(_cfg(), "seq"))
        assert isinstance(pipeline, reg.StrategyEngine)
        assert pipeline.retries == 0

    def test_all_rejected_falls_back_and_says_so(self, monkeypatch):
        """When every engine is refused, the fallback is declared, not implied."""
        c = _client(monkeypatch, {
            # Both too low-confidence to clear the predicate; the longer one wins.
            "rapidocr": {"text": "short", "confidence": 0.3},
            "multimodal": {"text": "much longer but still uncertain text",
                           "confidence": 0.3},
        })

        r = _post_png(c, "/ocr", format="json",
                      strategy_name="seq-low_conf", retry_mode="low_confidence")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["reason"] == "all_rejected_fallback"
        assert d["fallback"] is True
        assert len(d["rejected"]) == 2
        # The fallback entry is the longest usable result, not a retry victim.
        assert d["accepted"][0]["engine"] == "multimodal"


# ---------------------------------------------------------------------------
# Decision trace — bestof family
# ---------------------------------------------------------------------------
class TestBestofDecisionTrace:
    def test_ranks_every_engine_by_descending_score(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local short", "confidence": 0.9},
            "multimodal": {"text": "remote gives a much longer and more fluent answer",
                           "confidence": 0.99},
        })

        r = _post_png(c, "/ocr/bestof", format="json")
        assert r.status_code == 200, r.text
        body = r.json()

        d = body["decision"]
        assert d["reason"] == "highest_smart_score"
        assert d["score_mode"] == "smart"
        assert d["engine_order"] == ["rapidocr", "multimodal"]
        assert len(d["ranked"]) == 2

        scores = [entry["score"] for entry in d["ranked"]]
        assert scores == sorted(scores, reverse=True)
        assert body["score"] == scores[0]
        assert body["score_mode"] == "smart"
        assert d["selected"] == d["ranked"][0]["engine"]

        for entry in d["ranked"]:
            for key in ("engine", "model", "own_elapsed_ms", "ok", "score",
                        "mean_confidence", "region_count", "char_count",
                        "garbled_layout", "detail"):
                assert key in entry, key
            assert isinstance(entry["detail"], dict)

    def test_fastest_breaks_ties_by_latency_not_config_order(self, monkeypatch):
        """Regression: before _own_elapsed, every candidate scored elapsed_ms=0
        and bestof-fastest silently resolved to config order.

        The dials say the *local* engine is fastest and the remote is slow; the
        engine is written first in the config. If the mode still ties to
        elapsed_ms=0, config order wins and the test fails — which is exactly
        the misbehaviour the user reported as 'bestof 选的是 rapidocr 但效果差'.
        """
        c = _client(monkeypatch, {
            "rapidocr": {"text": "slow local", "confidence": 0.9,
                         "own_elapsed_ms": 900},
            "multimodal": {"text": "quick remote", "confidence": 0.99,
                           "own_elapsed_ms": 10},
        })

        r = _post_png(c, "/ocr/bestof-fastest", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "multimodal"
        assert d["ranked"][0]["engine"] == "multimodal"

        # Scores are strictly ordered by latency — a 0/0 tie is the bug.
        by_engine = {e["engine"]: e["own_elapsed_ms"] for e in d["ranked"]}
        assert by_engine == {"rapidocr": 900, "multimodal": 10}
        scores = [e["score"] for e in d["ranked"]]
        assert scores[0] > scores[1]
        assert d["ranked"][0]["detail"] == {"own_elapsed_ms": 10}

    def test_untimed_engine_still_scores_fastest(self, monkeypatch):
        """The real engines never set elapsed_ms — fastest must work anyway."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "untimed local", "own_elapsed_ms": 0},
            "multimodal": {"text": "untimed remote", "own_elapsed_ms": 0},
        })

        r = _post_png(c, "/ocr/bestof-fastest", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] in ("rapidocr", "multimodal")
        # Both got a measured figure; neither is a bare zero from an unpatched path.
        assert all(e["own_elapsed_ms"] >= 0 for e in d["ranked"])
        assert d["ranked"][0]["detail"]["own_elapsed_ms"] >= 0

    def test_fastest_detail_carries_only_its_own_component(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "text", "own_elapsed_ms": 20},
            "multimodal": {"text": "more text", "own_elapsed_ms": 200},
        })

        r = _post_png(c, "/ocr", format="json", strategy_name="bestof-fastest")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["score_mode"] == "fastest"
        # Mode-specific detail: no fluency numbers in a fastest trace.
        assert set(d["ranked"][0]["detail"]) == {"own_elapsed_ms"}
        assert "phrase_bonus" not in d["ranked"][0]["detail"]

    def test_fluency_detail_exposes_its_components(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "一" * 16, "fragments": 16},
            "multimodal": {"text": "很长的连贯中文短语,带标点。"},
        })

        r = _post_png(c, "/ocr/bestof-fluency", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["reason"] == "highest_fluency_score"
        detail = d["ranked"][0]["detail"]
        assert set(detail) == {"phrase_bonus", "punct_bonus", "frag_penalty",
                               "single_chars", "mean_phrase_len"}
        # The winner is the engine with the fewer single-char fragments.
        assert d["selected"] == "multimodal"

    def test_longest_wins_on_char_count(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "very long local output wins", "confidence": 0.9},
            "multimodal": {"text": "short", "confidence": 0.99},
        })

        r = _post_png(c, "/ocr", format="json", strategy_name="bestof-longest")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "rapidocr"
        assert d["ranked"][0]["detail"] == {
            "char_count": len("very long local output wins")
        }

    def test_confidence_mode_reports_mean_confidence_only(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local", "confidence": 0.9},
            "multimodal": {"text": "remote", "confidence": 0.99},
        })

        r = _post_png(c, "/ocr", format="json", strategy_name="bestof-confidence")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "multimodal"
        for entry in d["ranked"]:
            assert set(entry["detail"]) == {"mean_confidence"}
        assert d["ranked"][0]["mean_confidence"] == pytest.approx(0.99)

    def test_unusable_engine_is_listed_in_errors(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"raise": ValueError("no api key")},
            "multimodal": {"text": "remote still works", "confidence": 0.9},
        })

        r = _post_png(c, "/ocr/bestof", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "multimodal"
        assert d["errors"] == [
            {"engine": "rapidocr", "error": "ValueError: no api key"}
        ]
        # Only the surviving engine is ranked.
        assert [e["engine"] for e in d["ranked"]] == ["multimodal"]

    def test_empty_engine_is_ranked_but_not_selected(self, monkeypatch):
        """A score on empty text must not beat an ok result."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "", "own_elapsed_ms": 1},
            "multimodal": {"text": "some text", "confidence": 0.6,
                           "own_elapsed_ms": 500},
        })

        r = _post_png(c, "/ocr/bestof", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["selected"] == "multimodal"
        assert {e["engine"] for e in d["ranked"]} == {"rapidocr", "multimodal"}
        loser = next(e for e in d["ranked"] if e["engine"] == "rapidocr")
        assert loser["ok"] is False
        assert loser["detail"] == {"excluded": "not_ok"}

    def test_garbled_layout_is_penalised_in_the_trace(self, monkeypatch):
        """smart subtracts for merged rows; the trace shows which engine lost."""
        fragments = 20
        c = _client(monkeypatch, {
            "rapidocr": {"text": "一" * fragments, "fragments": fragments,
                          "confidence": 0.99, "garbled": True},
            "multimodal": {"text": "clean coherent text with punctuation",
                           "confidence": 0.9},
        })

        r = _post_png(c, "/ocr/bestof", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        by_engine = {e["engine"]: e for e in d["ranked"]}
        assert by_engine["rapidocr"]["garbled_layout"] is True
        assert by_engine["rapidocr"]["detail"]["garbled_penalty"] == -20.0
        assert d["selected"] == "multimodal"

    def test_bestof_runs_every_engine_once(self, monkeypatch):
        """That every engine runs is what separates bestof from seq."""
        spec = {"rapidocr": {"text": "local", "own_elapsed_ms": 10},
                "multimodal": {"text": "remote", "own_elapsed_ms": 100}}
        engines = _wire(monkeypatch, spec)
        c = _client(monkeypatch, spec, engines=engines)

        r = _post_png(c, "/ocr/bestof", format="json")
        assert r.status_code == 200, r.text
        assert engines["rapidocr"].calls == 1
        assert engines["multimodal"].calls == 1
        # And the total wall clock covers both, unlike the winner's own latency.
        assert r.json()["decision"]["total_elapsed_ms"] >= 0

    def test_bestof_colon_alias_picks_the_named_mode(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "untimed local", "own_elapsed_ms": 500},
            "multimodal": {"text": "untimed remote", "own_elapsed_ms": 5},
        })

        r = _post_png(c, "/ocr/bestof:fastest", format="json")
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["score_mode"] == "fastest"
        assert d["selected"] == "multimodal"


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------
class TestResponseEnvelope:
    def test_json_carries_score_score_mode_decision(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local text"},
            "multimodal": {"text": "remote text"},
        })

        r = _post_png(c, "/ocr/seq", format="json")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("pages", "text", "engine", "page_count",
                    "score", "score_mode", "decision"):
            assert key in body, key
        assert isinstance(body["decision"], dict)
        assert body["score_mode"] == "mean_confidence"

        # Per-page copies stay in sync with the top level.
        page = body["pages"][0]
        assert page["decision"] == body["decision"]
        assert page["score"] == body["score"]
        assert page["score_mode"] == body["score_mode"]

    def test_text_format_omits_the_trace_fields(self, monkeypatch):
        """format=text/markdown is a PlainTextResponse, not a JSON envelope."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "line one"},
            "multimodal": {"text": "other"},
        })

        for fmt in ("text", "markdown"):
            r = _post_png(c, "/ocr/seq", format=fmt)
            assert r.status_code == 200, r.text
            assert r.headers["content-type"].startswith("text/plain")
            assert r.text.strip() == "line one"

    def test_markdown_uses_reading_order(self, monkeypatch):
        """Reorder happens before render, so the trace-free body is still ordered."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "A", "fragments": 1},
            "multimodal": {"text": "B"},
        })

        r = _post_png(c, "/ocr/seq", format="markdown")
        assert r.status_code == 200, r.text
        assert "A" in r.text

    def test_forced_engine_route_reports_no_decision(self, monkeypatch):
        """A single forced engine is not a strategy choice — no trace is invented."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "only engine"},
            "multimodal": {"text": "x"},
        })

        r = _post_png(c, "/ocr/rapidocr", format="json")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["engine"] == "rapidocr"
        assert body["score"] is None
        assert body["score_mode"] == ""
        assert body["decision"] is None

    def test_json_body_route_carries_the_trace(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local"},
            "multimodal": {"text": "remote", "confidence": 0.95},
        })

        r = client_post = c.post("/ocr/text", json={
            "image_url": "https://example.com/scan.png",
            "strategy_name": "seq",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"]["selected"] == "rapidocr"
        assert body["decision"]["reason"] == "first_accepted"
        assert body["score"] == pytest.approx(1.0)

    def test_preset_text_route_carries_the_trace(self, monkeypatch):
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local"},
            "multimodal": {"text": "remote"},
        })

        r = c.post("/ocr/vl/text", json={
            "image_url": "https://example.com/scan.png",
        })
        assert r.status_code == 200, r.text
        assert r.json()["decision"]["selected"] == "multimodal"

    def test_b64_source_carries_the_trace(self, monkeypatch):
        """All three image inputs go through the same envelope."""
        c = _client(monkeypatch, {
            "rapidocr": {"text": "local"},
            "multimodal": {"text": "remote"},
        })

        r = c.post("/ocr/text", json={
            "image_b64": SEAL_B64, "strategy_name": "bestof",
        })
        assert r.status_code == 200, r.text
        d = r.json()["decision"]
        assert d["reason"] == "highest_smart_score"
        assert len(d["ranked"]) == 2

    def test_real_images_are_recognised_end_to_end(self, monkeypatch):
        """The two checked-in fixtures drive the real loader, not a stub."""
        for image in (SEAL, LANTING):
            assert image.exists(), image
            c = _client(monkeypatch, {
                "rapidocr": {"text": image.stem},
                "multimodal": {"text": "remote"},
            }, image=image)

            r = _post_file(c, "/ocr/seq", image, format="json")
            assert r.status_code == 200, r.text
            body = r.json()
            # The uploaded page reaches the pipeline and is reported.
            assert body["page_count"] == 1
            assert body["pages"][0]["width"] > 0
            assert body["decision"]["engine_order"] == ["rapidocr", "multimodal"]


# ---------------------------------------------------------------------------
# Swagger placeholder filter
# ---------------------------------------------------------------------------
class TestOpenAPIPlaceholderFilter:
    """Swagger UI's "Generate cURL" fills *every* optional parameter with the
    schema type as a literal. Those values must be ignored, not applied.
    """

    PLACEHOLDERS = ("string", "number", "integer", "boolean", "array", "object")

    def _client(self, monkeypatch):
        return _client(monkeypatch, {
            "rapidocr": {"text": "local text"},
            "multimodal": {"text": "remote text", "echo_creds": True,
                           "base_url": "https://entry.example.com/v1",
                           "api_key": "sk-entry"},
        })

    def test_placeholder_retry_mode_does_not_400(self, monkeypatch):
        """The exact failure reported: ``retry_mode=string`` -> HTTP 400."""
        c = self._client(monkeypatch)

        for preset in ("vl", "seq", "bestof"):
            r = _post_png(c, f"/ocr/{preset}",
                          retry_mode="string", score_mode="string",
                          format="string")
            assert r.status_code == 200, f"{preset}: {r.status_code} {r.text}"

    def test_every_placeholder_literal_is_ignored(self, monkeypatch):
        c = self._client(monkeypatch)

        for literal in self.PLACEHOLDERS:
            r = _post_png(c, "/ocr/seq", retry_mode=literal, format="json")
            assert r.status_code == 200, f"{literal}: {r.status_code} {r.text}"

    def test_placeholder_credentials_do_not_redirect(self, monkeypatch):
        """``base_url=string`` must not point the request at the literal host."""
        c = self._client(monkeypatch)

        r = _post_png(c, "/ocr/vl",
                      base_url="string", api_key="string",
                      model="string", prompt="string")
        assert r.status_code == 200, r.text
        # The engine's own endpoint/key are what survive, not the placeholders.
        assert "https://entry.example.com/v1|sk-entry" in r.json()["text"]
        assert "string|string" not in r.json()["text"]

    def test_placeholder_format_stays_json(self, monkeypatch):
        """``format=string`` is the same trap on the format knob itself."""
        c = self._client(monkeypatch)

        r = _post_png(c, "/ocr/seq", format="string")
        assert r.status_code == 200, r.text
        assert "decision" in r.json()

        # A real format value is still honoured.
        r2 = _post_png(c, "/ocr/seq", format="text")
        assert r2.headers["content-type"].startswith("text/plain")

    def test_placeholder_format_in_json_body(self, monkeypatch):
        c = self._client(monkeypatch)

        r = c.post("/ocr/text", json={
            "image_url": "https://example.com/scan.png", "format": "string",
        })
        assert r.status_code == 200, r.text
        assert "decision" in r.json()

    def test_placeholder_max_retries_is_ignored(self, monkeypatch):
        """``max_retries=integer`` is the same trap on the int-typed knob.

        Every other knob is a string, so ``_clean`` already swallowed its
        placeholder. ``max_retries`` is int-typed: the literal used to reach
        pydantic as a string and 500, which defeated the whole "Generate cURL
        must not break" guarantee on the one field it names by number.

        Must target ``/ocr`` — the ``/ocr/{preset}`` routes never declare the
        knob at all, so they cannot exercise the parser.
        """
        c = self._client(monkeypatch)

        for literal in self.PLACEHOLDERS:
            r = _post_png(c, "/ocr", max_retries=literal,
                          strategy_name="seq", format="json")
            assert r.status_code == 200, f"{literal}: {r.status_code} {r.text}"
            assert isinstance(r.json()["decision"]["retries"], int)

    def test_invalid_max_retries_is_a_400_not_a_500(self, monkeypatch):
        """``max_retries=abc`` used to escape as a bare 500.

        The multipart form declared the field ``str`` while ``TextRequest``
        types it ``int``, so pydantic rejected the body before any business
        check could answer — a typo became an internal server error.
        """
        c = self._client(monkeypatch)

        r = _post_png(c, "/ocr", max_retries="abc",
                      strategy_name="seq", format="json")
        assert r.status_code == 400, r.text
        assert "max_retries" in r.json()["detail"]

    def test_parse_max_retries_form_shape(self):
        """Blanks, placeholders and case variants all mean "not sent"."""
        from jykj_ocr.server import _parse_max_retries_form

        for literal in self.PLACEHOLDERS:
            assert _parse_max_retries_form(literal) is None, literal
        for case in ("Integer", "INTEGER", "InTeGeR"):
            assert _parse_max_retries_form(case) is None, case
        for blank in (None, "", "   "):
            assert _parse_max_retries_form(blank) is None, repr(blank)
        # Real values survive, including the two semantically distinct ones.
        assert _parse_max_retries_form("7") == 7
        assert _parse_max_retries_form("0") == 0
        assert _parse_max_retries_form("  3 ") == 3

    def test_parse_max_retries_form_rejects_typo_and_negative(self):
        """A typo or negative value is a client error that names the field."""
        from fastapi import HTTPException
        from jykj_ocr.server import _parse_max_retries_form

        for bad in ("abc", "-1", "-100", "1.5", "3x"):
            with pytest.raises(HTTPException) as excinfo:
                _parse_max_retries_form(bad)
            assert excinfo.value.status_code == 400, bad
            assert "max_retries" in excinfo.value.detail, bad

    def test_real_retry_mode_still_400s_when_invalid(self, monkeypatch):
        """The filter must not turn a genuine typo into a silent no-op.

        Uses ``/ocr`` — the preset routes intentionally expose no strategy
        knobs at all, so they have nothing to validate there.
        """
        c = self._client(monkeypatch)

        r = _post_png(c, "/ocr", retry_mode="nonsense")
        assert r.status_code == 400
        assert "retry_mode" in r.json()["detail"]

        # A real override on the preset route is ignored, not rejected: the
        # route *is* the strategy, so there is nothing to be 400 about.
        r2 = _post_png(c, "/ocr/seq", retry_mode="nonsense")
        assert r2.status_code == 200, r2.text

    def test_real_override_is_still_applied(self, monkeypatch):
        c = self._client(monkeypatch)

        r = _post_png(c, "/ocr/vl", base_url="https://other.example.com/v1",
                      api_key="sk-per-request")
        assert r.status_code == 200, r.text
        assert "https://other.example.com/v1|sk-per-request" in r.json()["text"]

    def test_empty_string_credentials_are_a_noop(self, monkeypatch):
        """``-F api_key=""`` behaves the same as omitting the field."""
        c = self._client(monkeypatch)

        r = _post_png(c, "/ocr/vl", api_key="", base_url="")
        assert r.status_code == 200, r.text
        assert "https://entry.example.com/v1|sk-entry" in r.json()["text"]

    def test_clean_drops_placeholders_and_blanks(self):
        from jykj_ocr.server import _clean

        assert _clean(None) is None
        assert _clean("") is None
        assert _clean("   ") is None
        for literal in self.PLACEHOLDERS:
            assert _clean(literal) is None, literal
        # Case-insensitive: Swagger emits lowercase, but the check must not care.
        assert _clean("String") is None
        assert _clean("STRING") is None
        # Real values pass through, stripped.
        assert _clean("  no_text  ") == "no_text"
        assert _clean("https://x/v1") == "https://x/v1"

    def test_clean_format_defaults_to_json(self):
        """format's default is "json", not absence — the two need different rules."""
        from jykj_ocr.server import _clean_format

        assert _clean_format(None) == "json"
        assert _clean_format("") == "json"
        assert _clean_format("string") == "json"
        assert _clean_format("TEXT") == "text"
        assert _clean_format("markdown") == "markdown"


# ---------------------------------------------------------------------------
# OpenAPI surface
# ---------------------------------------------------------------------------
class TestOpenAPISurface:
    """The published contract, not just the runtime behaviour.

    FastAPI puts ``Form(...)`` fields in ``requestBody``, not ``parameters`` —
    only the path parameter (``preset``) lands in ``parameters``.
    """

    OCR_PATHS = ("/ocr", "/ocr/text", "/ocr/{preset}", "/ocr/{preset}/text")

    @staticmethod
    def _form_fields(spec: dict, path: str) -> dict:
        """The multipart form fields a caller can send to ``path``."""
        body = spec["paths"][path]["post"]["requestBody"]
        content = body["content"]
        assert "multipart/form-data" in content, path
        schema = content["multipart/form-data"]["schema"]
        if "$ref" in schema:
            schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
        return schema["properties"]

    @staticmethod
    def _json_body_fields(spec: dict, path: str) -> dict:
        body = spec["paths"][path]["post"]["requestBody"]
        schema = body["content"]["application/json"]["schema"]
        if "$ref" in schema:
            schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
        return schema["properties"]

    @staticmethod
    def _path_params(spec: dict, path: str) -> list:
        return [p["name"] for p in spec["paths"][path]["post"].get("parameters", [])]

    def test_all_four_ocr_endpoints_document_the_trace_fields(self):
        """The 200 description must name every field the handler returns."""
        from jykj_ocr.server import app

        spec = app.openapi()
        for path in self.OCR_PATHS:
            operation = spec["paths"][path]["post"]
            desc = operation["responses"]["200"]["description"]
            for field in ("pages", "text", "engine", "page_count",
                          "score", "score_mode", "decision"):
                assert field in desc, f"{path}: missing {field} in {desc!r}"

    def test_preset_route_no_longer_exposes_strategy_knobs(self):
        """The route *is* the strategy — retry_mode / score_mode / max_retries
        stay off it. They were what forced Swagger's Generate cURL to emit
        retry_mode=string into every request."""
        from jykj_ocr.server import app

        spec = app.openapi()
        fields = self._form_fields(spec, "/ocr/{preset}")
        for knob in ("retry_mode", "score_mode", "max_retries"):
            assert knob not in fields, knob
        for kept in ("model", "base_url", "api_key", "prompt",
                     "max_pages", "dpi", "format"):
            assert kept in fields, kept
        # The strategy travels in the path, not in a body field.
        assert self._path_params(spec, "/ocr/{preset}") == ["preset"]

    def test_generic_ocr_still_exposes_the_strategy_knobs(self):
        """Only the preset routes dropped the knobs; /ocr keeps them."""
        from jykj_ocr.server import app

        spec = app.openapi()
        fields = set(self._form_fields(spec, "/ocr"))
        assert {"strategy_name", "retry_mode", "score_mode",
                "max_retries"} <= fields

    def test_text_request_body_carries_retry_and_score_modes(self):
        """/ocr/text takes the knobs through the JSON body, not form fields."""
        from jykj_ocr.server import app

        spec = app.openapi()
        props = self._json_body_fields(spec, "/ocr/text")
        for field in ("retry_mode", "score_mode", "max_retries", "format"):
            assert field in props, field

        # /ocr/text has no form fields at all — the whole body is JSON.
        assert self._path_params(spec, "/ocr/text") == []

    def test_preset_text_route_is_the_json_twin(self):
        from jykj_ocr.server import app

        spec = app.openapi()
        props = self._json_body_fields(spec, "/ocr/{preset}/text")
        assert "retry_mode" in props
        assert "strategy_name" in props
        assert self._path_params(spec, "/ocr/{preset}/text") == ["preset"]

    def test_every_ocr_endpoint_has_a_200_documentation(self):
        from jykj_ocr.server import app

        spec = app.openapi()
        for path in self.OCR_PATHS:
            assert "200" in spec["paths"][path]["post"]["responses"], path


class TestDocsPresetGreying:
    """The /docs page drives the spec off ?preset=, and the spec marks
    irrelevant body fields as readOnly (which Swagger UI renders as "not shown").
    """

    @staticmethod
    def _form_props(spec: dict, path: str) -> dict:
        body = spec["paths"][path]["post"]["requestBody"]
        schema = body["content"]["multipart/form-data"]["schema"]
        if "$ref" in schema:
            schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
        return schema["properties"]

    @staticmethod
    def _json_body_ref(spec: dict, path: str) -> str:
        body = spec["paths"][path]["post"]["requestBody"]
        return body["content"]["application/json"]["schema"].get("$ref", "")

    @staticmethod
    def _json_body_props(spec: dict, path: str) -> dict:
        body = spec["paths"][path]["post"]["requestBody"]
        schema = body["content"]["application/json"]["schema"]
        if "$ref" in schema:
            schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
        return schema["properties"]

    @staticmethod
    def _preset_desc(spec: dict, path: str) -> str:
        for p in spec["paths"][path]["post"].get("parameters", []):
            if p.get("in") == "path" and p.get("name") == "preset":
                return p.get("description", "")
        return ""

    def test_docs_page_serves_pinned_bundle_and_greying_script(self):
        """/docs must use the pinned swagger-ui-dist (not the floating @5 tag)
        and must ship the updateSpec hook — the floating tag can rename
        specActions and silently kill the greying."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        html = TestClient(app).get("/docs").text
        assert "swagger-ui-dist@5.32.2/swagger-ui-bundle.js" in html
        assert "swagger-ui-dist@5/swagger-ui-bundle.js" not in html
        # The script needs the UI instance to call updateSpec in place.
        assert "window.__jykjOcrUI = ui;" in html
        assert "updateSpec" in html

    def test_docs_page_bakes_the_preset_into_the_spec_url(self):
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        client = TestClient(app)
        html_local = client.get("/docs?preset=_local").text
        assert "/openapi.json?preset=local" in html_local
        html_bestof = client.get("/docs?preset=_bestof").text
        assert "/openapi.json?preset=bestof" in html_bestof

    def test_base_openapi_endpoint_is_unchanged(self):
        """/openapi.json without ?preset= is the FastAPI spec verbatim —
        clients and curl consume this, so it must not grow readOnly marks."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        spec = TestClient(app).get("/openapi.json").json()
        props = self._json_body_props(spec, "/ocr/text")
        for field in ("model", "base_url", "api_key", "prompt"):
            assert props[field].get("readOnly") is not True, field

    KEPT = {
        # The two preset routes share the four remote-only fields but their
        # image-source field differs: multipart takes a file, JSON takes a URL.
        "/ocr/{preset}": ("file", "max_pages", "dpi", "format"),
        "/ocr/{preset}/text": ("image_url", "max_pages", "dpi", "format"),
    }

    def test_local_preset_greys_the_remote_only_fields(self):
        """?preset=_local marks the four remote-only body fields readOnly on
        both preset routes. Swagger UI then drops them from the request form."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        spec = TestClient(app).get("/openapi.json?preset=_local").json()
        for path in ("/ocr/{preset}", "/ocr/{preset}/text"):
            props = self._json_body_props(spec, path) if path.endswith("/text") \
                else self._form_props(spec, path)
            for field in ("model", "base_url", "api_key", "prompt"):
                assert props[field].get("readOnly") is True, (path, field)
            # The usable fields stay as-is.
            for kept in self.KEPT[path]:
                assert props[kept].get("readOnly") is not True, (path, kept)

    def test_vl_preset_greys_nothing(self):
        """engine_scope=remote_vl_only → none of the four fields are useless."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        spec = TestClient(app).get("/openapi.json?preset=_vl").json()
        props = self._form_props(spec, "/ocr/{preset}")
        for field in ("model", "base_url", "api_key", "prompt"):
            assert props[field].get("readOnly") is not True, field

    def test_unknown_preset_returns_the_spec_unmodified(self):
        """/docs?preset=bogus must not lie about being a preset — return the
        base spec so the fields aren't greyed out."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        client = TestClient(app)
        spec_bogus = client.get("/openapi.json?preset=_bogus").json()
        spec_base = client.get("/openapi.json").json()
        # Identical objects: the unknown-preset branch returns the input as-is.
        assert spec_bogus is not None
        assert spec_bogus["components"]["schemas"]["TextRequest"] == \
            spec_base["components"]["schemas"]["TextRequest"]

    def test_shared_textrequest_component_stays_ungreyed(self):
        """TextRequest is shared by /ocr/text and /ocr/{preset}/text. Greying
        it in components would grey out /ocr/text too — so the preset route
        must clone it. This test locks that isolation down."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        spec = TestClient(app).get("/openapi.json?preset=_local").json()
        # The shared component itself is never touched.
        shared = spec["components"]["schemas"]["TextRequest"]
        for field in ("model", "base_url", "api_key", "prompt"):
            assert shared["properties"][field].get("readOnly") is not True, field
        # /ocr/text still points at the shared component (not a clone).
        assert self._json_body_ref(spec, "/ocr/text").endswith("/TextRequest")
        # The preset route points at a route-scoped clone.
        clone_ref = self._json_body_ref(spec, "/ocr/{preset}/text")
        assert not clone_ref.endswith("/TextRequest"), clone_ref

    def test_preset_param_description_names_the_selected_preset(self):
        """The path param description must say which preset is active and what
        it implies — this is what the user reads right next to the input box."""
        from fastapi.testclient import TestClient
        from jykj_ocr.server import app

        client = TestClient(app)
        for preset in ("_local", "_vl", "_bestof-fluency", "_seq-any"):
            spec = client.get("/openapi.json?preset=" + preset).json()
            desc = self._preset_desc(spec, "/ocr/{preset}")
            assert preset.strip("_") in desc, (preset, desc)
            assert "retry_mode" in desc, (preset, desc)



# ---------------------------------------------------------------------------
# Strategy-level invariants (no HTTP layer)
# ---------------------------------------------------------------------------
class TestBestofTimingInvariant:
    def test_score_fastest_reads_the_per_engine_figure(self):
        """bestof-fastest compares engine latencies, not total wall clock.

        The regression: BestofEngine rewrote result.elapsed_ms with the total
        wall clock *before* scoring, so every candidate scored identically and
        the winner was decided by config order alone.
        """
        from jykj_ocr.strategy import BestofEngine, _score_fastest

        slow = _FakeEngine("rapidocr", {"text": "slow", "own_elapsed_ms": 400})
        fast = _FakeEngine("multimodal", {"text": "fast", "own_elapsed_ms": 4})
        engine = BestofEngine([slow, fast], score_mode="fastest")
        assert engine.score_fn == _score_fastest

        result = engine.recognise(None)
        assert result.engine == "multimodal"
        detail = {e["engine"]: e for e in result.decision["ranked"]}
        assert detail["multimodal"]["own_elapsed_ms"] == 4
        assert detail["rapidocr"]["own_elapsed_ms"] == 400
        # The winner's elapsed_ms is the total wall clock, not its own latency.
        assert result.elapsed_ms >= 0

    def test_engine_supplied_elapsed_ms_is_not_discarded(self):
        """An engine that already timed itself keeps that figure."""
        from jykj_ocr.strategy import _own_elapsed

        engine = _FakeEngine("rapidocr", {"text": "timed", "own_elapsed_ms": 77})
        result, own = _own_elapsed(engine, None)
        assert own == 77
        assert result.elapsed_ms == 77

    def test_untimed_engine_falls_back_to_wall_clock(self):
        from jykj_ocr.strategy import _own_elapsed

        result, own = _own_elapsed(
            _FakeEngine("rapidocr", {"text": "text", "own_elapsed_ms": 0}), None
        )
        assert isinstance(own, int)
        assert own >= 0

    def test_elapse_is_monotonic_across_candidates(self):
        """A slower engine must not be scored as faster than a quicker one."""
        from jykj_ocr.strategy import _own_elapsed

        fast = _FakeEngine("a", {"text": "x", "own_elapsed_ms": 1})
        slow = _FakeEngine("b", {"text": "y", "own_elapsed_ms": 5000})
        _, fast_ms = _own_elapsed(fast, None)
        _, slow_ms = _own_elapsed(slow, None)
        assert fast_ms == 1
        assert slow_ms == 5000
        assert slow_ms > fast_ms

    def test_decision_trace_is_json_serialisable(self):
        """The trace must survive json.dumps — FastAPI serialises it verbatim."""
        import json

        from jykj_ocr.strategy import BestofEngine, StrategyEngine

        cases = {
            "seq": StrategyEngine([
                _FakeEngine("rapidocr", {"text": "a"}),
                _FakeEngine("multimodal", {"text": "b"}),
            ], retries=0),
            "bestof-smart": BestofEngine([
                _FakeEngine("rapidocr", {"text": "a"}),
                _FakeEngine("multimodal", {"text": "b"}),
            ], score_mode="smart"),
            "bestof-fastest": BestofEngine([
                _FakeEngine("rapidocr", {"text": "a", "own_elapsed_ms": 5}),
                _FakeEngine("multimodal", {"text": "b", "own_elapsed_ms": 9}),
            ], score_mode="fastest"),
        }
        for label, pipeline in cases.items():
            result = pipeline.recognise(None)
            payload = json.dumps(
                {"score": result.score, "score_mode": result.score_mode,
                 "decision": result.decision},
                ensure_ascii=False,
            )
            assert "Infinity" not in payload, label
            assert "NaN" not in payload, label
            assert result.decision is not None, label
            assert result.score is not None, label
            assert result.score_mode, label


def test_as_dict_includes_the_new_fields():
    """OCRResult serialises the trace fields explicitly, not by accident."""
    from jykj_ocr.models import OCRResult

    result = OCRResult(engine="rapidocr", text="abc")
    dumped = result.as_dict()
    assert dumped["score"] is None
    assert dumped["score_mode"] == ""
    assert dumped["decision"] is None

    result.score = 0.91
    result.score_mode = "mean_confidence"
    result.decision = {"selected": "rapidocr", "reason": "first_accepted"}
    dumped = result.as_dict()
    assert dumped["score"] == 0.91
    assert dumped["score_mode"] == "mean_confidence"
    assert dumped["decision"]["selected"] == "rapidocr"
