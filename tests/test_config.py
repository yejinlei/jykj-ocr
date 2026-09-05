# -*- coding: utf-8 -*-
"""Tests for configuration loading, aliases, and env-var resolution."""

from __future__ import annotations

import json

import pytest

from jykj_ocr.config import (
    EngineConfig,
    SILICONFLOW_BASE_URL,
    from_mapping,
    load_config,
    load_prompt,
    normalise_engine,
)
from jykj_ocr.engines.multimodal_engine import MultimodalEngine

_KEY_ENVS = (
    "OPENAI_API_KEY",
    "SILICONFLOW_API_KEY",
    "JYKJ_OCR_SILICONFLOW_API_KEY",
    "MULTIMODAL_API_KEY",
    "JYKJ_OCR_MULTIMODAL_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_keys(monkeypatch):
    """Make API-key tests deterministic regardless of the host environment."""
    for var in _KEY_ENVS:
        monkeypatch.delenv(var, raising=False)


class TestNormaliseEngine:
    def test_rapid_aliases(self):
        for alias in ("rapid", "rapid-ocr", "rapidocr-onnx"):
            assert normalise_engine(alias) == "rapidocr"

    def test_siliconflow_aliases(self):
        for alias in ("sf", "silicon-flow", "silicon_flow", "SILICONFLOW"):
            assert normalise_engine(alias) == "siliconflow"

    def test_multimodal_aliases(self):
        for alias in ("multi", "openai", "openai-compat", "openai-compatible", "llm"):
            assert normalise_engine(alias) == "multimodal"

    def test_blank_defaults_to_multimodal(self):
        assert normalise_engine("") == "multimodal"
        assert normalise_engine("   ") == "multimodal"
        assert normalise_engine(None) == "multimodal"

    def test_unknown_passes_through(self):
        assert normalise_engine("custom-engine") == "custom-engine"

    def test_whitespace_is_stripped(self):
        assert normalise_engine("  rapid  ") == "rapidocr"


class TestEngineConfig:
    def test_siliconflow_resolves_base_url(self, monkeypatch):
        """Without OPENAI_BASE_URL, siliconflow falls back to its built-in URL."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert EngineConfig(name="siliconflow").resolved_base_url == SILICONFLOW_BASE_URL

    def test_base_url_trailing_slash_stripped(self):
        cfg = EngineConfig(name="multimodal", base_url="https://example.test/v1/")
        assert cfg.resolved_base_url == "https://example.test/v1"

    def test_multimodal_has_no_default_base_url(self, monkeypatch):
        """multimodal has no built-in URL of its own — only the env var or config."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert EngineConfig(name="multimodal").resolved_base_url == ""

    def test_siliconflow_default_model(self, monkeypatch):
        """Without any env override, siliconflow resolves to its built-in default."""
        monkeypatch.delenv("JYKJ_OCR_SILICONFLOW_MODEL", raising=False)
        assert EngineConfig(name="siliconflow").resolved_model == "PaddlePaddle/PaddleOCR-VL-1.5"

    def test_explicit_model_wins(self):
        cfg = EngineConfig(
            name="siliconflow", model="PaddlePaddle/PaddleOCR-VL-1.5"
        )
        assert cfg.resolved_model == "PaddlePaddle/PaddleOCR-VL-1.5"

    def test_env_model_overrides_default(self, monkeypatch):
        """``JYKJ_OCR_SILICONFLOW_MODEL`` lets operators swap models per deploy
        without editing the config file."""
        monkeypatch.setenv("JYKJ_OCR_SILICONFLOW_MODEL", "moonshotai/Kimi-K2.7-Code")
        assert EngineConfig(name="siliconflow").resolved_model == "moonshotai/Kimi-K2.7-Code"

    def test_explicit_model_beats_env_model(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_SILICONFLOW_MODEL", "env-model")
        cfg = EngineConfig(name="siliconflow", model="explicit-model")
        assert cfg.resolved_model == "explicit-model"

    def test_resolved_api_key_prefers_explicit(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert EngineConfig(api_key="explicit").resolved_api_key == "explicit"

    def test_resolved_api_key_env_precedence(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "generic")
        monkeypatch.setenv("SILICONFLOW_API_KEY", "specific")
        assert EngineConfig(name="siliconflow").resolved_api_key == "specific"

    def test_resolved_api_key_project_prefix_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "generic")
        monkeypatch.setenv("JYKJ_OCR_SILICONFLOW_API_KEY", "scoped")
        assert EngineConfig(name="siliconflow").resolved_api_key == "scoped"

    def test_resolved_api_key_generic_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "generic")
        assert EngineConfig(name="multimodal").resolved_api_key == "generic"

    def test_resolved_api_key_empty_when_unset(self):
        assert EngineConfig(name="multimodal").resolved_api_key == ""

    def test_merged_extra_is_a_copy(self):
        cfg = EngineConfig(extra={"prompt_file": "p.txt"})
        merged = cfg.merged_extra()
        merged["prompt_file"] = "changed"
        assert cfg.extra["prompt_file"] == "p.txt"

    def test_dedupe_key_uses_resolved_tuple(self):
        """Dedupe key is (resolved_name, base_url, model, api_key, lang, prompt)
        — 6-tuple. """
        a = EngineConfig(name="multimodal", base_url="https://a.test", model="m", api_key="k1")
        b = EngineConfig(name="multimodal", base_url="https://a.test", model="m", api_key="k2")
        c = EngineConfig(name="multimodal", base_url="https://a.test", model="m", api_key="k1")
        assert len(a.dedupe_key()) == 6
        assert a.dedupe_key() != b.dedupe_key()
        assert a.dedupe_key() == c.dedupe_key()

    def test_dedupe_key_keeps_distinct_lang(self):
        """Two local entries differing only in ``lang`` must NOT collapse —
        they issue different recogniser calls and both should survive."""
        a = EngineConfig(name="rapidocr", lang="ch")
        b = EngineConfig(name="rapidocr", lang="en")
        assert a.dedupe_key() != b.dedupe_key()

    def test_dedupe_key_keeps_distinct_prompt(self):
        """Same provider+model with different prompts are different calls."""
        a = EngineConfig(name="multimodal", model="m", prompt="p1")
        b = EngineConfig(name="multimodal", model="m", prompt="p2")
        assert a.dedupe_key() != b.dedupe_key()

    def test_from_mapping_keeps_both_lang_variants(self):
        """Regression: identical ``resolved_*`` tuples used to silently drop the
        second ``rapidocr`` entry when only ``lang`` differed."""
        cfg = from_mapping({
            "engines": [
                {"name": "rapidocr", "lang": "ch"},
                {"name": "rapidocr", "lang": "en"},
            ],
        })
        assert [e.lang for e in cfg.engines] == ["ch", "en"]

    def test_from_mapping_still_collapses_true_duplicates(self):
        """Exact duplicates (same lang+prompt) must still collapse to one entry
        so the pipeline doesn't issue the same network call twice."""
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "model": "m", "prompt": "p"},
                {"name": "multimodal", "model": "m", "prompt": "p"},
                {"name": "multimodal", "model": "m", "prompt": "different"},
            ],
        })
        assert len(cfg.engines) == 2
        assert [e.prompt for e in cfg.engines] == ["p", "different"]


class TestMultipleMultimodalInstances:
    """``multimodal`` is a type with unlimited instances; each engine instance
    owns its provider + model + key, and strategy scoring sees them as peers."""

    def test_each_entry_becomes_an_independent_engine_instance(self, monkeypatch):
        """OPENAI_* env vars provide shared defaults; each entry owns its
        base_url/model/api_key so instances aren't confused."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://default.test")
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "base_url": "https://api.siliconflow.cn/v1",
                 "model": "PaddlePaddle/PaddleOCR-VL-1.5", "api_key": "sk-sf"},
                {"name": "multimodal", "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                 "model": "doubao-1-5-vision-pro-32k", "api_key": "sk-ark"},
            ]
        })
        engines = [MultimodalEngine(ecfg) for ecfg in cfg.engines]
        assert [e.base_url for e in engines] == [
            "https://api.siliconflow.cn/v1",
            "https://ark.cn-beijing.volces.com/api/v3",
        ]
        assert [e.model_name for e in engines] == [
            "PaddlePaddle/PaddleOCR-VL-1.5",
            "doubao-1-5-vision-pro-32k",
        ]
        assert [e.api_key for e in engines] == ["sk-sf", "sk-ark"]

    def test_engine_id_is_stable_across_instances(self, monkeypatch):
        """engine_id() returns the type name so OCRResult.engine / score_mode
        dispatch is unaffected by how many multimodal entries exist."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test")
        a = MultimodalEngine(EngineConfig(name="multimodal", model="m1"))
        b = MultimodalEngine(EngineConfig(name="multimodal", model="m2"))
        assert a.engine_id() == b.engine_id() == "multimodal"

    def test_env_fallback_only_applies_when_entry_silent(self, monkeypatch):
        """OPENAI_BASE_URL supplies a default for a silent entry, but an entry
        with an explicit base_url keeps it (no bleed from sibling env vars)."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test")
        monkeypatch.delenv("JYKJ_OCR_MULTIMODAL_MODEL", raising=False)
        explicit = MultimodalEngine(EngineConfig(name="multimodal",
                                                 base_url="https://explicit.test"))
        silent = MultimodalEngine(EngineConfig(name="multimodal"))
        assert explicit.base_url == "https://explicit.test"
        assert silent.base_url == "https://env.test"

    def test_per_entry_model_beats_env_and_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_MODEL", "env-model")
        explicit = MultimodalEngine(EngineConfig(name="multimodal", model="mine"))
        silent = MultimodalEngine(EngineConfig(name="multimodal"))
        assert explicit.model_name == "mine"
        assert silent.model_name == "env-model"

    def test_engine_class_uses_instance_config_not_type_default(self, monkeypatch):
        """Model comes from the entry, not the type's default — critical
        when multiple multimodal entries coexist."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_MODEL", "env-model")
        explicit = MultimodalEngine(EngineConfig(name="multimodal", model="custom"))
        assert explicit.model_name == "custom"

    def test_siliconflow_default_url_without_env(self, monkeypatch):
        """With no env overrides a ``siliconflow`` entry resolves to the
        vendor's built-in URL and model — the zero-config path."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("JYKJ_OCR_SILICONFLOW_MODEL", raising=False)
        engine = MultimodalEngine(EngineConfig(name="siliconflow"))
        assert engine.base_url == SILICONFLOW_BASE_URL
        assert engine.model_name == "PaddlePaddle/PaddleOCR-VL-1.5"

    def test_siliconflow_entry_still_reads_openai_env(self, monkeypatch):
        """Documented precedence is explicit config -> OPENAI_BASE_URL ->
        vendor default, so a siliconflow entry honours the standard env var
        too (same as any other OpenAI-compatible entry)."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://other.test/v1")
        monkeypatch.delenv("JYKJ_OCR_SILICONFLOW_MODEL", raising=False)
        engine = MultimodalEngine(EngineConfig(name="siliconflow"))
        assert engine.base_url == "https://other.test/v1"
        assert engine.model_name == "PaddlePaddle/PaddleOCR-VL-1.5"

    def test_generic_multimodal_reads_openai_env(self, monkeypatch):
        """A generic ``multimodal`` entry with no explicit config falls back
        to OPENAI_BASE_URL / OPENAI_API_KEY — the "one token, any provider"
        path."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://any.test/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
        monkeypatch.delenv("JYKJ_OCR_MULTIMODAL_MODEL", raising=False)
        engine = MultimodalEngine(EngineConfig(name="multimodal"))
        assert engine.base_url == "https://any.test/v1"
        assert engine.api_key == "sk-generic"
        assert engine.model_name == "PaddlePaddle/PaddleOCR-VL-1.5"


class TestFromMapping:
    def test_engine_names_are_normalised(self):
        cfg = from_mapping({"engines": [{"name": "rapid", "model": "x"}]})
        assert cfg.engines[0].name == "rapidocr"

    def test_string_engine_entry(self):
        cfg = from_mapping({"engines": ["sf"]})
        assert [e.name for e in cfg.engines] == ["siliconflow"]

    def test_unknown_engine_keys_landed_in_extra(self):
        cfg = from_mapping(
            {"engines": [{"name": "multimodal", "image_format": "jpg", "timeout": 5}]}
        )
        engine = cfg.engines[0]
        assert engine.timeout == 5
        assert engine.extra["image_format"] == "jpg"

    def test_strategy_as_string(self):
        cfg = from_mapping({"strategy": "rapidocr"})
        assert cfg.strategy == {"engine": "rapidocr"}

    def test_output_as_string(self):
        assert from_mapping({"output": "markdown"}).output == {"format": "markdown"}

    def test_garbage_strategy_and_output_become_empty_dicts(self):
        cfg = from_mapping({"strategy": ["not", "a", "dict"], "output": 42})
        assert cfg.strategy == {}
        assert cfg.output == {}

    def test_empty_mapping_gives_default_multimodal(self):
        cfg = from_mapping({})
        assert len(cfg.engines) == 1
        assert cfg.engines[0].name == "multimodal"
        assert cfg.engines[0].enabled is True

    def test_is_defensive_copy(self):
        raw = {"engines": [{"name": "multimodal", "timeout": 9}], "strategy": {"a": 1}}
        cfg = from_mapping(raw)
        cfg.engines[0].timeout = 1
        assert raw["engines"][0]["timeout"] == 9

    def test_engine_lookup_uses_aliases(self):
        cfg = from_mapping({"engines": [{"name": "rapidocr"}]})
        assert cfg.find_engine("rapid") is cfg.engines[0]
        assert cfg.find_engine("siliconflow") is None

    def test_enabled_engines_filters(self):
        cfg = from_mapping(
            {
                "engines": [
                    {"name": "rapidocr", "enabled": True},
                    {"name": "multimodal", "enabled": False},
                ]
            }
        )
        assert [e.name for e in cfg.enabled_engines()] == ["rapidocr"]

    def test_strategy_value_default(self):
        cfg = from_mapping({"strategy": {"max_retries": 2}})
        assert cfg.strategy_value("max_retries") == 2
        assert cfg.strategy_value("missing", "dflt") == "dflt"


class TestLoadConfig:
    def test_yaml_from_explicit_path(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        path.write_text(
            "strategy:\n  max_retries: 3\nengines:\n  - name: rapid\n", encoding="utf-8"
        )
        cfg = load_config(str(path))
        assert cfg.strategy["max_retries"] == 3
        assert cfg.engines[0].name == "rapidocr"

    def test_json_from_explicit_path(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"output": {"format": "text"}}), encoding="utf-8")
        assert load_config(str(path)).output == {"format": "text"}

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "does-not-exist.yaml"))
        assert cfg.engines[0].name == "multimodal"
        assert cfg.strategy == {}

    def test_env_var_selects_config_file(self, tmp_path, monkeypatch):
        path = tmp_path / "from_env.yaml"
        path.write_text("output:\n  format: markdown\n", encoding="utf-8")
        monkeypatch.setenv("JYKJ_OCR_CONFIG", str(path))
        cfg = load_config(None)
        assert cfg.output == {"format": "markdown"}

    def test_env_var_missing_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_CONFIG", str(tmp_path / "nope.yaml"))
        assert load_config(None).engines[0].name == "multimodal"

    def test_load_prompt_reads_file(self, tmp_path):
        path = tmp_path / "prompt.txt"
        path.write_text("识别图中的文字。", encoding="utf-8")
        assert load_prompt(str(path)) == "识别图中的文字。"

    def test_load_prompt_missing_raises(self, tmp_path):
        with pytest.raises(OSError):
            load_prompt(str(tmp_path / "missing.txt"))
