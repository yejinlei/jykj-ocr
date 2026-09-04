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
    def test_siliconflow_resolves_base_url(self):
        assert EngineConfig(name="siliconflow").resolved_base_url == SILICONFLOW_BASE_URL

    def test_base_url_trailing_slash_stripped(self):
        cfg = EngineConfig(name="multimodal", base_url="https://example.test/v1/")
        assert cfg.resolved_base_url == "https://example.test/v1"

    def test_multimodal_has_no_default_base_url(self):
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
