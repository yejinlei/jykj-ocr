# -*- coding: utf-8 -*-
"""Tests for configuration loading, aliases, and env-var resolution."""

from __future__ import annotations

import json

import pytest

from jykj_ocr.config import (
    EngineConfig,
    from_mapping,
    load_config,
    load_prompt,
    normalise_engine,
)
from jykj_ocr.engines.multimodal_engine import MultimodalEngine

_KEY_ENVS = (
    "OPENAI_API_KEY",
    "MULTIMODAL_API_KEY",
    "JYKJ_OCR_MULTIMODAL_API_KEY",
) + tuple(
    f"{name}_{n}_API_KEY"
    for n in (1, 2, 3)
    for name in ("JYKJ_OCR_MULTIMODAL", "MULTIMODAL")
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

    def test_siliconflow_aliases_fold_to_multimodal(self):
        """SiliconFlow is one configured instance of multimodal, not its own
        engine type — its aliases must land on the generic type."""
        for alias in ("sf", "silicon-flow", "silicon_flow", "SILICONFLOW"):
            assert normalise_engine(alias) == "multimodal"

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
    def test_siliconflow_has_no_built_in_base_url(self, monkeypatch):
        """No vendor default URL: every remote instance must name its endpoint.
        Silent vendor fallback would route one platform's key to another."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert EngineConfig(name="siliconflow").resolved_base_url == ""

    def test_base_url_trailing_slash_stripped(self):
        cfg = EngineConfig(name="multimodal", base_url="https://example.test/v1/")
        assert cfg.resolved_base_url == "https://example.test/v1"

    def test_multimodal_has_no_default_base_url(self, monkeypatch):
        """multimodal has no built-in URL of its own — only the env var or config."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert EngineConfig(name="multimodal").resolved_base_url == ""

    def test_siliconflow_has_no_built_in_model(self, monkeypatch):
        """Models are platform-specific, so no engine-level default survives."""
        monkeypatch.delenv("JYKJ_OCR_MULTIMODAL_MODEL", raising=False)
        assert EngineConfig(name="siliconflow").resolved_model == ""
        assert EngineConfig(name="multimodal").resolved_model == ""

    def test_explicit_model_wins(self):
        cfg = EngineConfig(name="multimodal", model="PaddleOCR-VL-1.5")
        assert cfg.resolved_model == "PaddleOCR-VL-1.5"

    def test_env_model_is_used(self, monkeypatch):
        """``JYKJ_OCR_MULTIMODAL_MODEL`` lets operators swap models per deploy
        without editing the config file."""
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_MODEL", "moonshotai/Kimi-K2.7-Code")
        assert EngineConfig(name="multimodal").resolved_model == "moonshotai/Kimi-K2.7-Code"

    def test_explicit_model_beats_env_model(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_MODEL", "env-model")
        cfg = EngineConfig(name="multimodal", model="explicit-model")
        assert cfg.resolved_model == "explicit-model"

    def test_resolved_api_key_prefers_explicit(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert EngineConfig(api_key="explicit").resolved_api_key == "explicit"

    def test_resolved_api_key_env_precedence(self, monkeypatch):
        """The type-scoped var serves every instance of the type, so distinct
        accounts need an explicit ``api_key`` on the entry — or the numbered
        per-instance variables (see TestInstanceScopedKeys)."""
        monkeypatch.setenv("OPENAI_API_KEY", "generic")
        monkeypatch.setenv("MULTIMODAL_API_KEY", "specific")
        assert EngineConfig(name="multimodal").resolved_api_key == "specific"

    def test_resolved_api_key_project_prefix_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "generic")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_API_KEY", "scoped")
        assert EngineConfig(name="multimodal").resolved_api_key == "scoped"

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

    def test_engine_requires_a_base_url(self, monkeypatch):
        """No vendor default URL, so an unnamed endpoint fails fast here rather
        than sending one platform's key to another platform and surfacing as
        an opaque HTTP 401 downstream."""
        from jykj_ocr.engine.base import EngineNotAvailable

        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(EngineNotAvailable, match="base URL"):
            MultimodalEngine(EngineConfig(name="multimodal", api_key="k"))

    def test_no_key_error_names_the_numbered_vars(self, monkeypatch):
        """The operator must be told which per-instance variable to set, not
        pointed at a single shared name that every entry already reads."""
        from jykj_ocr.engine.base import EngineNotAvailable, PageImage
        from PIL import Image

        monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test")
        engine = MultimodalEngine(EngineConfig(name="multimodal", instance=2))
        page = PageImage(pil_image=Image.new("RGB", (10, 10)))
        with pytest.raises(EngineNotAvailable, match="JYKJ_OCR_MULTIMODAL_2_API_KEY"):
            engine.recognise(page)

    def test_alias_entry_reads_openai_env(self, monkeypatch):
        """A config entry still written as ``siliconflow`` (legacy) resolves
        exactly like any other multimodal entry: explicit config ->
        OPENAI_BASE_URL, no vendor fallback."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://other.test/v1")
        monkeypatch.delenv("JYKJ_OCR_MULTIMODAL_MODEL", raising=False)
        engine = MultimodalEngine(EngineConfig(name="siliconflow"))
        assert engine.base_url == "https://other.test/v1"
        assert engine.engine_id() == "multimodal"

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
        # Last-resort model is the bare OCR-VL-1.5 ID; a configured platform
        # should always supply its own (vendor-prefixed or bare) model ID.
        assert engine.model_name == "PaddleOCR-VL-1.5"


class TestInstanceScopedKeys:
    """``multimodal`` is a type with unlimited instances, so the bare, type-scoped
    key variables cannot serve two entries at two different providers. The
    numbered per-instance variables do, without putting secrets in the yaml."""

    def test_instances_are_numbered_in_declared_order(self):
        cfg = from_mapping({
            "engines": [
                {"name": "rapidocr"},
                {"name": "multimodal", "model": "m1"},
                {"name": "multimodal", "model": "m2"},
                {"name": "multimodal", "model": "m3"},
            ],
        })
        assert [e.instance for e in cfg.engines] == [1, 1, 2, 3]

    def test_numbering_is_per_type(self):
        """rapidocr instances never consume multimodal numbers."""
        cfg = from_mapping({
            "engines": [
                {"name": "rapidocr", "lang": "ch"},
                {"name": "multimodal", "model": "m1"},
                {"name": "rapidocr", "lang": "en"},
                {"name": "multimodal", "model": "m2"},
            ],
        })
        assert [e.instance for e in cfg.engines] == [1, 1, 2, 2]

    def test_api_key_env_names_include_instance_scoped_first(self):
        cfg = EngineConfig(name="multimodal", instance=2)
        assert cfg.api_key_env_names() == [
            "JYKJ_OCR_MULTIMODAL_2_API_KEY",
            "MULTIMODAL_2_API_KEY",
            "JYKJ_OCR_MULTIMODAL_API_KEY",
            "MULTIMODAL_API_KEY",
            "OPENAI_API_KEY",
        ]

    def test_api_key_env_names_without_instance_are_shared(self):
        """A hand-built config (``instance`` unset) keeps the legacy 3 names."""
        assert EngineConfig(name="multimodal").api_key_env_names() == [
            "JYKJ_OCR_MULTIMODAL_API_KEY",
            "MULTIMODAL_API_KEY",
            "OPENAI_API_KEY",
        ]

    def test_entry_reads_its_own_numbered_key(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_1_API_KEY", "key-1")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_2_API_KEY", "key-2")
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "model": "m1"},
                {"name": "multimodal", "model": "m2"},
            ]
        })
        assert [e.resolved_api_key for e in cfg.engines] == ["key-1", "key-2"]

    def test_instance_scoped_beats_shared_name(self, monkeypatch):
        """The whole point: two entries, two providers, one key each."""
        monkeypatch.setenv("OPENAI_API_KEY", "shared")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_1_API_KEY", "key-1")
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "model": "m1"},
                {"name": "multimodal", "model": "m2"},
            ]
        })
        assert cfg.engines[0].resolved_api_key == "key-1"
        # Entry 2 has no numbered var, so it falls back to the shared one.
        assert cfg.engines[1].resolved_api_key == "shared"

    def test_explicit_api_key_beats_numbered_env(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_1_API_KEY", "key-1")
        cfg = from_mapping({
            "engines": [{"name": "multimodal", "api_key": "inline"}]
        })
        assert cfg.engines[0].resolved_api_key == "inline"

    def test_empty_numbered_value_is_skipped(self, monkeypatch):
        """An empty placeholder cannot masquerade as a real key."""
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_1_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "shared")
        cfg = from_mapping({"engines": [{"name": "multimodal", "model": "m1"}]})
        assert cfg.engines[0].resolved_api_key == "shared"

    def test_instance_is_a_yaml_field(self):
        """``instance`` must survive the from_mapping round trip."""
        cfg = from_mapping({
            "engines": [{"name": "multimodal", "instance": 2, "model": "m"}]
        })
        assert cfg.engines[0].instance == 2

    def test_pinned_instance_is_honored_and_skipped(self):
        """Pinning keeps an entry's key variable stable across reorderings."""
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "model": "pinned", "instance": 1},
                {"name": "multimodal", "model": "other"},
                {"name": "multimodal", "model": "third"},
            ]
        })
        assert [e.instance for e in cfg.engines] == [1, 2, 3]

    def test_indices_survive_dedupe(self):
        """Dedupe runs after numbering, so the survivor keeps the index it
        was declared with — gaps stay where duplicates were removed."""
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "model": "m"},
                {"name": "multimodal", "model": "m"},
                {"name": "multimodal", "model": "n"},
            ]
        })
        assert [e.instance for e in cfg.engines] == [1, 3]
        assert [e.model for e in cfg.engines] == ["m", "n"]

    def test_dedupe_ignores_instance(self):
        """The index is a config address, not part of the network call — the
        dedupe tuple stays 6 fields regardless of how many instances exist."""
        a = EngineConfig(name="multimodal", model="m", instance=1)
        b = EngineConfig(name="multimodal", model="m", instance=2)
        assert a.dedupe_key() == b.dedupe_key()
        assert len(a.dedupe_key()) == 6

    def test_engine_config_carries_instance(self, monkeypatch):
        """The effective Config handed to the engines keeps its index, so each
        entry reads its own numbered key."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test")
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal", "model": "m1"},
                {"name": "multimodal", "model": "m2"},
            ]
        })
        assert [e.instance for e in cfg.engines] == [1, 2]

    def test_snapshot_exposes_instance(self, monkeypatch):
        """``RuntimeConfig.snapshot()`` round-trips the index, so ``GET /config``
        can show which numbered variable each entry reads."""
        from jykj_ocr.server import RuntimeConfig

        monkeypatch.setenv("OPENAI_BASE_URL", "https://x.test")
        runtime = RuntimeConfig(from_mapping({
            "engines": [
                {"name": "multimodal", "model": "m1"},
                {"name": "multimodal", "model": "m2"},
            ]
        }))
        assert [e.instance for e in runtime.snapshot().engines] == [1, 2]


class TestInstanceScopedBaseUrlAndModel:
    """The same numbered scheme extends to ``base_url`` and ``model``, so a
    whole entry can be assembled from env vars with zero secrets in the yaml."""

    def test_base_url_env_names_numbered_first(self):
        assert EngineConfig(name="multimodal", instance=2).base_url_env_names() == [
            "OPENAI_BASE_URL_2",
            "OPENAI_BASE_URL",
            "JYKJ_OCR_MULTIMODAL_BASE_URL",
        ]

    def test_base_url_env_names_without_instance(self):
        assert EngineConfig(name="multimodal").base_url_env_names() == [
            "OPENAI_BASE_URL",
            "JYKJ_OCR_MULTIMODAL_BASE_URL",
        ]

    def test_model_env_names(self):
        assert EngineConfig(name="multimodal", instance=2).model_env_names() == [
            "JYKJ_OCR_MULTIMODAL_2_MODEL",
            "JYKJ_OCR_MULTIMODAL_MODEL",
        ]
        assert EngineConfig(name="multimodal").model_env_names() == [
            "JYKJ_OCR_MULTIMODAL_MODEL",
        ]

    def test_each_entry_reads_its_own_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL_1", "https://api.siliconflow.cn/v1/")
        monkeypatch.setenv("OPENAI_BASE_URL_2", "https://api.moark.com/v1")
        cfg = from_mapping({
            "engines": [
                {"name": "multimodal"},
                {"name": "multimodal"},
            ]
        })
        assert [e.resolved_base_url for e in cfg.engines] == [
            "https://api.siliconflow.cn/v1",
            "https://api.moark.com/v1",
        ]

    def test_numbered_base_url_beats_shared(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://shared.test")
        monkeypatch.setenv("OPENAI_BASE_URL_1", "https://numbered.test")
        cfg = from_mapping({"engines": [{"name": "multimodal"}]})
        assert cfg.engines[0].resolved_base_url == "https://numbered.test"

    def test_explicit_base_url_beats_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL_1", "https://numbered.test")
        cfg = from_mapping({
            "engines": [{"name": "multimodal", "base_url": "https://explicit.test"}]
        })
        assert cfg.engines[0].resolved_base_url == "https://explicit.test"

    def test_each_entry_reads_its_own_model(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_1_MODEL", "model-1")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_2_MODEL", "model-2")
        cfg = from_mapping({"engines": [{"name": "multimodal"}, {"name": "multimodal"}]})
        assert [e.resolved_model for e in cfg.engines] == ["model-1", "model-2"]

    def test_numbered_model_beats_shared_model(self, monkeypatch):
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_MODEL", "shared")
        monkeypatch.setenv("JYKJ_OCR_MULTIMODAL_1_MODEL", "numbered")
        cfg = from_mapping({"engines": [{"name": "multimodal"}]})
        assert cfg.engines[0].resolved_model == "numbered"

    def test_full_entry_from_env_vars_only(self, monkeypatch):
        """Three entries, no yaml fields at all beyond ``name``."""
        for n, (url, key, model) in enumerate(
            [
                ("https://api.siliconflow.cn/v1", "key-1", "PaddlePaddle/PaddleOCR-VL-1.5"),
                ("https://api.moark.com/v1", "key-2", "PaddleOCR-VL-1.5"),
                ("https://dashscope.aliyuncs.com/compatible-mode/v1", "key-3", "qwen-vl-max"),
            ],
            start=1,
        ):
            monkeypatch.setenv(f"OPENAI_BASE_URL_{n}", url)
            monkeypatch.setenv(f"JYKJ_OCR_MULTIMODAL_{n}_API_KEY", key)
            monkeypatch.setenv(f"JYKJ_OCR_MULTIMODAL_{n}_MODEL", model)
        cfg = from_mapping({"engines": [{"name": "multimodal"}] * 3})
        assert [e.instance for e in cfg.engines] == [1, 2, 3]
        assert [e.resolved_base_url for e in cfg.engines] == [
            "https://api.siliconflow.cn/v1",
            "https://api.moark.com/v1",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ]
        assert [e.resolved_api_key for e in cfg.engines] == ["key-1", "key-2", "key-3"]
        assert [e.resolved_model for e in cfg.engines] == [
            "PaddlePaddle/PaddleOCR-VL-1.5",
            "PaddleOCR-VL-1.5",
            "qwen-vl-max",
        ]

    def test_missing_base_url_error_names_numbered_vars(self, monkeypatch):
        """An unnamed endpoint must fail fast and say which variable to set."""
        from jykj_ocr.engine.base import EngineNotAvailable

        for v in ("OPENAI_BASE_URL", "OPENAI_BASE_URL_2", "JYKJ_OCR_MULTIMODAL_BASE_URL"):
            monkeypatch.delenv(v, raising=False)
        with pytest.raises(EngineNotAvailable, match="OPENAI_BASE_URL_2"):
            MultimodalEngine(EngineConfig(name="multimodal", instance=2))


class TestFromMapping:
    def test_engine_names_are_normalised(self):
        cfg = from_mapping({"engines": [{"name": "rapid", "model": "x"}]})
        assert cfg.engines[0].name == "rapidocr"

    def test_string_engine_entry(self):
        """The SiliconFlow shorthand collapses to the generic remote type."""
        cfg = from_mapping({"engines": ["sf"]})
        assert [e.name for e in cfg.engines] == ["multimodal"]

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
        assert cfg.find_engine("multimodal") is None

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
