# -*- coding: utf-8 -*-
"""Configuration loading and environment resolution.

Precedence (highest wins):

1. Explicit keyword arguments passed to :class:`Config.from_mapping`.
2. Environment variables (``JYKJ_OCR_*`` and engine-specific ``*_API_KEY``).
3. Values read from the YAML/JSON config file.
4. Built-in defaults.

Engine names are resolved here so the rest of the codebase never hard-codes a
provider string.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: SiliconFlow's default OpenAI-compatible base URL.
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

#: Prefix for the ``JYKJ_OCR_*`` environment variables.
_ENV_PREFIX = "JYKJ_OCR"

#: Aliases accepted for ``engines[].name`` / ``--engine``.
ENGINE_ALIASES: Dict[str, str] = {
    "rapid": "rapidocr",
    "rapid-ocr": "rapidocr",
    "rapidocr-onnx": "rapidocr",
    "sf": "siliconflow",
    "silicon-flow": "siliconflow",
    "silicon_flow": "siliconflow",
    "siliconflow": "siliconflow",
    "multi": "multimodal",
    "multimodal": "multimodal",
    "openai": "multimodal",
    "openai-compat": "multimodal",
    "openai-compatible": "multimodal",
    "llm": "multimodal",
}


def normalise_engine(name: str) -> str:
    """Map a user-supplied engine name onto a canonical engine id."""
    key = (name or "").strip().lower()
    if not key:
        return "multimodal"
    if key in ENGINE_ALIASES:
        return ENGINE_ALIASES[key]
    return key


@dataclass
class EngineConfig:
    """Per-engine settings. Unknown keys are preserved in :attr:`extra`."""

    name: str = "multimodal"
    enabled: bool = True
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    timeout: float = 120.0
    max_tokens: Optional[int] = None
    lang: str = "ch"
    prompt: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_base_url(self) -> str:
        """Resolve the base URL, checking env vars first.

        Order: explicit config -> ``OPENAI_BASE_URL`` (standard OpenAI SDK env
        var, so the same token works with any provider the user picks) ->
        engine-specific default (siliconflow). This keeps the remote engines
        provider-agnostic: set ``OPENAI_BASE_URL`` once and point it at any
        OpenAI-compatible endpoint.
        """
        if self.base_url:
            return self.base_url.rstrip("/")
        env_base = os.getenv("OPENAI_BASE_URL")
        if env_base:
            return env_base.rstrip("/")
        if self.name == "siliconflow":
            return SILICONFLOW_BASE_URL
        return ""

    @property
    def resolved_model(self) -> str:
        return self.model or (
            "PaddlePaddle/PaddleOCR-VL-1.5" if self.name == "siliconflow" else ""
        )

    @property
    def resolved_api_key(self) -> str:
        """Resolve the key, checking env vars first.

        Order: explicit config -> ``JYKJ_OCR_<NAME>_API_KEY`` -> ``<NAME>_API_KEY``
        -> ``OPENAI_API_KEY`` (so a generic token works with any provider).
        """
        if self.api_key:
            return self.api_key
        upper = self.name.upper().replace("-", "_").replace(".", "_")
        candidates = [
            f"{_ENV_PREFIX}_{upper}_API_KEY",
            f"{upper}_API_KEY",
            "OPENAI_API_KEY",
        ]
        for var in candidates:
            value = os.getenv(var)
            if value:
                return value
        return ""

    def merged_extra(self) -> Dict[str, Any]:
        return dict(self.extra)


_ENGINE_KNOWN_KEYS = {
    "name",
    "enabled",
    "model",
    "base_url",
    "api_key",
    "temperature",
    "timeout",
    "max_tokens",
    "lang",
    "prompt",
    "prompt_file",
}


@dataclass
class Config:
    """Top-level application configuration."""

    engines: List[EngineConfig] = field(default_factory=list)
    strategy: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    pdf: Dict[str, Any] = field(default_factory=dict)

    def engines_by_name(self) -> Dict[str, EngineConfig]:
        return {e.name: e for e in self.engines}

    def find_engine(self, name: str) -> Optional[EngineConfig]:
        want = normalise_engine(name)
        for engine in self.engines:
            if normalise_engine(engine.name) == want:
                return engine
        return None

    def enabled_engines(self) -> List[EngineConfig]:
        return [e for e in self.engines if e.enabled]

    def strategy_value(self, key: str, default: Any = None) -> Any:
        return self.strategy.get(key, default)

    def output_value(self, key: str, default: Any = None) -> Any:
        return self.output.get(key, default)


def _parse_engine(raw: Dict[str, Any]) -> EngineConfig:
    data = copy.deepcopy(raw or {})
    name = data.pop("name", "multimodal")
    engine = EngineConfig(name=normalise_engine(str(name)))
    for key, value in data.items():
        if key in _ENGINE_KNOWN_KEYS and hasattr(EngineConfig, key):
            setattr(engine, key, value)
        else:
            engine.extra[key] = value
    return engine


def from_mapping(data: Dict[str, Any]) -> Config:
    """Build a :class:`Config` from an already-parsed mapping."""
    data = copy.deepcopy(data or {})
    strategy = data.get("strategy") or {}
    output = data.get("output") or {}
    pdf = data.get("pdf") or {}

    if isinstance(strategy, str):
        strategy = {"engine": strategy}
    if isinstance(output, str):
        output = {"format": output}

    engines: List[EngineConfig] = []
    for raw in data.get("engines") or []:
        if isinstance(raw, str):
            engines.append(_parse_engine({"name": raw}))
        elif isinstance(raw, dict):
            engines.append(_parse_engine(raw))
    if not engines:
        engines.append(EngineConfig(name="multimodal"))

    return Config(
        engines=engines,
        strategy=strategy if isinstance(strategy, dict) else {},
        output=output if isinstance(output, dict) else {},
        pdf=pdf if isinstance(pdf, dict) else {},
    )


def _read_raw_file(path: str) -> Dict[str, Any]:
    """Read a YAML or JSON config file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    if path.lower().endswith(".json"):
        import json

        return json.loads(raw)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read .yaml config files. "
            "Install it with: pip install pyyaml"
        ) from exc
    return yaml.safe_load(raw) or {}


def load_config(path: Optional[str] = None) -> Config:
    """Load config from ``path``, ``JYKJ_OCR_CONFIG``, or a built-in default.

    The default points at ``config/config.yaml`` relative to the current working
    directory; a missing file is not an error, since every field has a fallback.
    """
    if path is None:
        path = os.getenv(f"{_ENV_PREFIX}_CONFIG")
    if not path:
        path = os.path.join("config", "config.yaml")
    if path and os.path.isfile(path):
        return from_mapping(_read_raw_file(path))
    return from_mapping({})


def load_prompt(path: str) -> str:
    """Read a prompt template from disk (used by the multimodal engine)."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


__all__ = [
    "Config",
    "EngineConfig",
    "SILICONFLOW_BASE_URL",
    "load_config",
    "load_prompt",
    "from_mapping",
    "normalise_engine",
]
