# -*- coding: utf-8 -*-
"""Configuration loading and environment resolution.

Precedence (highest wins):

1. Explicit keyword arguments passed to :class:`Config.from_mapping`.
2. Environment variables (``JYKJ_OCR_*`` and engine-specific ``*_API_KEY``).
3. Values read from the YAML/JSON config file.
4. Built-in defaults.

Engine names are resolved here so the rest of the codebase never hard-codes a
provider string. A single config may list unlimited ``multimodal`` entries —
each points at a different provider (SiliconFlow, Moonshot, DeepSeek,
DashScope, Ark, ...) distinguished by ``(base_url, model, api_key)``. Entries
with identical resolved tuples collapse at parse time (first wins).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Prefix for the ``JYKJ_OCR_*`` environment variables.
_ENV_PREFIX = "JYKJ_OCR"


def load_dotenv(path: str = ".env") -> None:
    """Best-effort ``.env`` loader (stdlib only).

    Existing env vars always win so an operator's explicit export is never
    overwritten. Both the CLI entrypoint and :func:`jykj_ocr.server.create_app`
    call this — uvicorn loads ``server:app`` directly, skipping the CLI.
    """
    import os as _os

    if not _os.path.isfile(path):
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
                if key and key not in _os.environ:
                    _os.environ[key] = value
    except OSError:
        return

#: Aliases accepted for ``engines[].name`` / ``--engine``.
ENGINE_ALIASES: Dict[str, str] = {
    "rapid": "rapidocr",
    "rapid-ocr": "rapidocr",
    "rapidocr-onnx": "rapidocr",
    "sf": "multimodal",
    "silicon-flow": "multimodal",
    "silicon_flow": "multimodal",
    "siliconflow": "multimodal",
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
    #: 1-based index among entries of the same canonical type. Filled in by
    #: :func:`from_mapping` in declared order, or pinned explicitly in yaml to
    #: keep an entry's key variable stable across reorderings. Lets every
    #: instance of one type read its own key (``JYKJ_OCR_MULTIMODAL_1_API_KEY``,
    #: ``_2_``, …). ``None`` when built by hand.
    instance: Optional[int] = None

    @property
    def resolved_base_url(self) -> str:
        """Resolve the base URL, checking env vars first.

        Order: explicit config -> ``OPENAI_BASE_URL_<N>`` (only when
        :attr:`instance` is set, so ``N`` entries of one type can point at
        ``N`` different providers) -> ``OPENAI_BASE_URL`` ->
        ``JYKJ_OCR_<NAME>_BASE_URL``. There is deliberately no vendor default:
        every remote instance must name its endpoint one of those ways,
        otherwise a key for provider A could be silently sent to provider B.
        """
        if self.base_url:
            return self.base_url.rstrip("/")
        for var in self.base_url_env_names():
            value = os.getenv(var)
            if value:
                return value.rstrip("/")
        return ""

    @property
    def resolved_model(self) -> str:
        """Resolve the model name, checking env vars first.

        Order: explicit config -> ``JYKJ_OCR_<NAME>_<N>_MODEL`` (only when
        :attr:`instance` is set) -> ``JYKJ_OCR_<NAME>_MODEL``. Models are
        platform-specific, so there is no engine-level default: an operator
        can still swap models per deployment via the env var (e.g.
        ``JYKJ_OCR_MULTIMODAL_MODEL=moonshotai/Kimi-K2.7-Code``) without
        editing the config file.
        """
        if self.model:
            return self.model
        for var in self.model_env_names():
            value = os.getenv(var)
            if value:
                return value
        return ""

    def _env_upper(self) -> str:
        """Env-var stem for this engine type (``multimodal`` -> ``MULTIMODAL``)."""
        return self.resolved_name.upper().replace("-", "_").replace(".", "_")

    def _instance_env_names(self, templates: List[str], generic: List[str]) -> List[str]:
        """Expand ``<N>`` templates by :attr:`instance`, numbered candidates first.

        ``templates`` use ``<N>`` as the instance placeholder — ``N`` entries of
        one type each get their own variable, without editing the config file.
        ``generic`` are the instance-less names that every entry of the type
        still falls back to. Numbered names go first so they always win.
        """
        out: List[str] = []
        if self.instance:
            out.extend(t.replace("<N>", str(self.instance)) for t in templates)
        out.extend(generic)
        return out

    def base_url_env_names(self) -> List[str]:
        """Ordered environment-variable candidates for :attr:`resolved_base_url`.

        ``OPENAI_BASE_URL_<N>`` lets ``N`` entries of one type point at ``N``
        different providers while the yaml stays free of URLs. The bare
        ``OPENAI_BASE_URL`` is shared by every entry of the type — that is what
        made two entries at two different providers send one platform's key to
        the other. There is no engine-specific default: an unnamed endpoint
        must fail rather than fall back to some vendor.
        """
        upper = self._env_upper()
        return self._instance_env_names(
            ["OPENAI_BASE_URL_<N>"],
            ["OPENAI_BASE_URL", f"{_ENV_PREFIX}_{upper}_BASE_URL"],
        )

    def model_env_names(self) -> List[str]:
        """Ordered environment-variable candidates for :attr:`resolved_model`.

        ``JYKJ_OCR_<NAME>_<N>_MODEL`` lets each entry name its own model, so
        the shared ``JYKJ_OCR_<NAME>_MODEL`` only needs to exist for the
        single-platform case.
        """
        upper = self._env_upper()
        return self._instance_env_names(
            [f"{_ENV_PREFIX}_{upper}_<N>_MODEL"],
            [f"{_ENV_PREFIX}_{upper}_MODEL"],
        )

    def api_key_env_names(self) -> List[str]:
        """Ordered environment-variable candidates for :attr:`resolved_api_key`.

        Exposed so the engines can name the exact variables they looked up in
        their error messages, instead of telling an operator to check a single
        generic name.
        """
        upper = self._env_upper()
        return self._instance_env_names(
            [f"{_ENV_PREFIX}_{upper}_<N>_API_KEY", f"{upper}_<N>_API_KEY"],
            [f"{_ENV_PREFIX}_{upper}_API_KEY", f"{upper}_API_KEY", "OPENAI_API_KEY"],
        )

    @property
    def resolved_api_key(self) -> str:
        """Resolve the key, checking env vars first.

        Order: explicit config -> ``JYKJ_OCR_<NAME>_<N>_API_KEY`` /
        ``<NAME>_<N>_API_KEY`` (only when :attr:`instance` is set) ->
        ``JYKJ_OCR_<NAME>_API_KEY`` -> ``<NAME>_API_KEY`` -> ``OPENAI_API_KEY``
        (so a generic token still works with any provider). The bare,
        instance-less variables are shared by every entry of the type, which is
        why two entries pointing at two different providers 401 — that is the
        signal to switch to the instance-scoped names.
        """
        if self.api_key:
            return self.api_key
        for var in self.api_key_env_names():
            value = os.getenv(var)
            if value:
                return value
        return ""

    def merged_extra(self) -> Dict[str, Any]:
        return dict(self.extra)

    def dedupe_key(self) -> Tuple[str, str, str, str, str, str]:
        """Distinguishing tuple used to collapse duplicates at parse time.

        Two entries with identical ``(resolved_name, resolved_base_url,
        resolved_model, resolved_api_key, lang, prompt)`` are the same
        provider+model combination issuing the same call; the first wins.

        ``resolved_name`` keeps an unrecognised engine name (e.g. ``acme-vl``)
        distinct from another entry with the same defaults — every engine type
        is its own instance. ``lang`` and ``prompt`` are the two named fields
        that change behaviour without changing the network call, so omitting
        them silently dropped a configured entry (e.g. two ``rapidocr`` rows,
        one ``lang: ch`` and one ``lang: en`` — only the first survived).
        """
        return (
            self.resolved_name,
            self.resolved_base_url,
            self.resolved_model,
            self.resolved_api_key,
            self.lang,
            self.prompt,
        )

    @property
    def resolved_name(self) -> str:
        """Canonical engine id after alias folding (e.g. ``sf`` → ``multimodal``)."""
        return normalise_engine(self.name)


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
    "instance",
}


def _assign_instances(engines: List["EngineConfig"]) -> None:
    """Number each type's instances 1..N in declared order (in place).

    The index is what lets ``N`` entries of one type each resolve a distinct key
    (``JYKJ_OCR_MULTIMODAL_1_API_KEY``, ``_2_``, …) while the yaml stays free of
    secrets. An operator may pin an entry's ``instance`` explicitly to keep its
    key stable when entries are reordered; unnumbered entries skip the pinned
    numbers. Called before dedupe so the surviving entry keeps the index its
    operator declared.
    """
    next_index: Dict[str, int] = {}
    for engine in engines:
        key = engine.resolved_name
        if not engine.instance:
            engine.instance = next_index.get(key, 0) + 1
        next_index[key] = max(next_index.get(key, 0), engine.instance)


def _dedupe_engines(engines: List["EngineConfig"]) -> List["EngineConfig"]:
    """Collapse entries with the same resolved ``(name, base_url, model, api_key)``.

    ``multimodal`` is a type with unlimited instances; entries differ by
    base_url / model / api_key. Identical entries would waste a network
    call, so the first wins.
    """
    seen: Dict[Tuple[str, str, str, str, str, str], "EngineConfig"] = {}
    for engine in engines:
        key = engine.dedupe_key()
        seen.setdefault(key, engine)
    return list(seen.values())


@dataclass
class Config:
    """Top-level application configuration."""

    engines: List[EngineConfig] = field(default_factory=list)
    strategy: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    pdf: Dict[str, Any] = field(default_factory=dict)

    def engines_by_name(self) -> Dict[str, EngineConfig]:
        """Group engines by canonical type. Multiple instances of the same
        type (e.g. two ``multimodal`` entries pointing at different vendors)
        collapse to the first entry — use :meth:`engines_of_type` for all.
        """
        out: Dict[str, EngineConfig] = {}
        for engine in self.engines:
            out.setdefault(engine.resolved_name, engine)
        return out

    def engines_of_type(self, engine_type: str) -> List[EngineConfig]:
        """All entries whose canonical type equals ``engine_type``."""
        want = normalise_engine(engine_type)
        return [e for e in self.engines if e.resolved_name == want]

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
    _assign_instances(engines)
    engines = _dedupe_engines(engines)

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
    "load_config",
    "load_prompt",
    "from_mapping",
    "normalise_engine",
]
