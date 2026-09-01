# -*- coding: utf-8 -*-
"""SiliconFlow OCR engine.

A thin specialisation of the OpenAI-compatible engine: same protocol, with
SiliconFlow's defaults and model name pinned so a config can just say
``name: siliconflow`` and work.
"""

from __future__ import annotations

from ..config import EngineConfig, SILICONFLOW_BASE_URL
from ..engine.base import register
from .multimodal_engine import DEFAULT_MODEL, MultimodalEngine

#: Default SiliconFlow OCR model. Override with ``model:`` in config.
SILICONFLOW_OCR_MODEL = DEFAULT_MODEL


class SiliconFlowEngine(MultimodalEngine):
    """Multimodal OCR hosted on SiliconFlow."""

    name = "siliconflow"


@register("siliconflow")
def _siliconflow_factory(config: EngineConfig) -> SiliconFlowEngine:
    config = config or EngineConfig(name="siliconflow")
    # Fill SiliconFlow defaults only where the config did not specify one already.
    if not config.model:
        config.model = SILICONFLOW_OCR_MODEL
    if not config.base_url:
        config.base_url = SILICONFLOW_BASE_URL
    return SiliconFlowEngine(config)


__all__ = ["SILICONFLOW_OCR_MODEL", "SiliconFlowEngine"]
