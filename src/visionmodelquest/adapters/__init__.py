from __future__ import annotations

from pathlib import Path

from visionmodelquest.adapters.base import ModelAdapter
from visionmodelquest.adapters.gemma3 import Gemma3Adapter
from visionmodelquest.adapters.mock import MockAdapter
from visionmodelquest.adapters.qwen35 import Qwen35Adapter
from visionmodelquest.adapters.smolvlm2 import SmolVLM2Adapter
from visionmodelquest.config import ModelDefinition

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "mock": MockAdapter,
    "qwen35": Qwen35Adapter,
    "smolvlm2": SmolVLM2Adapter,
    "gemma3": Gemma3Adapter,
}


def create_adapter(
    definition: ModelDefinition, cache_root: Path | None = None
) -> ModelAdapter:
    adapter_class = ADAPTERS.get(definition.adapter)
    if adapter_class is None:
        raise ValueError(f"adapter is not allowlisted: {definition.adapter}")
    return adapter_class(definition, cache_root)
