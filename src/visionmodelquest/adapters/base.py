from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from visionmodelquest.config import ModelDefinition


@dataclass(frozen=True)
class Generation:
    text: str
    preprocessing_seconds: float
    inference_seconds: float
    first_output_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str
    visual_tokens: int | None = None
    peak_gpu_memory_bytes: int | None = None


class ModelAdapter(ABC):
    adapter_version = "1.0.0"

    def __init__(self, definition: ModelDefinition, cache_root: Path | None = None) -> None:
        self.definition = definition
        self.cache_root = cache_root
        self.load_seconds = 0.0
        self.runtime_details: dict[str, object] = {}

    @abstractmethod
    def load(self) -> None:
        """Load and validate the exact local snapshot."""

    @abstractmethod
    def generate(self, image: Path, prompt: str) -> Generation:
        """Generate one deterministic response."""

    def close(self) -> None:
        """Release model resources."""
        return None
