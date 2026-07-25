from __future__ import annotations

import os
from pathlib import Path

from visionmodelquest.config import ModelDefinition


class CacheError(RuntimeError):
    """A pinned model snapshot is missing or incomplete."""


def default_cache_root() -> Path:
    configured = os.environ.get("VISIONMODELQUEST_HF_CACHE")
    if configured:
        return Path(configured).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repository_cache_name(model_id: str) -> str:
    organisation, name = model_id.split("/", 1)
    return f"models--{organisation}--{name}"


def resolve_snapshot(model: ModelDefinition, cache_root: Path | None = None) -> Path:
    if model.adapter == "mock":
        raise CacheError("the mock adapter does not use a model snapshot")
    root = (cache_root or default_cache_root()).resolve()
    snapshot = root / repository_cache_name(model.model_id) / "snapshots" / model.revision
    try:
        resolved = snapshot.resolve(strict=True)
    except FileNotFoundError as error:
        raise CacheError(
            f"Pinned snapshot is not cached: {model.model_id}@{model.revision}"
        ) from error
    if not resolved.is_dir() or not resolved.is_relative_to(root):
        raise CacheError("resolved snapshot escaped the configured cache root")
    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    missing = sorted(name for name in required if not (resolved / name).is_file())
    if not list(resolved.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        raise CacheError(f"Pinned snapshot is incomplete; missing: {', '.join(missing)}")
    return resolved


def cache_status(model: ModelDefinition, cache_root: Path | None = None) -> dict[str, object]:
    if model.adapter == "mock":
        return {"cached": True, "snapshot": None, "reason": None}
    try:
        snapshot = resolve_snapshot(model, cache_root)
        return {"cached": True, "snapshot": str(snapshot), "reason": None}
    except CacheError as error:
        return {"cached": False, "snapshot": None, "reason": str(error)}

