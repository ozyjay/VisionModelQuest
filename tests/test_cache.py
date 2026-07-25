from pathlib import Path

import pytest

from visionmodelquest.cache import CacheError, cache_status, resolve_snapshot
from visionmodelquest.config import load_models


def test_missing_snapshot_is_a_structured_cache_status(tmp_path: Path):
    model = load_models()["qwen35-0.8b"]
    status = cache_status(model, tmp_path)
    assert status["cached"] is False
    assert "not cached" in str(status["reason"])


def test_complete_exact_snapshot_resolves(tmp_path: Path):
    model = load_models()["qwen35-0.8b"]
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen3.5-0.8B"
        / "snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (snapshot / name).touch()
    assert resolve_snapshot(model, tmp_path) == snapshot.resolve()


def test_incomplete_snapshot_is_rejected(tmp_path: Path):
    model = load_models()["qwen35-0.8b"]
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen3.5-0.8B"
        / "snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    with pytest.raises(CacheError, match="incomplete"):
        resolve_snapshot(model, tmp_path)

