from pathlib import Path

import pytest

from visionmodelquest.adapters.base import Generation
from visionmodelquest.adapters.mock import MockAdapter
from visionmodelquest.benchmarks.runner import deterministic_hash, run_adapter, validate_image
from visionmodelquest.config import PRESETS, ROOT, load_models, load_workload


def test_all_checked_in_images_are_valid():
    fixtures, _ = load_workload()
    for fixture in fixtures.fixtures:
        details = validate_image(ROOT / "fixtures" / fixture.image)
        assert details["width"] >= 1024
        assert details["height"] >= 768
        assert details["encoded_bytes"] > 0
        assert details["format"] == "PNG"


def test_invalid_image_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.png"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fixture"):
        validate_image(path)


def test_mock_quick_run_is_deterministic_and_privacy_safe():
    definition = load_models()["mock"]
    fixtures, questions = load_workload()
    selected = {"simple_desk"}
    first = run_adapter(
        definition,
        MockAdapter(definition),
        fixtures,
        questions,
        PRESETS["quick"],
        root=ROOT,
        selected_fixture_ids=selected,
    )
    second = run_adapter(
        definition,
        MockAdapter(definition),
        fixtures,
        questions,
        PRESETS["quick"],
        root=ROOT,
        selected_fixture_ids=selected,
    )
    first_hashes = [sample["output_hash"] for sample in first["samples"]]
    second_hashes = [sample["output_hash"] for sample in second["samples"]]
    assert first["status"] == "passed"
    assert first_hashes == second_hashes
    assert all("quality_capture" not in sample for sample in first["samples"])
    assert all("prompt" not in sample and "output" not in sample for sample in first["samples"])


def test_quality_capture_is_explicit():
    definition = load_models()["mock"]
    fixtures, questions = load_workload()
    result = run_adapter(
        definition,
        MockAdapter(definition),
        fixtures,
        questions,
        PRESETS["quick"],
        root=ROOT,
        selected_fixture_ids={"simple_desk"},
        quality_capture=True,
    )
    assert all("quality_capture" in sample for sample in result["samples"])


def test_quality_capture_retains_invalid_output_only_when_explicit():
    definition = load_models()["mock"]
    fixtures, questions = load_workload()

    class InvalidAdapter(MockAdapter):
        def generate(self, image: Path, prompt: str) -> Generation:
            return Generation(
                text="not valid JSON",
                preprocessing_seconds=0.0,
                inference_seconds=0.1,
                first_output_seconds=0.1,
                prompt_tokens=1,
                completion_tokens=3,
                finish_reason="stop",
            )

    captured = run_adapter(
        definition,
        InvalidAdapter(definition),
        fixtures,
        questions,
        PRESETS["quick"],
        root=ROOT,
        selected_fixture_ids={"simple_desk"},
        quality_capture=True,
    )
    failed = [
        sample
        for sample in captured["samples"]
        if sample["contract"] == "scene_json_v1"
    ]
    assert all(sample["quality_capture"]["output"] == "not valid JSON" for sample in failed)
    assert captured["status"] == "failed"
    assert captured["failure_category"] == "output_invalid"


def test_deterministic_hash():
    assert deterministic_hash("same") == deterministic_hash("same")
    assert deterministic_hash("same") != deterministic_hash("different")


def test_missing_gpu_is_a_structured_model_failure(tmp_path: Path):
    definition = load_models()["qwen35-0.8b"]
    fixtures, questions = load_workload()

    class MissingGpuAdapter:
        load_seconds = 0.0
        runtime_details = {}

        def load(self):
            raise RuntimeError("ROCm PyTorch did not expose an available cuda device")

        def close(self):
            pass

    result = run_adapter(
        definition,
        MissingGpuAdapter(),
        fixtures,
        questions,
        PRESETS["quick"],
        root=ROOT,
        selected_fixture_ids={"simple_desk"},
    )
    assert result["status"] == "failed"
    assert result["failure_category"] == "hardware_unavailable"
