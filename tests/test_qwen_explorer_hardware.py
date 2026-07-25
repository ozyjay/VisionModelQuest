from __future__ import annotations

import os
from pathlib import Path

import pytest

from visionmodelquest.config import ROOT, load_models
from visionmodelquest.experiment_worker import ExperimentRuntime
from visionmodelquest.explorer.images import SessionImages

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.rocm,
    pytest.mark.large_model,
]


@pytest.mark.skipif(
    os.environ.get("VISIONMODELQUEST_RUN_HARDWARE") != "1",
    reason="set VISIONMODELQUEST_RUN_HARDWARE=1 for physical Qwen validation",
)
def test_qwen_processor_and_model_smoke(tmp_path: Path) -> None:
    images = SessionImages(tmp_path / "sessions")
    record = images.import_image(
        ROOT / "fixtures" / "images" / "simple_desk.ppm",
        fixture=True,
        provenance="Hardware smoke fixture",
    )
    definition = load_models()["qwen35-0.8b"]
    runtime = ExperimentRuntime(
        definition,
        images.root,
        cache_root=None,
        log_root=tmp_path / "logs",
    )

    processor = runtime.initialise_processor()
    inspection = runtime.inspect_image(record.image_id, 140)

    assert processor["processor_class"] == definition.expected_processor_class
    assert inspection["image_grid_thw"] == (1, 14, 20)
    assert inspection["processed_width"] == 320
    assert inspection["processed_height"] == 224
    assert inspection["actual_visual_tokens"] == 70

    loaded = runtime.load_model()
    assert loaded["runtime"]["model_class"] == definition.expected_model_class
    runtime.unload_model()
    images.close()
