from __future__ import annotations

import json
from pathlib import Path

from visionmodelquest.explorer.repository import ExperimentRepository


def revision_payload() -> dict[str, object]:
    return {
        "experiment_name": "Desk description",
        "model_key": "mock",
        "model_id": "local/mock",
        "model_revision": "mock-v1",
        "adapter_name": "mock",
        "adapter_version": "1.0.0",
        "response_contract": "free_text_v1",
        "system_instruction": "Describe visible evidence.",
        "user_question": "What is visible?",
        "visual_token_budget": 140,
        "completion_token_limit": 64,
        "image_reference": {"kind": "asset", "sha256": "a" * 64},
        "notes": "",
        "output_hash": None,
        "validation_state": None,
        "timing_summary": {},
        "preprocessing_inspection": None,
    }


def test_saving_changes_creates_immutable_revisions(tmp_path: Path) -> None:
    repository = ExperimentRepository(tmp_path / "experiments")
    first = repository.save(revision_payload())
    first_path = (
        tmp_path
        / "experiments"
        / first.experiment_id
        / "revision-0001.json"
    )
    original = first_path.read_bytes()
    changed = revision_payload()
    changed["user_question"] = "Count the visible items."

    second = repository.save(changed, experiment_id=first.experiment_id)

    assert second.revision_number == 2
    assert first_path.read_bytes() == original
    assert len(repository.list_revisions()) == 2
    index = json.loads((tmp_path / "experiments" / "index.json").read_text())
    assert index["experiments"][0]["latest_revision"] == 2
