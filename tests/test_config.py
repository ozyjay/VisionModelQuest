import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from visionmodelquest.config import ModelDefinition, load_models, load_workload


def test_checked_in_configuration_is_valid_and_allowlisted():
    models = load_models()
    assert set(models) == {
        "mock",
        "qwen35-0.8b",
        "qwen35-2b",
        "qwen35-4b",
        "smolvlm2-2.2b",
        "gemma3-4b",
    }
    assert all(model.local_files_only for model in models.values())
    assert all(not model.trust_remote_code for model in models.values())


def test_real_model_requires_full_exact_revision():
    payload = load_models()["qwen35-0.8b"].model_dump()
    payload["revision"] = "main"
    with pytest.raises(ValidationError, match="40-character"):
        ModelDefinition.model_validate(payload)


def test_adapter_is_an_allowlist():
    payload = load_models()["mock"].model_dump()
    payload["adapter"] = "arbitrary.module.Class"
    with pytest.raises(ValidationError):
        ModelDefinition.model_validate(payload)


def test_local_files_only_and_remote_code_cannot_be_relaxed():
    payload = load_models()["qwen35-0.8b"].model_dump()
    payload["local_files_only"] = False
    payload["trust_remote_code"] = True
    with pytest.raises(ValidationError):
        ModelDefinition.model_validate(payload)


def test_workload_rejects_unknown_question(tmp_path: Path):
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fixtures": [
                    {
                        "id": "example",
                        "image": "images/example.ppm",
                        "licence": "CC0",
                        "provenance": "test",
                        "allowed_questions": ["missing"],
                        "reference_facts": [],
                        "prohibited_claims": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    question_path = tmp_path / "questions.json"
    question_path.write_text(
        json.dumps(
            {
                "version": 1,
                "questions": [
                    {
                        "id": "known",
                        "contract": "free_text_v1",
                        "text": "Describe it.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown questions"):
        load_workload(fixture_path, question_path)


def test_fixture_paths_cannot_escape_manifest_root(tmp_path: Path):
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fixtures": [
                    {
                        "id": "example",
                        "image": "../private.jpg",
                        "licence": "CC0",
                        "provenance": "test",
                        "allowed_questions": ["known"],
                        "reference_facts": [],
                        "prohibited_claims": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    question_path = tmp_path / "questions.json"
    question_path.write_text(
        '{"version":1,"questions":[{"id":"known","contract":"free_text_v1","text":"Describe."}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="safe relative"):
        load_workload(fixture_path, question_path)

