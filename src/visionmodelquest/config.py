from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = ROOT / "config" / "models.json"
FIXTURE_MANIFEST_PATH = ROOT / "fixtures" / "manifests" / "v2.json"
QUESTION_PATH = ROOT / "fixtures" / "questions" / "v1.json"

AdapterName = Literal["mock", "qwen35", "smolvlm2", "gemma3"]
ContractName = Literal["scene_json_v1", "free_text_v1"]
FailureCategory = Literal[
    "cache_missing",
    "configuration_error",
    "dependency_missing",
    "generation_error",
    "hardware_unavailable",
    "image_invalid",
    "incompatible",
    "interrupted",
    "output_invalid",
    "thermal_limit",
    "timeout",
    "unsupported",
]

DEFAULT_MAX_TEMPERATURE_CELSIUS = 95.0


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelDefinition(StrictModel):
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str
    model_id: str
    revision: str
    adapter: AdapterName
    contracts: tuple[ContractName, ...]
    expected_processor_class: str
    expected_model_class: str
    dtype: Literal["bfloat16", "float32"]
    device: Literal["cuda:0", "cpu"]
    local_files_only: Literal[True]
    trust_remote_code: Literal[False]
    attention_implementation: Literal["sdpa"] | None = None
    visual_token_budget: int | None = Field(default=None, ge=1)
    maximum_completion_tokens: int = Field(ge=1, le=4096)
    startup_timeout_seconds: float = Field(gt=0, le=3600)
    generation_timeout_seconds: float = Field(gt=0, le=3600)
    hardware_verification_required: bool
    adapter_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    quantisation: str | None = None

    @model_validator(mode="after")
    def immutable_revision(self) -> ModelDefinition:
        if self.adapter != "mock" and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("non-mock models require a full 40-character commit revision")
        if self.adapter != "mock" and "/" not in self.model_id:
            raise ValueError("model_id must be a Hugging Face repository ID")
        return self


class QuestionDefinition(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    contract: ContractName
    text: str = Field(min_length=1, max_length=500)


class FixtureDefinition(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    image: str
    licence: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    allowed_questions: tuple[str, ...] = Field(min_length=1)
    reference_facts: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    expected_object_labels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def safe_relative_image(self) -> FixtureDefinition:
        path = Path(self.image)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("fixture image must be a safe relative path")
        return self


class FixtureManifest(StrictModel):
    version: int = Field(ge=1)
    fixtures: tuple[FixtureDefinition, ...] = Field(min_length=1)


class QuestionManifest(StrictModel):
    version: int = Field(ge=1)
    questions: tuple[QuestionDefinition, ...] = Field(min_length=1)


class Preset(StrictModel):
    name: Literal["quick", "standard", "stability"]
    warmups: int = Field(ge=0)
    measured_requests: int = Field(ge=1)
    duration_seconds: float | None = Field(default=None, gt=0)
    sample_interval_seconds: float = Field(default=1.0, gt=0)


PRESETS = {
    "quick": Preset(name="quick", warmups=1, measured_requests=2),
    "standard": Preset(name="standard", warmups=2, measured_requests=10),
    "stability": Preset(
        name="stability",
        warmups=2,
        measured_requests=1,
        duration_seconds=900,
        sample_interval_seconds=1,
    ),
}


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}") from error


def load_models(path: Path = MODEL_CONFIG_PATH) -> dict[str, ModelDefinition]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("model configuration must contain a models list")
    definitions = [ModelDefinition.model_validate(item) for item in payload["models"]]
    result = {item.key: item for item in definitions}
    if len(result) != len(definitions):
        raise ValueError("model keys must be unique")
    return result


def load_workload(
    fixture_path: Path = FIXTURE_MANIFEST_PATH,
    question_path: Path = QUESTION_PATH,
) -> tuple[FixtureManifest, QuestionManifest]:
    fixtures = FixtureManifest.model_validate(_load_json(fixture_path))
    questions = QuestionManifest.model_validate(_load_json(question_path))
    question_ids = {item.id for item in questions.questions}
    if len(question_ids) != len(questions.questions):
        raise ValueError("question IDs must be unique")
    fixture_ids = {item.id for item in fixtures.fixtures}
    if len(fixture_ids) != len(fixtures.fixtures):
        raise ValueError("fixture IDs must be unique")
    for fixture in fixtures.fixtures:
        unknown = set(fixture.allowed_questions) - question_ids
        if unknown:
            raise ValueError(f"fixture {fixture.id} has unknown questions: {sorted(unknown)}")
    return fixtures, questions
