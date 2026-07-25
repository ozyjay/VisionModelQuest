from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SceneObject(StrictOutput):
    label: str = Field(min_length=1, max_length=48)
    description: str = Field(min_length=1, max_length=150)
    approximate_location: Literal["left", "centre", "right", "foreground", "background"]

    @field_validator("description")
    @classmethod
    def concise_description(cls, value: str) -> str:
        if len(re.findall(r"\b[\w'-]+\b", value)) > 15:
            raise ValueError("object descriptions may contain no more than 15 words")
        return value


class SceneDescription(StrictOutput):
    summary: str = Field(min_length=1, max_length=360)
    objects: list[SceneObject] = Field(max_length=3)
    relationships: list[str] = Field(max_length=1)
    uncertainties: list[str] = Field(max_length=1)
    safety_notes: list[str] = Field(max_length=1)

    @field_validator("relationships", "uncertainties", "safety_notes")
    @classmethod
    def bounded_strings(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 180 for value in values):
            raise ValueError("list entries must contain 1 to 180 characters")
        return values


class HumanReview(StrictOutput):
    fixture_id: str
    question_id: str
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    visible_factual_correctness: int = Field(ge=0, le=4)
    object_coverage: int = Field(ge=0, le=4)
    hallucinated_objects: int = Field(ge=0, le=4)
    counting_accuracy: int = Field(ge=0, le=4)
    spatial_relationship_accuracy: int = Field(ge=0, le=4)
    uncertainty_calibration: int = Field(ge=0, le=4)
    text_reading_accuracy: int = Field(ge=0, le=4)
    concise_answer_quality: int = Field(ge=0, le=4)
    json_contract_compliance: int = Field(ge=0, le=4)
    public_safety_compliance: int = Field(ge=0, le=4)
    notes: str = Field(default="", max_length=1000)

