from __future__ import annotations

import json
from pathlib import Path

from visionmodelquest.schemas import HumanReview

REVIEW_DIMENSIONS = (
    "visible_factual_correctness",
    "object_coverage",
    "hallucinated_objects",
    "counting_accuracy",
    "spatial_relationship_accuracy",
    "uncertainty_calibration",
    "text_reading_accuracy",
    "concise_answer_quality",
    "json_contract_compliance",
    "public_safety_compliance",
)


def review_template(
    fixture_id: str, question_id: str, output_hash: str
) -> dict[str, object]:
    result: dict[str, object] = {
        "fixture_id": fixture_id,
        "question_id": question_id,
        "output_hash": output_hash,
    }
    result.update({dimension: 0 for dimension in REVIEW_DIMENSIONS})
    result["notes"] = ""
    return result


def load_reviews(path: Path) -> list[HumanReview]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("review file must contain a list")
    return [HumanReview.model_validate(item) for item in payload]

