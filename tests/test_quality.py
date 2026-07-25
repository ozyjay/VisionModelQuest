import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from visionmodelquest.benchmarks.quality import load_reviews, review_template


def test_review_template_uses_bounded_scale():
    result = review_template("fixture", "question", "a" * 64)
    assert result["visible_factual_correctness"] == 0
    assert result["public_safety_compliance"] == 0


def test_review_validation_rejects_out_of_range_score(tmp_path: Path):
    review = review_template("fixture", "question", "a" * 64)
    review["visible_factual_correctness"] = 5
    path = tmp_path / "review.json"
    path.write_text(json.dumps([review]), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_reviews(path)

