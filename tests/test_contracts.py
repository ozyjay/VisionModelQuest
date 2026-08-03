import json

import pytest

from visionmodelquest.contracts import ContractError, build_prompt, parse_output


def valid_payload() -> dict[str, object]:
    return {
        "summary": "A person is beside a table.",
        "objects": [
            {
                "label": "person",
                "description": "A person stands beside a table.",
                "approximate_location": "left",
            }
        ],
        "relationships": ["The person is beside the table."],
        "uncertainties": [],
        "safety_notes": [],
    }


def test_strict_structured_output_accepts_one_optional_fence():
    raw = f"```json\n{json.dumps(valid_payload())}\n```"
    result = parse_output("scene_json_v1", raw)
    assert result.summary == "A person is beside a table."


def test_strict_structured_output_accepts_one_unclosed_leading_fence():
    raw = f"```json\n{json.dumps(valid_payload())}"
    result = parse_output("scene_json_v1", raw)
    assert result.summary == "A person is beside a table."


def test_strict_structured_output_rejects_content_after_json():
    raw = f"```json\n{json.dumps(valid_payload())}\nextra text"
    with pytest.raises(ContractError, match="JSON invalid"):
        parse_output("scene_json_v1", raw)


def test_strict_structured_output_rejects_extra_keys():
    payload = valid_payload()
    payload["identity"] = "someone"
    with pytest.raises(ContractError):
        parse_output("scene_json_v1", json.dumps(payload))


def test_object_and_relationship_bounds_are_enforced():
    payload = valid_payload()
    payload["objects"] = payload["objects"] * 4
    with pytest.raises(ContractError, match=r"schema invalid: objects: .*at most 3"):
        parse_output("scene_json_v1", json.dumps(payload))


def test_malformed_json_reports_location_and_reason():
    with pytest.raises(ContractError, match=r"JSON invalid at line 1, column 2"):
        parse_output("scene_json_v1", "{")


def test_description_word_limit_is_enforced():
    payload = valid_payload()
    payload["objects"][0]["description"] = " ".join(["visible"] * 16)
    with pytest.raises(ContractError):
        parse_output("scene_json_v1", json.dumps(payload))


@pytest.mark.parametrize(
    "unsafe",
    [
        "Their name is Example Visitor.",
        "A disabled person is visible.",
        "The person is 42 years old.",
    ],
)
def test_public_safety_rejects_identity_and_sensitive_claims(unsafe: str):
    payload = valid_payload()
    payload["summary"] = unsafe
    with pytest.raises(ContractError, match="prohibited"):
        parse_output("scene_json_v1", json.dumps(payload))


def test_free_text_is_bounded_and_safe():
    assert parse_output("free_text_v1", "A generic person is beside a table.") == (
        "A generic person is beside a table."
    )
    with pytest.raises(ContractError):
        parse_output("free_text_v1", "x" * 1201)


def test_prompts_are_deterministic_and_do_not_enable_identity():
    first = build_prompt("scene_json_v1", "Describe the scene.")
    assert first == build_prompt("scene_json_v1", "Describe the scene.")
    assert "Do not identify anyone" in first
    assert '"objects": [' in first
    assert '"approximate_location": "centre"' in first
    assert "never an array of strings" in first
