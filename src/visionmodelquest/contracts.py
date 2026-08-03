from __future__ import annotations

import json
import re

from pydantic import ValidationError

from visionmodelquest.schemas import SceneDescription

SYSTEM_SAFETY = (
    "Describe only visible evidence. Use generic 'person' language. Do not identify anyone "
    "or infer age, emotion, ethnicity, religion, health, disability, sexuality, criminality, "
    "politics or other sensitive traits. State uncertainty instead of inventing certainty."
)

SCENE_JSON_INSTRUCTION = """Return only one complete JSON object with exactly this structure:
{
  "summary": "A concise description of the visible scene.",
  "objects": [
    {
      "label": "object label",
      "description": "Visible evidence in no more than 15 words.",
      "approximate_location": "centre"
    }
  ],
  "relationships": ["One concise visible relationship."],
  "uncertainties": ["One concise uncertainty."],
  "safety_notes": []
}
The objects value must be an array of objects, never an array of strings. Every object must
contain exactly label, description and approximate_location. approximate_location must be
exactly one of: left, centre, right, foreground, background. relationships, uncertainties
and safety_notes must always be arrays of strings, including when empty. Use at most three
objects, one relationship, one uncertainty and one safety note. Prefer omitting low-value
detail over violating the structure. Finish the complete JSON object before adding detail."""

PROHIBITED_PATTERNS = (
    r"\b(?:facial|face) recognition\b",
    r"\b(?:his|her|their) name is\b",
    r"\bnamed [A-Z][A-Za-z'-]+\b",
    r"\b\d{1,3}[ -]years?[ -]old\b",
    r"\b(?:ethnicity|religion|sexuality|criminality|political views?)\b",
    r"\b(?:disabled|autistic|depressed|anxious) person\b",
)


class ContractError(ValueError):
    """Model output does not satisfy the selected contract."""


def build_prompt(contract: str, question: str) -> str:
    if contract == "scene_json_v1":
        return f"{SYSTEM_SAFETY}\n\n{SCENE_JSON_INSTRUCTION}\n\nQuestion: {question}"
    if contract == "free_text_v1":
        return (
            f"{SYSTEM_SAFETY}\n\nAnswer in plain text using no more than three concise "
            f"sentences.\n\nQuestion: {question}"
        )
    raise ContractError(f"unknown contract: {contract}")


def validate_public_safety(text: str) -> None:
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROHIBITED_PATTERNS):
        raise ContractError("output contains a prohibited identity or sensitive-trait claim")


def _remove_single_fence(raw: str) -> str:
    cleaned = raw.strip()
    opening = re.match(r"```(?:json)?[ \t]*(?:\r?\n|$)", cleaned)
    if opening is None:
        return cleaned
    content = cleaned[opening.end() :].rstrip()
    if content.endswith("```"):
        content = content[:-3].rstrip()
    return content


def _validation_reason(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "output"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def parse_output(contract: str, raw: str) -> SceneDescription | str:
    if not raw.strip():
        raise ContractError("model output is empty")
    validate_public_safety(raw)
    if contract == "free_text_v1":
        if len(raw) > 1200:
            raise ContractError("free-text output exceeds 1,200 characters")
        return raw.strip()
    if contract != "scene_json_v1":
        raise ContractError(f"unknown contract: {contract}")
    try:
        payload = json.loads(_remove_single_fence(raw))
    except json.JSONDecodeError as error:
        raise ContractError(
            f"scene_json_v1 JSON invalid at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ContractError("structured output must be one JSON object")
    try:
        return SceneDescription.model_validate(payload)
    except ValidationError as error:
        raise ContractError(f"scene_json_v1 schema invalid: {_validation_reason(error)}") from error
