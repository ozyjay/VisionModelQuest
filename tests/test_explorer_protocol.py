from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from visionmodelquest.explorer.protocol import PROTOCOL_VERSION, Request, parse_request


def request_payload(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "request",
        "request_id": "request_1",
        "operation": "initialise_processor",
        "model_key": "mock",
        "payload": {},
    }
    result.update(overrides)
    return result


def test_protocol_accepts_only_versioned_known_operations() -> None:
    request = parse_request(json.dumps(request_payload()))
    assert isinstance(request, Request)
    assert request.operation == "initialise_processor"

    with pytest.raises(ValidationError):
        parse_request(json.dumps(request_payload(operation="execute_command")))
    with pytest.raises(ValidationError):
        parse_request(json.dumps(request_payload(protocol_version=2)))


def test_protocol_rejects_malformed_and_oversized_lines() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_request("{")
    with pytest.raises(ValueError, match="64 KiB"):
        parse_request("x" * (64 * 1024 + 1))
