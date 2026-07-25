from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = 1
Operation = Literal[
    "initialise_processor",
    "inspect_image",
    "load_model",
    "generate",
    "unload_model",
    "shutdown",
]


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal[1]
    kind: Literal["request", "response", "event"]
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")


class Request(Envelope):
    kind: Literal["request"] = "request"
    operation: Operation
    model_key: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    payload: dict[str, Any] = Field(default_factory=dict)


class Response(Envelope):
    kind: Literal["response"] = "response"
    operation: Operation
    status: Literal["ok", "error"]
    worker_state: str
    started_at: str
    completed_at: str
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class Event(Envelope):
    kind: Literal["event"] = "event"
    event: Literal["state_changed", "output_fragment", "model_auto_unloaded"]
    worker_state: str
    timestamp: str
    payload: dict[str, Any] = Field(default_factory=dict)


def parse_request(line: str, *, maximum_bytes: int = 64 * 1024) -> Request:
    if len(line.encode("utf-8")) > maximum_bytes:
        raise ValueError("protocol request exceeds 64 KiB")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("request is not valid JSON") from error
    return Request.model_validate(payload)


def serialise(envelope: BaseModel) -> str:
    return envelope.model_dump_json(exclude_none=True)
