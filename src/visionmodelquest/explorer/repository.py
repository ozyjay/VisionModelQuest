from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRevision:
    experiment_id: str
    experiment_name: str
    revision_number: int
    model_key: str
    model_id: str
    model_revision: str
    adapter_name: str
    adapter_version: str
    response_contract: str
    system_instruction: str
    user_question: str
    visual_token_budget: int | None
    completion_token_limit: int
    image_reference: dict[str, str]
    created_at: str
    notes: str
    output_hash: str | None
    validation_state: str | None
    timing_summary: dict[str, float | int | None]
    preprocessing_inspection: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "index.json"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.index_path.exists():
            self._atomic_json(self.index_path, {"version": 1, "experiments": []})

    def save(
        self,
        payload: dict[str, Any],
        *,
        experiment_id: str | None = None,
    ) -> ExperimentRevision:
        identifier = experiment_id or uuid.uuid4().hex
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("invalid experiment ID")
        directory = self.root / identifier
        directory.mkdir(mode=0o700, exist_ok=True)
        existing = sorted(directory.glob("revision-*.json"))
        revision_number = len(existing) + 1
        revision = ExperimentRevision(
            experiment_id=identifier,
            revision_number=revision_number,
            created_at=datetime.now(UTC).isoformat(),
            **payload,
        )
        destination = directory / f"revision-{revision_number:04d}.json"
        if destination.exists():
            raise FileExistsError("experiment revision already exists")
        self._atomic_json(destination, revision.as_dict(), exclusive=True)
        self._update_index(revision)
        return revision

    def list_revisions(self) -> list[ExperimentRevision]:
        revisions: list[ExperimentRevision] = []
        for path in sorted(self.root.glob("*/revision-*.json"), reverse=True):
            try:
                revisions.append(
                    ExperimentRevision(**json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return revisions

    def _update_index(self, revision: ExperimentRevision) -> None:
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {"version": 1, "experiments": []}
        entries = [
            item
            for item in index.get("experiments", [])
            if item.get("experiment_id") != revision.experiment_id
        ]
        entries.append(
            {
                "experiment_id": revision.experiment_id,
                "name": revision.experiment_name,
                "latest_revision": revision.revision_number,
                "updated_at": revision.created_at,
            }
        )
        index["experiments"] = sorted(entries, key=lambda item: item["updated_at"], reverse=True)
        self._atomic_json(self.index_path, index)

    @staticmethod
    def _atomic_json(path: Path, payload: object, *, exclusive: bool = False) -> None:
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            if exclusive and path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
