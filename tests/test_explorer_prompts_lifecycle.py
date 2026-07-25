from __future__ import annotations

import pytest

from visionmodelquest.contracts import SYSTEM_SAFETY
from visionmodelquest.explorer.lifecycle import Lifecycle, WorkerState
from visionmodelquest.explorer.prompts import (
    canonical_diff,
    compile_prompt,
    is_canonical,
)


def test_prompt_preserves_roles_and_appends_contract_deterministically() -> None:
    first = compile_prompt(SYSTEM_SAFETY, "What is visible?", "free_text_v1")
    second = compile_prompt(SYSTEM_SAFETY, "What is visible?", "free_text_v1")

    assert first == second
    assert first.messages()[0]["role"] == "system"
    assert first.messages()[1]["role"] == "user"
    assert first.user_content.startswith("What is visible?")
    assert is_canonical(SYSTEM_SAFETY)
    assert not is_canonical(SYSTEM_SAFETY + " Extra")
    assert "+Extra" in canonical_diff(SYSTEM_SAFETY + "\nExtra")


def test_lifecycle_rejects_invalid_transitions() -> None:
    lifecycle = Lifecycle()
    lifecycle.transition(WorkerState.STARTING)
    lifecycle.transition(WorkerState.PROCESSOR_READY)
    lifecycle.transition(WorkerState.LOADING_MODEL)
    lifecycle.transition(WorkerState.MODEL_READY)
    lifecycle.transition(WorkerState.GENERATING)
    lifecycle.transition(WorkerState.CANCELLING)
    lifecycle.transition(WorkerState.RESTARTING)
    lifecycle.transition(WorkerState.STARTING)

    with pytest.raises(ValueError, match="invalid worker transition"):
        lifecycle.transition(WorkerState.GENERATING)
