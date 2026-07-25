from __future__ import annotations

from enum import StrEnum


class WorkerState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting_worker"
    PROCESSOR_READY = "processor_ready"
    LOADING_MODEL = "loading_model"
    MODEL_READY = "model_ready"
    GENERATING = "generating"
    UNLOADING = "unloading"
    CANCELLING = "cancelling"
    RESTARTING = "restarting_worker"
    FAILED = "worker_failed"


ALLOWED_TRANSITIONS: dict[WorkerState, frozenset[WorkerState]] = {
    WorkerState.STOPPED: frozenset({WorkerState.STARTING}),
    WorkerState.STARTING: frozenset({WorkerState.PROCESSOR_READY, WorkerState.FAILED}),
    WorkerState.PROCESSOR_READY: frozenset(
        {WorkerState.LOADING_MODEL, WorkerState.STOPPED, WorkerState.FAILED}
    ),
    WorkerState.LOADING_MODEL: frozenset({WorkerState.MODEL_READY, WorkerState.FAILED}),
    WorkerState.MODEL_READY: frozenset(
        {
            WorkerState.GENERATING,
            WorkerState.UNLOADING,
            WorkerState.STOPPED,
            WorkerState.FAILED,
        }
    ),
    WorkerState.GENERATING: frozenset(
        {WorkerState.MODEL_READY, WorkerState.CANCELLING, WorkerState.FAILED}
    ),
    WorkerState.UNLOADING: frozenset({WorkerState.PROCESSOR_READY, WorkerState.FAILED}),
    WorkerState.CANCELLING: frozenset({WorkerState.RESTARTING, WorkerState.FAILED}),
    WorkerState.RESTARTING: frozenset({WorkerState.STARTING, WorkerState.FAILED}),
    WorkerState.FAILED: frozenset({WorkerState.STARTING, WorkerState.STOPPED}),
}


class Lifecycle:
    def __init__(self) -> None:
        self.state = WorkerState.STOPPED

    def transition(self, target: WorkerState, *, force: bool = False) -> WorkerState:
        if not force and target not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid worker transition: {self.state} → {target}")
        self.state = target
        return target


def automatic_restart_delay(crash_count: int, *, maximum_crashes: int = 3) -> int | None:
    """Return bounded exponential backoff, or None when automatic recovery must stop."""
    if crash_count < 1 or maximum_crashes < 1:
        raise ValueError("crash counts must be positive")
    if crash_count >= maximum_crashes:
        return None
    return 2 ** (crash_count - 1)
