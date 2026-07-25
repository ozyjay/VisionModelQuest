from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExplorerPaths:
    state: Path
    runtime: Path
    experiments: Path
    assets: Path
    logs: Path
    sessions: Path

    @classmethod
    def from_environment(cls) -> ExplorerPaths:
        state_home = Path(
            os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        )
        runtime_home = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/visionmodelquest-{os.getuid()}"))
        state = state_home / "visionmodelquest"
        runtime = runtime_home / "visionmodelquest"
        return cls(
            state=state,
            runtime=runtime,
            experiments=state / "experiments",
            assets=state / "assets" / "sha256",
            logs=state / "logs",
            sessions=runtime / "sessions",
        )

    def prepare(self) -> None:
        for path in (
            self.state,
            self.runtime,
            self.experiments,
            self.assets,
            self.logs,
            self.sessions,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.chmod(0o700)
