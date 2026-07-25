from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gtk


@pytest.mark.skipif(
    not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")),
    reason="requires an active graphical session",
)
def test_inspector_supports_keyboard_selection() -> None:
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    assert Gtk.init_check()
    from visionmodelquest.explorer.geometry import build_token_regions
    from visionmodelquest.ui.inspector import TokenInspector

    inspector = TokenInspector()
    inspector.regions = build_token_regions(
        source_width=8,
        source_height=6,
        processed_width=320,
        processed_height=224,
        raw_rows=14,
        raw_columns=20,
        merge_size=2,
    )
    inspector.merged_columns = 10
    inspector.select(0)
    inspector.select(inspector.selected_index + inspector.merged_columns)
    assert inspector.selected_index == 10


def test_worker_controller_preserves_virtual_environment_launcher(
    tmp_path: Path,
) -> None:
    from visionmodelquest.ui.controller import WorkerController

    environment = tmp_path / "inference-env"
    (environment / "bin").mkdir(parents=True)
    launcher = environment / "bin" / "python"
    launcher.symlink_to(Path(sys.executable).resolve())
    session = tmp_path / "session"
    (session / "images").mkdir(parents=True)
    (session / "processed").mkdir()
    controller = WorkerController(
        python=launcher,
        model_key="mock",
        session_root=session,
        runtime_root=tmp_path / "runtime",
        log_root=tmp_path / "logs",
    )

    assert controller.python == launcher.absolute()
    assert controller.python != launcher.resolve()
