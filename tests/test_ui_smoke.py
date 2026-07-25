from __future__ import annotations

import os

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
