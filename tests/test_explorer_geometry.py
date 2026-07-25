from __future__ import annotations

import pytest

from visionmodelquest.explorer.geometry import (
    ViewTransform,
    build_token_regions,
    fit_transform,
    token_at_point,
)


def test_acceptance_fixture_geometry_has_70_merged_tokens() -> None:
    regions = build_token_regions(
        source_width=8,
        source_height=6,
        processed_width=320,
        processed_height=224,
        raw_rows=14,
        raw_columns=20,
        merge_size=2,
    )

    assert len(regions) == 70
    assert regions[0].processed_bounds.width == 32
    assert regions[0].processed_bounds.height == 32
    assert regions[-1].row == 6
    assert regions[-1].column == 9
    assert regions[-1].raw_patches == ((12, 18), (12, 19), (13, 18), (13, 19))


def test_source_mapping_and_hit_testing() -> None:
    regions = build_token_regions(
        source_width=8,
        source_height=6,
        processed_width=320,
        processed_height=224,
        raw_rows=14,
        raw_columns=20,
        merge_size=2,
    )

    assert regions[0].source_bounds.width == pytest.approx(0.8)
    assert regions[0].source_bounds.height == pytest.approx(6 / 7)
    assert token_at_point(regions, 33, 33) == regions[11]
    assert token_at_point(regions, 320, 224) is None


def test_view_transform_round_trip_and_fit() -> None:
    transform = fit_transform(640, 480, 320, 224)
    point = transform.processed_to_widget(83.5, 41.25)
    assert transform.widget_to_processed(*point) == pytest.approx((83.5, 41.25))
    assert transform.scale > 0

    with pytest.raises(ValueError, match="positive"):
        ViewTransform(0, 0, 0).widget_to_processed(1, 1)
