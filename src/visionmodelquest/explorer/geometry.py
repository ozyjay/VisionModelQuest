from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Bounds:
    x: float
    y: float
    width: float
    height: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TokenRegion:
    sequence_index: int
    row: int
    column: int
    processed_bounds: Bounds
    source_bounds: Bounds
    raw_patches: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_patches"] = [list(item) for item in self.raw_patches]
        return payload


@dataclass(frozen=True)
class ImageInspection:
    original_width: int
    original_height: int
    media_type: str
    encoded_size: int
    processed_width: int
    processed_height: int
    resize_method: str
    colour_conversion: str
    rescale_factor: float
    channel_mean: tuple[float, ...]
    channel_standard_deviation: tuple[float, ...]
    image_grid_thw: tuple[int, int, int]
    patch_size: int
    merge_size: int
    requested_visual_token_budget: int
    raw_grid_rows: int
    raw_grid_columns: int
    merged_grid_rows: int
    merged_grid_columns: int
    actual_visual_tokens: int
    processed_image_id: str
    tokens: tuple[TokenRegion, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tokens"] = [token.as_dict() for token in self.tokens]
        return payload


@dataclass(frozen=True)
class ViewTransform:
    scale: float
    offset_x: float
    offset_y: float

    def processed_to_widget(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def widget_to_processed(self, x: float, y: float) -> tuple[float, float]:
        if self.scale <= 0:
            raise ValueError("view scale must be positive")
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale


def fit_transform(
    widget_width: float,
    widget_height: float,
    image_width: float,
    image_height: float,
    *,
    padding: float = 24,
) -> ViewTransform:
    if min(widget_width, widget_height, image_width, image_height) <= 0:
        return ViewTransform(1.0, 0.0, 0.0)
    available_width = max(1.0, widget_width - 2 * padding)
    available_height = max(1.0, widget_height - 2 * padding)
    scale = min(available_width / image_width, available_height / image_height)
    return ViewTransform(
        scale=scale,
        offset_x=(widget_width - image_width * scale) / 2,
        offset_y=(widget_height - image_height * scale) / 2,
    )


def build_token_regions(
    *,
    source_width: int,
    source_height: int,
    processed_width: int,
    processed_height: int,
    raw_rows: int,
    raw_columns: int,
    merge_size: int,
) -> tuple[TokenRegion, ...]:
    values = (
        source_width,
        source_height,
        processed_width,
        processed_height,
        raw_rows,
        raw_columns,
        merge_size,
    )
    if any(value <= 0 for value in values):
        raise ValueError("token geometry dimensions must be positive")
    if raw_rows % merge_size or raw_columns % merge_size:
        raise ValueError("raw patch grid must divide exactly by merge size")
    merged_rows = raw_rows // merge_size
    merged_columns = raw_columns // merge_size
    cell_width = processed_width / merged_columns
    cell_height = processed_height / merged_rows
    scale_x = source_width / processed_width
    scale_y = source_height / processed_height
    regions: list[TokenRegion] = []
    for row in range(merged_rows):
        for column in range(merged_columns):
            bounds = Bounds(
                x=column * cell_width,
                y=row * cell_height,
                width=cell_width,
                height=cell_height,
            )
            source = Bounds(
                x=bounds.x * scale_x,
                y=bounds.y * scale_y,
                width=bounds.width * scale_x,
                height=bounds.height * scale_y,
            )
            patches = tuple(
                (raw_row, raw_column)
                for raw_row in range(row * merge_size, (row + 1) * merge_size)
                for raw_column in range(column * merge_size, (column + 1) * merge_size)
            )
            regions.append(
                TokenRegion(
                    sequence_index=row * merged_columns + column,
                    row=row,
                    column=column,
                    processed_bounds=bounds,
                    source_bounds=source,
                    raw_patches=patches,
                )
            )
    return tuple(regions)


def token_at_point(
    regions: tuple[TokenRegion, ...],
    x: float,
    y: float,
) -> TokenRegion | None:
    for region in regions:
        bounds = region.processed_bounds
        if (
            bounds.x <= x < bounds.x + bounds.width
            and bounds.y <= y < bounds.y + bounds.height
        ):
            return region
    return None


def clamp_zoom(value: float) -> float:
    return min(12.0, max(0.1, value)) if math.isfinite(value) else 1.0
