from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from visionmodelquest.explorer.geometry import ImageInspection, build_token_regions


def qwen_inspection(
    *,
    image_path: Path,
    processed_root: Path,
    image_processor: Any,
    visual_token_budget: int,
) -> ImageInspection:
    with Image.open(image_path) as opened:
        media_type = Image.MIME.get(opened.format or "", "application/octet-stream")
        original_width, original_height = opened.size
        raster = opened.convert("RGB")
    result = image_processor(images=[raster], return_tensors="pt")
    grid_values = result["image_grid_thw"][0].tolist()
    temporal, raw_rows, raw_columns = (int(value) for value in grid_values)
    patch_size = int(image_processor.patch_size)
    merge_size = int(image_processor.merge_size)
    processed_width = raw_columns * patch_size
    processed_height = raw_rows * patch_size
    processed_id = uuid.uuid4().hex
    processed_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    processed_path = processed_root / f"{processed_id}.png"
    resample = getattr(image_processor, "resample", Image.Resampling.BICUBIC)
    raster.resize((processed_width, processed_height), resample=resample).save(
        processed_path, "PNG"
    )
    processed_path.chmod(0o600)
    tokens = build_token_regions(
        source_width=original_width,
        source_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
        raw_rows=raw_rows,
        raw_columns=raw_columns,
        merge_size=merge_size,
    )
    mean = tuple(float(value) for value in getattr(image_processor, "image_mean", ()))
    standard_deviation = tuple(
        float(value) for value in getattr(image_processor, "image_std", ())
    )
    return ImageInspection(
        original_width=original_width,
        original_height=original_height,
        media_type=media_type,
        encoded_size=image_path.stat().st_size,
        processed_width=processed_width,
        processed_height=processed_height,
        resize_method=str(resample),
        colour_conversion="RGB",
        rescale_factor=float(getattr(image_processor, "rescale_factor", 1 / 255)),
        channel_mean=mean,
        channel_standard_deviation=standard_deviation,
        image_grid_thw=(temporal, raw_rows, raw_columns),
        patch_size=patch_size,
        merge_size=merge_size,
        requested_visual_token_budget=visual_token_budget,
        raw_grid_rows=raw_rows,
        raw_grid_columns=raw_columns,
        merged_grid_rows=raw_rows // merge_size,
        merged_grid_columns=raw_columns // merge_size,
        actual_visual_tokens=len(tokens) * temporal,
        processed_image_id=processed_id,
        tokens=tokens,
    )


def mock_inspection(
    *,
    image_path: Path,
    processed_root: Path,
    visual_token_budget: int = 140,
) -> ImageInspection:
    class MockQwenProcessor:
        patch_size = 16
        merge_size = 2
        resample = Image.Resampling.BICUBIC
        rescale_factor = 1 / 255
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)

        def __call__(self, *, images: list[Image.Image], return_tensors: str) -> dict[str, Any]:
            del images, return_tensors

            class Grid:
                @staticmethod
                def tolist() -> list[int]:
                    return [1, 14, 20]

            return {"image_grid_thw": [Grid()]}

    return qwen_inspection(
        image_path=image_path,
        processed_root=processed_root,
        image_processor=MockQwenProcessor(),
        visual_token_budget=visual_token_budget,
    )
