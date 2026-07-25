from __future__ import annotations

from typing import Any

from visionmodelquest.adapters.transformers import TransformersAdapter


class Qwen35Adapter(TransformersAdapter):
    def load(self) -> None:
        super().load()
        image_processor: Any = getattr(self.processor, "image_processor", None)
        budget = self.definition.visual_token_budget
        if image_processor is None or budget is None:
            raise RuntimeError("Qwen3.5 requires an image processor and visual-token budget")
        if getattr(image_processor, "patch_size", None) != 16:
            raise RuntimeError("Qwen3.5 image processor must use patch size 16")
        if getattr(image_processor, "merge_size", None) != 2:
            raise RuntimeError("Qwen3.5 image processor must use merge size 2")
        maximum_pixels = budget * 16**2 * 2**2
        image_processor.size["shortest_edge"] = min(65_536, maximum_pixels)
        image_processor.size["longest_edge"] = maximum_pixels
        self.runtime_details["visual_token_budget"] = budget

