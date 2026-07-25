from __future__ import annotations

import hashlib
import time
from pathlib import Path

from PIL import Image

from visionmodelquest.adapters.base import Generation, ModelAdapter


class MockAdapter(ModelAdapter):
    def load(self) -> None:
        started = time.perf_counter()
        self.runtime_details = {"engine": "deterministic-mock", "device": "cpu"}
        self.load_seconds = time.perf_counter() - started

    def generate(self, image: Path, prompt: str) -> Generation:
        started = time.perf_counter()
        with Image.open(image) as opened:
            opened.verify()
        digest = hashlib.sha256(f"{image.name}\0{prompt}".encode()).hexdigest()[:12]
        preprocessing = time.perf_counter() - started
        inference_started = time.perf_counter()
        if "Return only one JSON object" in prompt:
            text = (
                '{"summary":"A prepared benchmark fixture is visible.",'
                '"objects":[{"label":"fixture","description":"A local benchmark image is visible.",'
                '"approximate_location":"centre"}],"relationships":[],"uncertainties":'
                f'["Mock output marker {digest}."],"safety_notes":[]}}'
            )
        else:
            text = f"A prepared local benchmark fixture is visible. Mock marker: {digest}."
        inference = time.perf_counter() - inference_started
        return Generation(
            text=text,
            preprocessing_seconds=preprocessing,
            inference_seconds=inference,
            first_output_seconds=inference,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
        )

