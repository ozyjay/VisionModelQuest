from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

from PIL import Image

from visionmodelquest.adapters.base import Generation, ModelAdapter
from visionmodelquest.cache import resolve_snapshot


class TransformersAdapter(ModelAdapter):
    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if self.definition.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available cuda device")
        snapshot = resolve_snapshot(self.definition, self.cache_root)
        dtype = torch.bfloat16 if self.definition.dtype == "bfloat16" else torch.float32
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        if type(self.processor).__name__ != self.definition.expected_processor_class:
            raise RuntimeError(
                f"Expected processor {self.definition.expected_processor_class}, "
                f"received {type(self.processor).__name__}"
            )
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": dtype,
        }
        if self.definition.attention_implementation:
            kwargs["attn_implementation"] = self.definition.attention_implementation
        self.model = AutoModelForMultimodalLM.from_pretrained(snapshot, **kwargs)
        if type(self.model).__name__ != self.definition.expected_model_class:
            raise RuntimeError(
                f"Expected model {self.definition.expected_model_class}, "
                f"received {type(self.model).__name__}"
            )
        self.device = torch.device(self.definition.device)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch
        self.load_seconds = time.perf_counter() - started
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "processor_class": type(self.processor).__name__,
            "model_class": type(self.model).__name__,
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            ),
            "snapshot_path": str(snapshot),
        }

    def _messages(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate(self, image: Path, prompt: str) -> Generation:
        with Image.open(image) as opened:
            raster = opened.convert("RGB")
        preprocessing_started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            self._messages(raster, prompt),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.device)
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(0)
        inference_started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.definition.maximum_completion_tokens,
                do_sample=False,
                use_cache=True,
            )
        inference_seconds = time.perf_counter() - inference_started
        generated = output[0, prompt_tokens:]
        text = self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if any(marker in text.casefold() for marker in ("<think>", "</think>", "<|channel>")):
            raise ValueError("model output exposed a reasoning channel")
        peak = (
            int(self.torch.cuda.max_memory_allocated(0))
            if self.device.type == "cuda"
            else None
        )
        return Generation(
            text=text,
            preprocessing_seconds=preprocessing_seconds,
            inference_seconds=inference_seconds,
            first_output_seconds=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated.shape[-1]),
            finish_reason=(
                "length"
                if int(generated.shape[-1]) >= self.definition.maximum_completion_tokens
                else "stop"
            ),
            peak_gpu_memory_bytes=peak,
        )

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "torch") and self.definition.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

