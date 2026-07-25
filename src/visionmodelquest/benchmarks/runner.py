from __future__ import annotations

import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil
from PIL import Image, UnidentifiedImageError

from visionmodelquest.adapters.base import ModelAdapter
from visionmodelquest.config import (
    FixtureManifest,
    ModelDefinition,
    Preset,
    QuestionManifest,
)
from visionmodelquest.contracts import ContractError, build_prompt, parse_output
from visionmodelquest.hardware import environment_fingerprint, memory_snapshot, temperature_readings


def deterministic_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_image(path: Path) -> dict[str, int | str]:
    try:
        size = path.stat().st_size
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or "unknown"
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"invalid fixture image: {path}") from error
    if size <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid fixture image dimensions or size: {path}")
    return {
        "width": width,
        "height": height,
        "encoded_bytes": size,
        "format": image_format,
    }


def _sample(
    *,
    adapter: ModelAdapter,
    image_path: Path,
    fixture_id: str,
    question_id: str,
    contract: str,
    question: str,
    warmup: bool,
    quality_capture: bool,
) -> dict[str, Any]:
    image_details = validate_image(image_path)
    prompt = build_prompt(contract, question)
    host_before = memory_snapshot()
    temperatures_before = temperature_readings()
    started = time.perf_counter()
    try:
        generation = adapter.generate(image_path, prompt)
        validation_started = time.perf_counter()
        parsed = parse_output(contract, generation.text)
        validation_seconds = time.perf_counter() - validation_started
        total_seconds = time.perf_counter() - started
        completion_tokens = generation.completion_tokens
        throughput = (
            completion_tokens / generation.inference_seconds
            if completion_tokens is not None and generation.inference_seconds > 0
            else None
        )
        result: dict[str, Any] = {
            "fixture_id": fixture_id,
            "question_id": question_id,
            "contract": contract,
            "warmup": warmup,
            "status": "passed",
            "failure_category": None,
            "image": image_details,
            "preprocessing_seconds": generation.preprocessing_seconds,
            "time_to_first_output_seconds": generation.first_output_seconds,
            "inference_seconds": generation.inference_seconds,
            "validation_seconds": validation_seconds,
            "total_seconds": total_seconds,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": throughput,
            "finish_reason": generation.finish_reason,
            "visual_tokens": generation.visual_tokens,
            "structured_output_valid": contract != "scene_json_v1" or parsed is not None,
            "output_hash": deterministic_hash(generation.text),
            "peak_gpu_memory_bytes": generation.peak_gpu_memory_bytes,
            "memory_before": host_before,
            "memory_after": memory_snapshot(),
            "hottest_temperature_before": _hottest(temperatures_before),
            "hottest_temperature_after": _hottest(temperature_readings()),
        }
        if quality_capture:
            result["quality_capture"] = {
                "prompt": prompt,
                "output": generation.text,
            }
        return result
    except ContractError as error:
        return _failure_sample(
            fixture_id, question_id, contract, warmup, "output_invalid", str(error), started
        )
    except Exception as error:
        return _failure_sample(
            fixture_id, question_id, contract, warmup, "generation_error", str(error), started
        )


def _failure_sample(
    fixture_id: str,
    question_id: str,
    contract: str,
    warmup: bool,
    category: str,
    reason: str,
    started: float,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "question_id": question_id,
        "contract": contract,
        "warmup": warmup,
        "status": "failed",
        "failure_category": category,
        "failure_reason": reason[:500],
        "total_seconds": time.perf_counter() - started,
    }


def _hottest(readings: list[dict[str, object]]) -> dict[str, object] | None:
    return max(readings, key=lambda item: float(item["celsius"])) if readings else None


class ResourceSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, object]] = []
        self.finished = threading.Event()
        self.started = False
        self.thread = threading.Thread(
            target=self._monitor,
            name="visionmodelquest-resource-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._capture()
        self.started = True
        self.thread.start()

    def close(self) -> None:
        self.finished.set()
        if self.started:
            self.thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._capture()

    def _capture(self) -> None:
        self.samples.append(
            {
                "monotonic_seconds": time.monotonic(),
                "memory": memory_snapshot(),
                "hottest_temperature": _hottest(temperature_readings()),
            }
        )

    def _monitor(self) -> None:
        while not self.finished.wait(self.interval_seconds):
            self._capture()

    def summary(self) -> dict[str, object]:
        memory_keys = ("process_rss_bytes", "gtt_used_bytes", "vram_used_bytes")
        peaks: dict[str, int | None] = {}
        for key in memory_keys:
            values = [
                int(sample["memory"][key])
                for sample in self.samples
                if sample["memory"].get(key) is not None
            ]
            peaks[f"peak_{key}"] = max(values) if values else None
        temperatures = [
            sample["hottest_temperature"]
            for sample in self.samples
            if sample["hottest_temperature"] is not None
        ]
        return {
            "sample_count": len(self.samples),
            **peaks,
            "hottest_temperature": (
                max(temperatures, key=lambda item: float(item["celsius"]))
                if temperatures
                else None
            ),
        }


def run_adapter(
    definition: ModelDefinition,
    adapter: ModelAdapter,
    fixtures: FixtureManifest,
    questions: QuestionManifest,
    preset: Preset,
    *,
    root: Path,
    selected_fixture_ids: set[str] | None = None,
    quality_capture: bool = False,
    stability_duration_seconds: float | None = None,
) -> dict[str, Any]:
    process = psutil.Process()
    memory_before_load = memory_snapshot()
    result: dict[str, Any] = {
        "model_key": definition.key,
        "display_name": definition.display_name,
        "model_id": definition.model_id,
        "revision": definition.revision,
        "adapter": definition.adapter,
        "adapter_version": definition.adapter_version,
        "dtype": definition.dtype,
        "quantisation": definition.quantisation,
        "device": definition.device,
        "status": "failed",
        "failure_category": None,
        "samples": [],
        "memory_before_load": memory_before_load,
        "process_id": process.pid,
    }
    sampler = ResourceSampler(preset.sample_interval_seconds)
    try:
        adapter.load()
        result["model_load_seconds"] = adapter.load_seconds
        result["runtime_details"] = adapter.runtime_details
        result["memory_after_load"] = memory_snapshot()
        sampler.start()
        questions_by_id = {item.id: item for item in questions.questions}
        tasks = [
            (fixture, questions_by_id[question_id])
            for fixture in fixtures.fixtures
            if selected_fixture_ids is None or fixture.id in selected_fixture_ids
            for question_id in fixture.allowed_questions
            if questions_by_id[question_id].contract in definition.contracts
        ]
        if not tasks:
            raise ValueError("selection produced no compatible fixture/question tasks")
        for index in range(preset.warmups):
            fixture, question = tasks[index % len(tasks)]
            result["samples"].append(
                _sample(
                    adapter=adapter,
                    image_path=root / "fixtures" / fixture.image,
                    fixture_id=fixture.id,
                    question_id=question.id,
                    contract=question.contract,
                    question=question.text,
                    warmup=True,
                    quality_capture=quality_capture,
                )
            )
        result["warm_up_seconds"] = sum(
            float(sample["total_seconds"])
            for sample in result["samples"]
            if sample["warmup"]
        )
        stability_duration = stability_duration_seconds or preset.duration_seconds
        deadline = time.monotonic() + stability_duration if stability_duration else None
        iteration = 0
        while True:
            fixture, question = tasks[iteration % len(tasks)]
            result["samples"].append(
                _sample(
                    adapter=adapter,
                    image_path=root / "fixtures" / fixture.image,
                    fixture_id=fixture.id,
                    question_id=question.id,
                    contract=question.contract,
                    question=question.text,
                    warmup=False,
                    quality_capture=quality_capture,
                )
            )
            iteration += 1
            if deadline is not None:
                if time.monotonic() >= deadline:
                    break
            elif iteration >= preset.measured_requests * len(tasks):
                break
        measured = [sample for sample in result["samples"] if not sample["warmup"]]
        result["status"] = (
            "passed" if measured and all(item["status"] == "passed" for item in measured) else "failed"
        )
        if result["status"] == "failed":
            result["failure_category"] = "generation_error"
    except Exception as error:
        result["failure_category"] = _categorise_error(error)
        result["failure_reason"] = str(error)[:500]
    finally:
        sampler.close()
        result["resource_sampling"] = sampler.summary()
        unload_started = time.perf_counter()
        try:
            adapter.close()
        finally:
            result["model_unload_seconds"] = time.perf_counter() - unload_started
            result["memory_after_unload"] = memory_snapshot()
            result["memory_recovered_bytes"] = (
                int(result["memory_after_load"]["process_rss_bytes"])
                - int(result["memory_after_unload"]["process_rss_bytes"])
                if result.get("memory_after_load")
                else None
            )
    return result


def _categorise_error(error: Exception) -> str:
    text = str(error).casefold()
    if "not cached" in text or "snapshot" in text and "missing" in text:
        return "cache_missing"
    if isinstance(error, ImportError) or "no module named" in text:
        return "dependency_missing"
    if "cuda" in text or "rocm" in text or "gpu" in text:
        return "hardware_unavailable"
    if isinstance(error, ValueError):
        return "configuration_error"
    return "incompatible"


def new_report(preset: Preset, *, quality_capture: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex[:12],
        "preset": preset.name,
        "quality_capture": quality_capture,
        "environment": environment_fingerprint(),
        "models": [],
        "human_reviews": [],
    }
