from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from visionmodelquest.adapters import create_adapter
from visionmodelquest.cache import resolve_snapshot
from visionmodelquest.config import ModelDefinition, load_models
from visionmodelquest.contracts import ContractError, parse_output
from visionmodelquest.explorer.inspection import mock_inspection, qwen_inspection
from visionmodelquest.explorer.lifecycle import WorkerState
from visionmodelquest.explorer.logging import local_logger
from visionmodelquest.explorer.prompts import compile_prompt
from visionmodelquest.explorer.protocol import (
    PROTOCOL_VERSION,
    Event,
    Request,
    Response,
    parse_request,
    serialise,
    timestamp,
)

COMPLETION_LIMITS = frozenset({32, 64, 128, 256, 512, 1024})


def _emit(payload: Response | Event) -> None:
    print(serialise(payload), flush=True)


class ExperimentRuntime:
    def __init__(
        self,
        definition: ModelDefinition,
        session_root: Path,
        cache_root: Path | None,
        log_root: Path,
    ) -> None:
        self.definition = definition
        self.session_root = session_root.resolve(strict=True)
        self.images_root = (self.session_root / "images").resolve(strict=True)
        self.processed_root = (self.session_root / "processed").resolve(strict=True)
        self.cache_root = cache_root
        self.processor: Any = None
        self.adapter: Any = None
        self.inspections: dict[str, dict[str, Any]] = {}
        self.state = WorkerState.STARTING
        self.last_model_activity = time.monotonic()
        self.log = local_logger("worker", log_root)
        self.model_log = local_logger("model-loading", log_root)
        self.generation_log = local_logger("generation-timings", log_root)
        self.validation_log = local_logger("validation-failures", log_root)
        self.diagnostic_log = local_logger("worker-diagnostics", log_root)

    def image_path(self, image_id: str) -> Path:
        if not image_id.isalnum() or len(image_id) != 32:
            raise ValueError("invalid opaque image ID")
        candidate = (self.images_root / f"{image_id}.png").resolve(strict=True)
        if not candidate.is_relative_to(self.images_root):
            raise ValueError("image path escaped the trusted session root")
        return candidate

    def initialise_processor(self) -> dict[str, Any]:
        if self.definition.adapter == "mock":
            self.processor = "mock"
        else:
            from transformers import AutoProcessor

            snapshot = resolve_snapshot(self.definition, self.cache_root)
            self.processor = AutoProcessor.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
            )
            if type(self.processor).__name__ != self.definition.expected_processor_class:
                raise RuntimeError("cached processor class does not match the allowlist")
            if self.definition.adapter == "qwen35":
                image_processor = self.processor.image_processor
                maximum_pixels = int(self.definition.visual_token_budget or 140) * 16**2 * 2**2
                image_processor.size["shortest_edge"] = min(65_536, maximum_pixels)
                image_processor.size["longest_edge"] = maximum_pixels
        self.state = WorkerState.PROCESSOR_READY
        return {
            "model_key": self.definition.key,
            "adapter": self.definition.adapter,
            "adapter_version": self.definition.adapter_version,
            "processor_class": (
                "MockProcessor" if self.processor == "mock" else type(self.processor).__name__
            ),
            "offline": True,
        }

    def inspect_image(self, image_id: str, budget: int) -> dict[str, Any]:
        expected_budget = self.definition.visual_token_budget or 140
        if budget != expected_budget:
            raise ValueError("visual-token budget is not allowlisted for this model")
        image_path = self.image_path(image_id)
        if self.definition.adapter == "mock":
            inspection = mock_inspection(
                image_path=image_path,
                processed_root=self.processed_root,
                visual_token_budget=budget,
            )
        elif self.definition.adapter == "qwen35":
            inspection = qwen_inspection(
                image_path=image_path,
                processed_root=self.processed_root,
                image_processor=self.processor.image_processor,
                visual_token_budget=budget,
            )
        else:
            raise ValueError("this adapter does not support preprocessing inspection in v1")
        payload = inspection.as_dict()
        try:
            manifest = json.loads(
                (self.session_root / "session.json").read_text(encoding="utf-8")
            )
            original = manifest["images"][image_id]
            payload["media_type"] = str(original["media_type"])
            payload["encoded_size"] = int(original["encoded_size"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        self.inspections[image_id] = payload
        return payload

    def load_model(self) -> dict[str, Any]:
        self.state = WorkerState.LOADING_MODEL
        self.model_log.info("load_started model_key=%s", self.definition.key)
        self.adapter = create_adapter(self.definition, self.cache_root)
        self.adapter.load()
        self.processor = self.adapter.processor if hasattr(self.adapter, "processor") else "mock"
        self.state = WorkerState.MODEL_READY
        self.last_model_activity = time.monotonic()
        self.model_log.info(
            "load_completed model_key=%s seconds=%.6f",
            self.definition.key,
            self.adapter.load_seconds,
        )
        return {
            "load_seconds": self.adapter.load_seconds,
            "runtime": _safe_runtime_details(self.adapter.runtime_details),
        }

    def unload_model(self) -> dict[str, Any]:
        self.state = WorkerState.UNLOADING
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        gc.collect()
        self.state = WorkerState.PROCESSOR_READY
        return {"unloaded": True}

    def generate(
        self,
        request_id: str,
        payload: dict[str, Any],
        fragment: Callable[[str], None],
    ) -> dict[str, Any]:
        if self.adapter is None:
            raise ValueError("load the model before generating")
        contract = _bounded_string(payload, "contract", 40)
        if contract not in self.definition.contracts:
            raise ValueError("response contract is not allowlisted for this model")
        completion_limit = int(payload.get("completion_token_limit", 0))
        if (
            completion_limit not in COMPLETION_LIMITS
            or completion_limit > self.definition.maximum_completion_tokens
        ):
            raise ValueError("completion-token limit is not allowlisted for this model")
        system = _bounded_string(payload, "system_instruction", 8_000)
        question = _bounded_string(payload, "user_question", 4_000)
        image_id = _bounded_string(payload, "image_id", 32)
        image_path = self.image_path(image_id)
        inspection = self.inspections.get(image_id)
        if inspection is None:
            inspection = self.inspect_image(
                image_id,
                int(self.definition.visual_token_budget or 140),
            )
        compiled = compile_prompt(system, question, contract)
        self.state = WorkerState.GENERATING
        started = time.perf_counter()
        if self.definition.adapter == "mock":
            generation = self.adapter.generate(
                image_path,
                f"{compiled.system_instruction}\n\n{compiled.user_content}",
            )
            fragment(generation.text)
            first_output = generation.first_output_seconds
        else:
            generation, first_output = self._generate_transformers(
                image_path, compiled.messages(), completion_limit, fragment
            )
        duration = time.perf_counter() - started
        self.state = WorkerState.MODEL_READY
        self.last_model_activity = time.monotonic()
        try:
            parse_output(contract, generation.text)
            validation_state = "valid"
            validation_message = "Output satisfies the selected response contract."
        except ContractError as error:
            validation_state = "invalid"
            validation_message = str(error)
            self.validation_log.info(
                "validation_failed request_id=%s contract=%s",
                request_id,
                contract,
            )
        output_hash = hashlib.sha256(generation.text.encode("utf-8")).hexdigest()
        completion_tokens = generation.completion_tokens
        self.generation_log.info(
            "generation_completed request_id=%s seconds=%.6f completion_tokens=%s",
            request_id,
            generation.inference_seconds,
            completion_tokens,
        )
        return {
            "raw_output": generation.text,
            "output_hash": output_hash,
            "validation_state": validation_state,
            "validation_message": validation_message,
            "input_token_count": generation.prompt_tokens,
            "visual_token_count": inspection["actual_visual_tokens"],
            "completion_token_count": completion_tokens,
            "preprocessing_seconds": generation.preprocessing_seconds,
            "time_to_first_token_seconds": first_output,
            "generation_seconds": generation.inference_seconds,
            "total_seconds": duration,
            "tokens_per_second": (
                completion_tokens / generation.inference_seconds
                if completion_tokens and generation.inference_seconds > 0
                else None
            ),
            "finish_reason": generation.finish_reason,
            "preprocessing_inspection": inspection,
            "request_id": request_id,
        }

    def _generate_transformers(
        self,
        image_path: Path,
        messages: tuple[dict[str, str], ...],
        completion_limit: int,
        fragment: Callable[[str], None],
    ) -> tuple[Any, float | None]:
        from transformers import TextIteratorStreamer

        with Image.open(image_path) as opened:
            raster = opened.convert("RGB")
        structured_messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": messages[0]["content"]}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raster},
                    {"type": "text", "text": messages[1]["content"]},
                ],
            },
        ]
        preprocessing_started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            structured_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.adapter.device)
        streamer = TextIteratorStreamer(
            self.processor,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_tokens: list[Any] = []
        error: list[BaseException] = []
        inference_started = time.perf_counter()

        def run_generation() -> None:
            try:
                with self.adapter.torch.inference_mode():
                    output = self.adapter.model.generate(
                        **inputs,
                        max_new_tokens=completion_limit,
                        do_sample=False,
                        use_cache=True,
                        streamer=streamer,
                    )
                generated_tokens.append(output)
            except BaseException as caught:
                error.append(caught)
                streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
        chunks: list[str] = []
        first_output: float | None = None
        for chunk in streamer:
            if chunk:
                if first_output is None:
                    first_output = time.perf_counter() - inference_started
                chunks.append(chunk)
                fragment(chunk)
        thread.join()
        if error:
            raise RuntimeError("model generation failed") from error[0]
        inference_seconds = time.perf_counter() - inference_started
        output = generated_tokens[0]
        generated = output[0, prompt_tokens:]
        from visionmodelquest.adapters.base import Generation

        text = "".join(chunks)
        if any(marker in text.casefold() for marker in ("<think>", "</think>", "<|channel>")):
            raise ValueError("model output exposed a reasoning channel")
        generation = Generation(
            text=text,
            preprocessing_seconds=preprocessing_seconds,
            inference_seconds=inference_seconds,
            first_output_seconds=first_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated.shape[-1]),
            finish_reason=(
                "length" if int(generated.shape[-1]) >= completion_limit else "stop"
            ),
        )
        return generation, first_output


def _safe_runtime_details(values: dict[str, object]) -> dict[str, object]:
    permitted = {
        "torch_version",
        "hip_version",
        "transformers_version",
        "processor_class",
        "model_class",
        "device",
        "device_name",
        "visual_token_budget",
        "engine",
    }
    return {key: value for key, value in values.items() if key in permitted}


def _bounded_string(payload: dict[str, Any], name: str, maximum: int) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} is missing or exceeds its bound")
    return value


def run(arguments: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    models = load_models()
    definition = models.get(arguments.model_key)
    if definition is None:
        raise SystemExit("model key is not allowlisted")
    runtime = ExperimentRuntime(
        definition,
        arguments.session_root,
        arguments.cache_root,
        arguments.log_root,
    )
    runtime.log.info("worker_started model_key=%s pid=%s", definition.key, os.getpid())
    for line in sys.stdin:
        started_at = timestamp()
        request: Request | None = None
        try:
            request = parse_request(line)
            if request.model_key and request.model_key != definition.key:
                raise ValueError("request model key does not match this worker")
            operation = request.operation
            if operation == "initialise_processor":
                result = runtime.initialise_processor()
            elif operation == "inspect_image":
                result = runtime.inspect_image(
                    _bounded_string(request.payload, "image_id", 32),
                    int(request.payload.get("visual_token_budget", 0)),
                )
            elif operation == "load_model":
                result = runtime.load_model()
            elif operation == "generate":
                current_request_id = request.request_id
                result = runtime.generate(
                    current_request_id,
                    request.payload,
                    lambda chunk, emitted_request_id=current_request_id: _emit(
                        Event(
                            protocol_version=PROTOCOL_VERSION,
                            request_id=emitted_request_id,
                            event="output_fragment",
                            worker_state=runtime.state,
                            timestamp=timestamp(),
                            payload={"text": chunk},
                        )
                    ),
                )
            elif operation == "unload_model":
                result = runtime.unload_model()
            elif operation == "shutdown":
                if runtime.adapter is not None:
                    runtime.unload_model()
                runtime.state = WorkerState.STOPPED
                result = {"shutdown": True}
            else:
                raise ValueError("unsupported operation")
            _emit(
                Response(
                    protocol_version=PROTOCOL_VERSION,
                    request_id=request.request_id,
                    operation=operation,
                    status="ok",
                    worker_state=runtime.state,
                    started_at=started_at,
                    completed_at=timestamp(),
                    result=result,
                )
            )
            runtime.log.info("protocol operation=%s status=ok", operation)
            if operation == "shutdown":
                return 0
        except Exception as error:
            operation = request.operation if request else "shutdown"
            request_id = request.request_id if request else "invalid"
            runtime.log.error(
                "protocol operation=%s status=error category=%s",
                operation,
                type(error).__name__,
            )
            runtime.diagnostic_log.error(
                "operation=%s category=%s\n%s",
                operation,
                type(error).__name__,
                traceback.format_exc(),
            )
            _emit(
                Response(
                    protocol_version=PROTOCOL_VERSION,
                    request_id=request_id,
                    operation=operation,
                    status="error",
                    worker_state=runtime.state,
                    started_at=started_at,
                    completed_at=timestamp(),
                    error={
                        "category": _error_category(error),
                        "message": _safe_error(error),
                    },
                )
            )
    return 0


def _error_category(error: Exception) -> str:
    if isinstance(error, (ValueError, KeyError)):
        return "invalid_request"
    if isinstance(error, FileNotFoundError):
        return "cache_or_image_missing"
    return "worker_error"


def _safe_error(error: Exception) -> str:
    if isinstance(error, (ValueError, ContractError)):
        return str(error)[:500]
    return "The worker could not complete the operation. See the local worker log."


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Native explorer inference worker")
    result.add_argument("--model-key", required=True)
    result.add_argument("--session-root", required=True, type=Path)
    result.add_argument("--log-root", required=True, type=Path)
    result.add_argument("--cache-root", type=Path)
    return result


def main() -> None:
    raise SystemExit(run(parser().parse_args()))


if __name__ == "__main__":
    main()
