from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from visionmodelquest.benchmarks.quality import review_template
from visionmodelquest.benchmarks.reporting import comparison_recommendations, write_report
from visionmodelquest.benchmarks.runner import new_report
from visionmodelquest.cache import cache_status, default_cache_root
from visionmodelquest.config import (
    DEFAULT_MAX_TEMPERATURE_CELSIUS,
    PRESETS,
    ROOT,
    load_models,
    load_workload,
)
from visionmodelquest.hardware import temperature_readings, write_probe

ACTIVE_WORKER_PATH = ROOT / "var" / "active-worker.pid"
TEMPERATURE_POLL_INTERVAL_SECONDS = 1.0


def _hottest_temperature() -> dict[str, object] | None:
    readings = temperature_readings()
    return max(readings, key=lambda item: float(item["celsius"])) if readings else None


def _monitor_temperature(
    process: subprocess.Popen[str],
    max_temperature_celsius: float,
    finished: threading.Event,
    thermal_trip: list[dict[str, object]],
) -> None:
    while not finished.is_set():
        hottest = _hottest_temperature()
        if hottest is not None and float(hottest["celsius"]) >= max_temperature_celsius:
            thermal_trip.append(hottest)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return
        finished.wait(TEMPERATURE_POLL_INTERVAL_SECONDS)


def _run_model(
    model_key: str,
    preset_name: str,
    *,
    cache_root: Path,
    fixtures: list[str],
    quality_capture: bool,
    stability_duration_seconds: float | None,
    max_temperature_celsius: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="visionmodelquest-") as temporary:
        result_path = Path(temporary) / "result.json"
        command = [
            sys.executable,
            "-m",
            "visionmodelquest.worker",
            "--model-key",
            model_key,
            "--preset",
            preset_name,
            "--result",
            str(result_path),
            "--cache-root",
            str(cache_root),
        ]
        if fixtures:
            command.extend(["--fixtures", *fixtures])
        if quality_capture:
            command.append("--quality-capture")
        if stability_duration_seconds is not None:
            command.extend(
                ["--stability-duration-seconds", str(stability_duration_seconds)]
            )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        ACTIVE_WORKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_WORKER_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
        models = load_models()
        definition = models[model_key]
        preset = PRESETS[preset_name]
        task_multiplier = 120 if preset.name == "stability" else max(1, preset.measured_requests)
        timeout = (
            definition.startup_timeout_seconds
            + definition.generation_timeout_seconds * task_multiplier
        )
        monitor_finished = threading.Event()
        thermal_trip: list[dict[str, object]] = []
        monitor = threading.Thread(
            target=_monitor_temperature,
            args=(process, max_temperature_celsius, monitor_finished, thermal_trip),
            name="visionmodelquest-thermal-monitor",
            daemon=True,
        )
        monitor.start()
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                return {
                    "model_key": model_key,
                    "display_name": definition.display_name,
                    "model_id": definition.model_id,
                    "revision": definition.revision,
                    "status": "failed",
                    "failure_category": "timeout",
                    "failure_reason": f"worker exceeded the {timeout:g}-second deadline",
                    "process_exit": process.returncode,
                    "samples": [],
                    "aggregate": {},
                }
            except KeyboardInterrupt:
                _terminate_process_group(process)
                raise
        finally:
            monitor_finished.set()
            monitor.join(timeout=TEMPERATURE_POLL_INTERVAL_SECONDS * 2)
            try:
                if ACTIVE_WORKER_PATH.read_text(encoding="utf-8").strip() == str(process.pid):
                    ACTIVE_WORKER_PATH.unlink()
            except OSError:
                pass
        if thermal_trip:
            reading = thermal_trip[0]
            return {
                "model_key": model_key,
                "display_name": definition.display_name,
                "model_id": definition.model_id,
                "revision": definition.revision,
                "status": "failed",
                "failure_category": "thermal_limit",
                "failure_reason": (
                    f"thermal safety limit of {max_temperature_celsius:g} °C reached by "
                    f"{reading['label']} ({float(reading['celsius']):g} °C)"
                ),
                "thermal_limit_celsius": max_temperature_celsius,
                "thermal_trip": reading,
                "process_exit": process.returncode,
                "samples": [],
                "aggregate": {},
            }
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
        else:
            result = {}
        if not isinstance(result, dict) or not result:
            return {
                "model_key": model_key,
                "display_name": definition.display_name,
                "model_id": definition.model_id,
                "revision": definition.revision,
                "status": "failed",
                "failure_category": "incompatible",
                "failure_reason": (stderr or stdout or "worker wrote no result")[-500:],
                "process_exit": process.returncode,
                "samples": [],
                "aggregate": {},
            }
        result["process_exit"] = process.returncode
        return result


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def _probe(arguments: argparse.Namespace) -> int:
    result = write_probe(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate(_: argparse.Namespace) -> int:
    models = load_models()
    fixtures, questions = load_workload()
    print(
        f"Validated {len(models)} models, {len(fixtures.fixtures)} fixtures and "
        f"{len(questions.questions)} questions."
    )
    return 0


def _models(arguments: argparse.Namespace) -> int:
    models = load_models()
    cache_root = arguments.cache_root or default_cache_root()
    payload = [
        {
            "key": definition.key,
            "model_id": definition.model_id,
            "revision": definition.revision,
            "adapter": definition.adapter,
            **cache_status(definition, cache_root),
        }
        for definition in models.values()
    ]
    print(json.dumps(payload, indent=2))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    models = load_models()
    selected = arguments.models or ["mock"]
    unknown = sorted(set(selected) - models.keys())
    if unknown:
        print(f"Unknown model keys: {', '.join(unknown)}", file=sys.stderr)
        return 2
    load_workload()
    if not 40 <= arguments.max_temperature_celsius <= 120:
        print("Maximum temperature must be between 40 and 120 °C.", file=sys.stderr)
        return 2
    preset = PRESETS[arguments.preset.lower()]
    report = new_report(preset, quality_capture=arguments.quality_capture)
    report["thermal_safety"] = {
        "enabled": True,
        "maximum_celsius": arguments.max_temperature_celsius,
        "poll_interval_seconds": TEMPERATURE_POLL_INTERVAL_SECONDS,
    }
    cache_root = arguments.cache_root or default_cache_root()
    interrupted = False
    try:
        for model_key in selected:
            print(f"Running {model_key} ({preset.name})...", flush=True)
            model_result = _run_model(
                model_key,
                preset.name,
                cache_root=cache_root,
                fixtures=arguments.fixtures or [],
                quality_capture=arguments.quality_capture,
                stability_duration_seconds=arguments.duration_seconds,
                max_temperature_celsius=arguments.max_temperature_celsius,
            )
            report["models"].append(model_result)
            if model_result.get("failure_category") == "thermal_limit":
                report["thermal_abort"] = True
                break
    except KeyboardInterrupt:
        interrupted = True
    if interrupted:
        report["interrupted"] = True
    report["recommendations"] = comparison_recommendations(report["models"])
    json_path, markdown_path = write_report(report, arguments.output_directory)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    failed = interrupted or any(item.get("status") != "passed" for item in report["models"])
    return 1 if failed else 0


def _review_template(arguments: argparse.Namespace) -> int:
    payload = [
        review_template(arguments.fixture_id, arguments.question_id, arguments.output_hash)
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Review template: {arguments.output}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="VisionModelQuest offline benchmark laboratory")
    commands = root.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe", help="Print the detected environment fingerprint")
    probe.add_argument("--output", type=Path)
    probe.set_defaults(function=_probe)

    validate = commands.add_parser("validate", help="Validate versioned configuration")
    validate.set_defaults(function=_validate)

    models = commands.add_parser("models", help="Show allowlisted model cache status")
    models.add_argument("--cache-root", type=Path)
    models.set_defaults(function=_models)

    run = commands.add_parser("run", help="Run selected models sequentially")
    run.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    run.add_argument("--models", nargs="+")
    run.add_argument("--fixtures", nargs="+")
    run.add_argument("--cache-root", type=Path)
    run.add_argument("--duration-seconds", type=float)
    run.add_argument(
        "--max-temperature-celsius",
        type=float,
        default=DEFAULT_MAX_TEMPERATURE_CELSIUS,
        help="Immediately terminate benchmarking at this temperature (default: 95)",
    )
    run.add_argument("--quality-capture", action="store_true")
    run.add_argument("--output-directory", type=Path, default=ROOT / "reports")
    run.set_defaults(function=_run)

    review = commands.add_parser("review-template", help="Create a bounded human-review file")
    review.add_argument("--fixture-id", required=True)
    review.add_argument("--question-id", required=True)
    review.add_argument("--output-hash", required=True)
    review.add_argument("--output", type=Path, required=True)
    review.set_defaults(function=_review_template)
    return root


def main() -> None:
    arguments = parser().parse_args()
    try:
        code = arguments.function(arguments)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
