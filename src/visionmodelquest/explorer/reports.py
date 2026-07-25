from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportSummary:
    path: Path
    run_id: str
    preset: str
    timestamp: str
    quality_capture: bool
    model_count: int
    status: str


def list_reports(root: Path) -> list[ReportSummary]:
    summaries: list[ReportSummary] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            payload = load_report(path)
        except ValueError:
            continue
        models = payload.get("models", [])
        summaries.append(
            ReportSummary(
                path=path,
                run_id=str(payload.get("run_id", path.stem)),
                preset=str(payload.get("preset", "unknown")),
                timestamp=str(payload.get("started_at") or path.stem.split("-")[0]),
                quality_capture=bool(payload.get("quality_capture")),
                model_count=len(models) if isinstance(models, list) else 0,
                status=(
                    "passed"
                    if isinstance(models, list)
                    and models
                    and all(item.get("status") == "passed" for item in models)
                    else "failed"
                ),
            )
        )
    return summaries


def load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("benchmark report is not valid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("benchmark report has an unsupported structure")
    return payload


def safe_report_details(payload: dict[str, Any], *, reveal_capture: bool = False) -> str:
    lines = [
        f"Run: {payload.get('run_id', 'unknown')}",
        f"Preset: {payload.get('preset', 'unknown')}",
        f"Quality capture: {'yes' if payload.get('quality_capture') else 'no'}",
        "",
    ]
    for model in payload.get("models", []):
        aggregate = model.get("aggregate", {})
        latency = aggregate.get("latency_seconds", {})
        throughput = aggregate.get("tokens_per_second", {})
        lines.extend(
            [
                str(model.get("display_name") or model.get("model_key") or "Unknown model"),
                f"  Status: {model.get('status', 'unknown')}",
                f"  Revision: {model.get('revision', 'unknown')}",
                f"  Requests: {aggregate.get('request_count', 0)}",
                f"  Median latency: {latency.get('median', 'n/a')} s",
                f"  Median throughput: {throughput.get('median', 'n/a')} tokens/s",
            ]
        )
        if reveal_capture and payload.get("quality_capture"):
            for sample in model.get("samples", []):
                captured = sample.get("quality_capture")
                if isinstance(captured, dict) and captured.get("output"):
                    lines.append(f"  Captured output: {captured['output']}")
        lines.append("")
    return "\n".join(lines)
