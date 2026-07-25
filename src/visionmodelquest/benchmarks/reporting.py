from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionmodelquest.benchmarks.metrics import summarise


def aggregate_report(report: dict[str, Any]) -> dict[str, Any]:
    samples = [
        sample
        for result in report.get("models", [])
        for sample in result.get("samples", [])
        if sample.get("status") == "passed" and not sample.get("warmup")
    ]
    latency = [float(item["total_seconds"]) for item in samples]
    inference = [float(item["inference_seconds"]) for item in samples]
    throughput = [
        float(item["tokens_per_second"])
        for item in samples
        if item.get("tokens_per_second") is not None
    ]
    structured = [item for item in samples if item.get("contract") == "scene_json_v1"]
    valid = sum(bool(item.get("structured_output_valid")) for item in structured)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[
            (
                str(sample.get("fixture_id")),
                str(sample.get("question_id")),
                str(sample.get("contract")),
            )
        ].append(sample)
    workloads = []
    for (fixture_id, question_id, contract), workload_samples in sorted(grouped.items()):
        workload_latency = [float(item["total_seconds"]) for item in workload_samples]
        workload_throughput = [
            float(item["tokens_per_second"])
            for item in workload_samples
            if item.get("tokens_per_second") is not None
        ]
        workloads.append(
            {
                "fixture_id": fixture_id,
                "question_id": question_id,
                "contract": contract,
                "latency_seconds": summarise(workload_latency),
                "tokens_per_second": summarise(workload_throughput),
                "structured_output_success_rate": (
                    sum(bool(item.get("structured_output_valid")) for item in workload_samples)
                    / len(workload_samples)
                    if contract == "scene_json_v1"
                    else None
                ),
            }
        )
    return {
        "request_count": len(samples),
        "latency_seconds": summarise(latency),
        "inference_seconds": summarise(inference),
        "tokens_per_second": summarise(throughput),
        "structured_output_success_rate": valid / len(structured) if structured else None,
        "workloads": workloads,
    }


def comparison_recommendations(models: list[dict[str, Any]]) -> dict[str, object]:
    passed = [model for model in models if model.get("status") == "passed"]
    recommendations: dict[str, object] = {
        "fastest_usable_model": None,
        "best_overall_balance": None,
        "highest_quality": None,
        "lowest_memory": None,
        "best_structured_output_reliability": None,
        "unsuitable_or_incompatible": [
            {
                "model_key": model.get("model_key"),
                "failure_category": model.get("failure_category"),
            }
            for model in models
            if model.get("status") != "passed"
        ],
    }
    with_latency = [
        model
        for model in passed
        if model.get("aggregate", {}).get("latency_seconds", {}).get("median") is not None
    ]
    if with_latency:
        recommendations["fastest_usable_model"] = min(
            with_latency,
            key=lambda model: float(model["aggregate"]["latency_seconds"]["median"]),
        )["model_key"]
    with_reliability = [
        model
        for model in passed
        if model.get("aggregate", {}).get("structured_output_success_rate") is not None
    ]
    if with_reliability:
        recommendations["best_structured_output_reliability"] = max(
            with_reliability,
            key=lambda model: float(model["aggregate"]["structured_output_success_rate"]),
        )["model_key"]
    with_memory = [
        model
        for model in passed
        if model.get("resource_sampling", {}).get("peak_process_rss_bytes") is not None
    ]
    if with_memory:
        recommendations["lowest_memory"] = min(
            with_memory,
            key=lambda model: int(model["resource_sampling"]["peak_process_rss_bytes"]),
        )["model_key"]
    recommendations["recommendation_note"] = (
        "Best overall balance and highest quality require completed human review; "
        "they are not inferred from performance."
    )
    return recommendations


def _markdown(report: dict[str, Any]) -> str:
    recommendations = report.get("recommendations", {})

    def recommendation(key: str) -> str:
        return str(recommendations.get(key) or "not established")

    lines = [
        "# VisionModelQuest benchmark report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Preset: `{report['preset']}`",
        f"- Quality capture: `{str(report['quality_capture']).lower()}`",
        "",
        "## Compatibility and performance",
        "",
        "| Model | Status | Requests | Median latency (s) | p95 latency (s) | JSON success |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in report.get("models", []):
        aggregate = model.get("aggregate", {})
        latency = aggregate.get("latency_seconds", {})
        success = aggregate.get("structured_output_success_rate")
        lines.append(
            "| {name} | {status} | {count} | {median} | {p95} | {success} |".format(
                name=model.get("display_name", model.get("model_key")),
                status=model.get("status"),
                count=aggregate.get("request_count", 0),
                median=_display(latency.get("median")),
                p95=_display(latency.get("p95")),
                success=_display(success, percentage=True),
            )
        )
    lines.extend(
        [
            "",
            "## Workload components",
            "",
            "| Model | Fixture | Question | Contract | Median latency (s) | p95 latency (s) |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for model in report.get("models", []):
        for workload in model.get("aggregate", {}).get("workloads", []):
            latency = workload.get("latency_seconds", {})
            lines.append(
                "| {model} | {fixture} | {question} | {contract} | {median} | {p95} |".format(
                    model=model.get("model_key"),
                    fixture=workload.get("fixture_id"),
                    question=workload.get("question_id"),
                    contract=workload.get("contract"),
                    median=_display(latency.get("median")),
                    p95=_display(latency.get("p95")),
                )
            )
    lines.extend(
        [
            "",
            "## Recommendations by use case",
            "",
            f"- Fastest usable: `{recommendation('fastest_usable_model')}`",
            f"- Best overall balance: `{recommendation('best_overall_balance')}`",
            f"- Highest quality: `{recommendation('highest_quality')}`",
            f"- Lowest memory: `{recommendation('lowest_memory')}`",
            (
                "- Best structured-output reliability: "
                f"`{recommendation('best_structured_output_reliability')}`"
            ),
            "",
            "## Interpretation",
            "",
            "No cross-workload composite score is calculated. Quality, latency, throughput, "
            "memory, temperature and structured-output reliability remain separate evidence.",
            "",
            "Human quality scores are authoritative when supplied. Missing scores are reported "
            "as not reviewed and are not inferred from performance.",
            "",
            "Normal-mode samples contain fixture and question IDs plus output hashes, not prompts, "
            "images or model output.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: object, *, percentage: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number * 100:.1f}%" if percentage else f"{number:.3f}"


def write_report(
    report: dict[str, Any],
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}-{report['run_id']}"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path
