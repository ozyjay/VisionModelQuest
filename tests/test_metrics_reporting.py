import json
from pathlib import Path

import pytest

from visionmodelquest.benchmarks.metrics import nearest_rank_percentile, summarise
from visionmodelquest.benchmarks.reporting import (
    aggregate_report,
    comparison_recommendations,
    write_report,
)


def test_nearest_rank_p95():
    assert nearest_rank_percentile(list(range(1, 21)), 0.95) == 19
    assert nearest_rank_percentile([4], 0.95) == 4
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 0.95)


def test_statistics_preserve_missing_values():
    assert summarise([])["median"] is None


def test_report_generation_and_privacy(tmp_path: Path):
    model = {
        "model_key": "mock",
        "display_name": "Mock",
        "status": "passed",
        "samples": [
            {
                "status": "passed",
                "warmup": False,
                "total_seconds": 1.0,
                "inference_seconds": 0.5,
                "tokens_per_second": 10.0,
                "structured_output_valid": True,
                "fixture_id": "safe-id",
                "question_id": "safe-question",
                "output_hash": "a" * 64,
            }
        ],
    }
    model["aggregate"] = aggregate_report({"models": [model]})
    report = {
        "run_id": "test",
        "preset": "quick",
        "quality_capture": False,
        "environment": {},
        "models": [model],
    }
    json_path, markdown_path = write_report(report, tmp_path)
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["models"][0]["samples"][0]["fixture_id"] == "safe-id"
    assert "prompt" not in json_path.read_text(encoding="utf-8")
    assert "Compatibility and performance" in markdown_path.read_text(encoding="utf-8")
    assert model["aggregate"]["workloads"][0]["fixture_id"] == "safe-id"


def test_aggregate_counts_failed_requests_and_structured_outputs():
    samples = [
        {
            "status": "passed",
            "warmup": True,
            "contract": "scene_json_v1",
            "structured_output_valid": True,
            "fixture_id": "warmup",
            "question_id": "scene_json",
            "total_seconds": 0.5,
            "inference_seconds": 0.4,
            "tokens_per_second": 10.0,
        },
        {
            "status": "passed",
            "warmup": False,
            "contract": "scene_json_v1",
            "structured_output_valid": True,
            "fixture_id": "valid",
            "question_id": "scene_json",
            "total_seconds": 1.0,
            "inference_seconds": 0.8,
            "tokens_per_second": 10.0,
        },
        {
            "status": "failed",
            "warmup": False,
            "contract": "scene_json_v1",
            "failure_category": "output_invalid",
            "fixture_id": "invalid",
            "question_id": "scene_json",
            "total_seconds": 1.2,
        },
    ]

    aggregate = aggregate_report({"models": [{"samples": samples}]})

    assert aggregate["request_count"] == 2
    assert aggregate["latency_seconds"]["count"] == 1
    assert aggregate["structured_output_success_rate"] == 0.5
    invalid_workload = next(
        workload
        for workload in aggregate["workloads"]
        if workload["fixture_id"] == "invalid"
    )
    assert invalid_workload["latency_seconds"]["count"] == 0
    assert invalid_workload["structured_output_success_rate"] == 0.0


def test_recommendations_do_not_infer_subjective_quality():
    model = {
        "model_key": "mock",
        "status": "passed",
        "aggregate": {
            "latency_seconds": {"median": 1.0},
            "structured_output_success_rate": 1.0,
        },
        "resource_sampling": {"peak_process_rss_bytes": 10},
    }
    result = comparison_recommendations([model])
    assert result["fastest_usable_model"] == "mock"
    assert result["lowest_memory"] == "mock"
    assert result["highest_quality"] is None
