from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionmodelquest.adapters import create_adapter
from visionmodelquest.benchmarks.reporting import aggregate_report
from visionmodelquest.benchmarks.runner import run_adapter
from visionmodelquest.config import PRESETS, ROOT, load_models, load_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal allowlisted benchmark worker")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--fixtures", nargs="*")
    parser.add_argument("--quality-capture", action="store_true")
    parser.add_argument("--stability-duration-seconds", type=float)
    arguments = parser.parse_args()
    models = load_models()
    definition = models.get(arguments.model_key)
    if definition is None:
        raise SystemExit("model key is not allowlisted")
    fixtures, questions = load_workload()
    adapter = create_adapter(definition, arguments.cache_root)
    result = run_adapter(
        definition,
        adapter,
        fixtures,
        questions,
        PRESETS[arguments.preset],
        root=ROOT,
        selected_fixture_ids=set(arguments.fixtures) if arguments.fixtures else None,
        quality_capture=arguments.quality_capture,
        stability_duration_seconds=arguments.stability_duration_seconds,
    )
    result["aggregate"] = aggregate_report({"models": [result]})
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    arguments.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
