from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import psutil
from PIL import Image


def request(request_id: str, operation: str, payload: dict[str, object] | None = None) -> str:
    return json.dumps(
        {
            "protocol_version": 1,
            "kind": "request",
            "request_id": request_id,
            "operation": operation,
            "model_key": "mock",
            "payload": payload or {},
        }
    )


def test_mock_worker_initialises_inspects_generates_and_shuts_down(tmp_path: Path) -> None:
    session = tmp_path / "session"
    images = session / "images"
    processed = session / "processed"
    images.mkdir(parents=True)
    processed.mkdir()
    image_id = "a" * 32
    Image.new("RGB", (8, 6), "blue").save(images / f"{image_id}.png")
    lines = [
        request("one", "initialise_processor"),
        request(
            "two",
            "inspect_image",
            {"image_id": image_id, "visual_token_budget": 140},
        ),
        request("three", "load_model"),
        request(
            "four",
            "generate",
            {
                "image_id": image_id,
                "contract": "free_text_v1",
                "system_instruction": "Describe visible evidence.",
                "user_question": "What is visible?",
                "completion_token_limit": 64,
            },
        ),
        request("five", "shutdown"),
    ]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionmodelquest.experiment_worker",
            "--model-key",
            "mock",
            "--session-root",
            str(session),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    messages = [json.loads(line) for line in completed.stdout.splitlines()]
    inspection = next(item for item in messages if item.get("request_id") == "two")
    assert inspection["result"]["processed_width"] == 320
    assert inspection["result"]["processed_height"] == 224
    assert inspection["result"]["actual_visual_tokens"] == 70
    generation = next(
        item
        for item in messages
        if item.get("kind") == "response" and item.get("request_id") == "four"
    )
    assert generation["result"]["validation_state"] == "valid"
    assert generation["result"]["output_hash"]
    assert generation["result"]["visual_token_count"] == 70
    assert generation["result"]["preprocessing_inspection"]["image_grid_thw"] == [1, 14, 20]


def test_worker_opens_no_network_listener(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "images").mkdir(parents=True)
    (session / "processed").mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "visionmodelquest.experiment_worker",
            "--model-key",
            "mock",
            "--session-root",
            str(session),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdin and process.stdout
        process.stdin.write(request("start", "initialise_processor") + "\n")
        process.stdin.flush()
        assert json.loads(process.stdout.readline())["status"] == "ok"
        assert psutil.Process(process.pid).net_connections(kind="inet") == []
        process.stdin.write(request("stop", "shutdown") + "\n")
        process.stdin.flush()
        assert json.loads(process.stdout.readline())["status"] == "ok"
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
