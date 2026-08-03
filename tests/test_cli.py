import subprocess
import threading
from pathlib import Path

from visionmodelquest import cli


class FakeTimedOutProcess:
    pid = 424242
    returncode = None

    def communicate(self, timeout):
        raise subprocess.TimeoutExpired(["worker"], timeout)

    def poll(self):
        return None


def test_timeout_becomes_structured_result(monkeypatch, tmp_path: Path):
    process = FakeTimedOutProcess()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        cli,
        "_terminate_process_group",
        lambda candidate: setattr(candidate, "returncode", -15),
    )
    monkeypatch.setattr(cli, "ACTIVE_WORKER_PATH", tmp_path / "worker.pid")
    result = cli._run_model(
        "mock",
        "quick",
        cache_root=tmp_path,
        fixtures=[],
        quality_capture=False,
        stability_duration_seconds=None,
        max_temperature_celsius=95,
    )
    assert result["status"] == "failed"
    assert result["failure_category"] == "timeout"
    assert not (tmp_path / "worker.pid").exists()


def test_process_cleanup_targets_the_worker_group(monkeypatch):
    calls = []

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15

    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    cli._terminate_process_group(Process())
    assert calls[0][0] == 123


def test_thermal_monitor_kills_worker_at_limit(monkeypatch):
    calls = []

    class Process:
        pid = 456

        def poll(self):
            return None

    monkeypatch.setattr(
        cli,
        "_hottest_temperature",
        lambda: {"sensor": "/sensor", "label": "GPU", "celsius": 95.0},
    )
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    trip = []

    cli._monitor_temperature(Process(), 95.0, threading.Event(), trip)

    assert trip == [{"sensor": "/sensor", "label": "GPU", "celsius": 95.0}]
    assert calls == [(456, cli.signal.SIGKILL)]


def test_thermal_monitor_ignores_temperature_below_limit(monkeypatch):
    finished = threading.Event()
    calls = []

    def reading():
        finished.set()
        return {"sensor": "/sensor", "label": "GPU", "celsius": 94.9}

    monkeypatch.setattr(cli, "_hottest_temperature", reading)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    cli._monitor_temperature(FakeTimedOutProcess(), 95.0, finished, [])

    assert calls == []


def test_thermal_trip_becomes_structured_result(monkeypatch, tmp_path: Path):
    killed = threading.Event()

    class Process:
        pid = 789
        returncode = None

        def communicate(self, timeout):
            assert killed.wait(timeout=1)
            self.returncode = -9
            return "", ""

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        cli,
        "_hottest_temperature",
        lambda: {"sensor": "/sensor", "label": "GPU", "celsius": 96.0},
    )
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: killed.set())
    monkeypatch.setattr(cli, "ACTIVE_WORKER_PATH", tmp_path / "worker.pid")

    result = cli._run_model(
        "mock",
        "quick",
        cache_root=tmp_path,
        fixtures=[],
        quality_capture=False,
        stability_duration_seconds=None,
        max_temperature_celsius=95,
    )

    assert result["status"] == "failed"
    assert result["failure_category"] == "thermal_limit"
    assert result["thermal_trip"]["celsius"] == 96.0
    assert not (tmp_path / "worker.pid").exists()
