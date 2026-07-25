import subprocess
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
