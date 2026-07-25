from pathlib import Path

from visionmodelquest import hardware


def test_memory_fallbacks_are_nullable(monkeypatch):
    monkeypatch.setattr(hardware, "_read_first_int", lambda paths: None)
    result = hardware.memory_snapshot()
    assert result["gtt_used_bytes"] is None
    assert result["vram_used_bytes"] is None
    assert result["host_total_bytes"] > 0


def test_missing_rocm_commands_do_not_fail_probe(monkeypatch):
    monkeypatch.setattr(hardware, "_command_version", lambda arguments: None)
    result = hardware.environment_fingerprint()
    assert result["rocm"]["rocminfo"] is None
    assert "python" in result


def test_probe_can_be_written(tmp_path: Path):
    path = tmp_path / "probe.json"
    hardware.write_probe(path)
    assert path.is_file()

