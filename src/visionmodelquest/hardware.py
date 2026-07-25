from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

PACKAGE_NAMES = ("pillow", "psutil", "pydantic", "torch", "torchvision", "transformers")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"ID", "NAME", "PRETTY_NAME", "VERSION_ID"}:
            result[key.lower()] = value.strip().strip('"')
    return result


def _command_version(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text[:2000] if text else None


def _torch_details() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "version": None,
            "hip_version": None,
            "cuda_available": False,
            "device_name": None,
            "device_count": 0,
        }
    cuda_available = bool(torch.cuda.is_available())
    return {
        "available": True,
        "version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def temperature_readings() -> list[dict[str, object]]:
    readings: list[dict[str, object]] = []
    thermal_root = Path("/sys/class/thermal")
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        try:
            value = float((zone / "temp").read_text().strip()) / 1000
            label = (zone / "type").read_text().strip()
        except (OSError, ValueError):
            continue
        readings.append({"sensor": str(zone), "label": label, "celsius": value})
    hwmon_root = Path("/sys/class/hwmon")
    for sensor in sorted(hwmon_root.glob("hwmon*/temp*_input")):
        try:
            value = float(sensor.read_text().strip()) / 1000
            label_path = sensor.with_name(sensor.name.replace("_input", "_label"))
            label = label_path.read_text().strip() if label_path.is_file() else sensor.stem
        except (OSError, ValueError):
            continue
        readings.append({"sensor": str(sensor), "label": label, "celsius": value})
    return readings


def memory_snapshot() -> dict[str, int | None]:
    memory = psutil.virtual_memory()
    process = psutil.Process()
    return {
        "host_total_bytes": int(memory.total),
        "host_available_bytes": int(memory.available),
        "process_rss_bytes": int(process.memory_info().rss),
        "gtt_used_bytes": _read_first_int(
            (
                Path("/sys/class/drm/card1/device/mem_info_gtt_used"),
                Path("/sys/class/drm/card0/device/mem_info_gtt_used"),
            )
        ),
        "vram_used_bytes": _read_first_int(
            (
                Path("/sys/class/drm/card1/device/mem_info_vram_used"),
                Path("/sys/class/drm/card0/device/mem_info_vram_used"),
            )
        ),
    }


def _read_first_int(paths: tuple[Path, ...]) -> int | None:
    for path in paths:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def environment_fingerprint() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os_release": _os_release(),
        "packages": _package_versions(),
        "torch": _torch_details(),
        "rocm": {
            "rocminfo": _command_version(["rocminfo", "--version"]),
            "rocm_smi": _command_version(["rocm-smi", "--showproductname"]),
            "gfx_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        },
        "memory": memory_snapshot(),
        "temperatures": temperature_readings(),
    }


def write_probe(path: Path | None = None) -> dict[str, Any]:
    result = environment_fingerprint()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

