import re
import tomllib
from pathlib import Path


def test_rocm_wheels_are_explicit_and_hash_pinned():
    requirements = Path("requirements-rocm72.txt").read_text(encoding="utf-8").splitlines()
    direct_wheels = {
        line.partition(" @ ")[0]: line
        for line in requirements
        if " @ https://repo.radeon.com/rocm/" in line
    }
    assert set(direct_wheels) == {"torch", "torchvision", "triton"}
    for line in direct_wheels.values():
        assert re.search(r"#sha256=[0-9a-f]{64}$", line)
        assert "rocm-rel-7.2.1" in line
        assert "cp312-cp312-linux_x86_64.whl" in line


def test_rocm_environment_includes_smolvlm_processor_dependency():
    configuration = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = configuration["project"]["optional-dependencies"]["rocm"]
    setup = Path("scripts/setup.ps1").read_text(encoding="utf-8")

    assert "num2words==0.5.14" in dependencies
    assert "'.[dev,rocm]'" in setup
    assert "import num2words, torch, transformers" in setup
