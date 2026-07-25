import re
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
