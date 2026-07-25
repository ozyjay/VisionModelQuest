from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionmodelquest.explorer.images import (
    ImageValidationError,
    SessionImages,
    clean_stale_sessions,
    validate_image,
)


def image(path: Path, *, image_format: str = "PNG", size: tuple[int, int] = (8, 6)) -> Path:
    Image.new("RGB", size, (30, 80, 140)).save(path, image_format)
    return path


def test_session_import_validates_signature_and_uses_opaque_copy(tmp_path: Path) -> None:
    source = image(tmp_path / "mislabelled.txt")
    session = SessionImages(tmp_path / "sessions")

    record = session.import_image(source)

    assert record.media_type == "image/png"
    assert record.width == 8
    assert Path(record.session_path).parent == session.images
    assert Path(record.session_path).suffix == ".png"
    session.close()
    assert not session.root.exists()


def test_image_validation_rejects_corrupt_oversized_dimensions_and_symlinks(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ImageValidationError):
        validate_image(corrupt)

    too_large = image(tmp_path / "large.png", size=(8193, 1))
    with pytest.raises(ImageValidationError, match="8192"):
        validate_image(too_large)

    target = image(tmp_path / "target.png")
    link = tmp_path / "linked.png"
    link.symlink_to(target)
    with pytest.raises(ImageValidationError, match="Symbolic"):
        validate_image(link)


def test_assets_are_content_addressed_and_stale_sessions_are_cleaned(tmp_path: Path) -> None:
    session = SessionImages(tmp_path / "sessions")
    record = session.import_image(image(tmp_path / "source.png"))
    asset = session.save_asset(record.image_id, tmp_path / "assets")
    assert asset.name == record.sha256
    assert (asset / "image.png").is_file()
    session.close()

    stale = tmp_path / "sessions" / "stale"
    stale.mkdir()
    assert clean_stale_sessions(tmp_path / "sessions") == 1
    assert not stale.exists()
