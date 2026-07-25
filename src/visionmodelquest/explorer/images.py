from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
LOCAL_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
FIXTURE_FORMATS = {**LOCAL_FORMATS, "PPM": "image/x-portable-pixmap"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    sha256: str
    width: int
    height: int
    media_type: str
    encoded_size: int
    session_path: str
    original_name: str
    provenance: str
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ImageValidationError(ValueError):
    pass


def validate_image(path: Path, *, fixture: bool = False) -> tuple[str, int, int, int]:
    if path.is_symlink():
        raise ImageValidationError("Symbolic links are not accepted for local images.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ImageValidationError("The selected image is unavailable.") from error
    if not resolved.is_file():
        raise ImageValidationError("The selected image is not a regular file.")
    size = resolved.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ImageValidationError("Images must be no larger than 16 MiB.")
    formats = FIXTURE_FORMATS if fixture else LOCAL_FORMATS
    try:
        with Image.open(resolved) as opened:
            media_type = formats.get(opened.format or "")
            if media_type is None:
                raise ImageValidationError("Choose a PNG, JPEG or WebP image.")
            width, height = opened.size
            if width <= 0 or height <= 0 or max(width, height) > MAX_IMAGE_DIMENSION:
                raise ImageValidationError("Image dimensions must not exceed 8192 × 8192.")
            opened.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ImageValidationError("The selected file is not a valid decoded image.") from error
    return media_type, width, height, size


class SessionImages:
    def __init__(self, session_root: Path, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id.isalnum():
            raise ValueError("invalid session ID")
        self.root = session_root.resolve() / self.session_id
        self.images = self.root / "images"
        self.processed = self.root / "processed"
        self.images.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.processed.mkdir(mode=0o700)
        self._records: dict[str, ImageRecord] = {}

    def import_image(
        self,
        source: Path,
        *,
        fixture: bool = False,
        provenance: str = "Local image (session only)",
    ) -> ImageRecord:
        media_type, width, height, size = validate_image(source, fixture=fixture)
        resolved = source.resolve(strict=True)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        image_id = uuid.uuid4().hex
        destination = self.images / f"{image_id}.png"
        with Image.open(resolved) as opened:
            opened.convert("RGB").save(destination, "PNG")
        destination.chmod(0o600)
        record = ImageRecord(
            image_id=image_id,
            sha256=digest,
            width=width,
            height=height,
            media_type=media_type,
            encoded_size=size,
            session_path=str(destination),
            original_name=resolved.name,
            provenance=provenance.strip()[:500],
            created_at=datetime.now(UTC).isoformat(),
        )
        self._records[image_id] = record
        self._write_manifest()
        return record

    def resolve(self, image_id: str) -> Path:
        if image_id not in self._records:
            raise ImageValidationError("Unknown session image.")
        candidate = (self.images / f"{image_id}.png").resolve(strict=True)
        if not candidate.is_relative_to(self.images.resolve()):
            raise ImageValidationError("Session image escaped its trusted directory.")
        return candidate

    def save_asset(self, image_id: str, assets_root: Path) -> Path:
        record = self._records[image_id]
        destination = assets_root.resolve() / record.sha256
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        image_destination = destination / "image.png"
        if not image_destination.exists():
            shutil.copyfile(self.resolve(image_id), image_destination)
            image_destination.chmod(0o600)
        metadata = destination / "metadata.json"
        if not metadata.exists():
            persistent_metadata = {
                "sha256": record.sha256,
                "width": record.width,
                "height": record.height,
                "media_type": record.media_type,
                "encoded_size": record.encoded_size,
                "original_name": record.original_name,
                "provenance": record.provenance,
                "created_at": record.created_at,
            }
            metadata.write_text(
                json.dumps(persistent_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata.chmod(0o600)
        return destination

    def close(self) -> None:
        if self.root.exists() and self.root.is_relative_to(self.root.parent):
            shutil.rmtree(self.root)

    def _write_manifest(self) -> None:
        manifest_path = self.root / "session.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "owner_pid": os.getpid(),
                    "images": {key: value.as_dict() for key, value in self._records.items()},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)


def clean_stale_sessions(sessions_root: Path, *, active_session: str | None = None) -> int:
    if not sessions_root.exists():
        return 0
    removed = 0
    resolved_root = sessions_root.resolve()
    for child in sessions_root.iterdir():
        if not child.is_dir() or child.name == active_session:
            continue
        resolved = child.resolve()
        if resolved.parent != resolved_root:
            continue
        try:
            manifest = json.loads((resolved / "session.json").read_text(encoding="utf-8"))
            owner_pid = int(manifest.get("owner_pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            owner_pid = 0
        if owner_pid > 1 and Path(f"/proc/{owner_pid}").exists():
            continue
        shutil.rmtree(resolved)
        removed += 1
    return removed
