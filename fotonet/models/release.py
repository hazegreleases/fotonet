"""Resolve the official fotonete checkpoint from its GitHub release."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


RELEASE_VERSION = "v1.0.0"
RELEASE_URL = (
    "https://github.com/hazegreleases/fotonet/releases/download/"
    f"{RELEASE_VERSION}/fotonete.pt"
)
RELEASE_SHA256 = "1A953D4ABBAC6D5292FEA1AD344DBB361396DD57BFDBE4EE93AD62185EAA7730"
CHECKPOINT_FILENAME = "fotonete.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Fotonete checkpoint not found: {path}")
    actual = _sha256(path)
    if actual != RELEASE_SHA256:
        raise RuntimeError(
            f"Fotonete checkpoint checksum mismatch for '{path}': "
            f"expected {RELEASE_SHA256}, got {actual}."
        )
    return path


def resolve_release_checkpoint(model_ref: str = "fotonete") -> Path:
    """Return the verified v1 checkpoint, downloading it once when needed."""
    if str(model_ref).strip().lower() != "fotonete":
        raise ValueError(f"No released checkpoint is registered for model '{model_ref}'.")

    override = os.environ.get("FOTONETE_MODEL_PATH")
    if override:
        return _validate(Path(override).expanduser())

    local_checkpoint = Path.cwd() / CHECKPOINT_FILENAME
    if local_checkpoint.is_file():
        return _validate(local_checkpoint)

    cache_dir = Path(
        os.environ.get("FOTONETE_CACHE_DIR", Path.home() / ".cache" / "fotonet" / "models")
    ).expanduser()
    cached_checkpoint = cache_dir / CHECKPOINT_FILENAME
    if cached_checkpoint.is_file():
        try:
            return _validate(cached_checkpoint)
        except (FileNotFoundError, RuntimeError):
            cached_checkpoint.unlink(missing_ok=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_dir, prefix="fotonete.", suffix=".download", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(RELEASE_URL, headers={"User-Agent": "fotonet-release-resolver/1"})
            with urlopen(request, timeout=60) as response, temporary:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    temporary.write(chunk)
        _validate(temporary_path)
        os.replace(temporary_path, cached_checkpoint)
        return cached_checkpoint
    except OSError as exc:
        raise RuntimeError(
            "Could not download the official fotonete checkpoint. "
            f"Install it manually or set FOTONETE_MODEL_PATH. Download URL: {RELEASE_URL}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

