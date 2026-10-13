"""Small artifact/runtime helpers shared by experiment entry points."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import warnings


def configure_logging(verbose: bool = False) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["LOGURU_LEVEL"] = "INFO" if verbose else "WARNING"
    os.environ["TRANSFORMERS_VERBOSITY"] = "info" if verbose else "warning"
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    if not verbose:
        warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated.*", category=FutureWarning)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def package_versions() -> dict:
    from importlib.metadata import version
    return {name: version(name) for name in (
        "torch", "torchvision", "transformers", "stable-worldmodel",
        "stable-pretraining", "h5py", "numpy",
    )}
