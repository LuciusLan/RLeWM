"""Compatibility names for the unified, verified source-replay implementation."""

from pathlib import Path

from experiments.provenance import archive_entry, backend_identity
from experiments.provenance import replay_recorded_code


def code_identity(entrypoint: str = "p2") -> dict:
    return backend_identity(entrypoint)


def find_archive(entrypoint: str, recorded_code: dict) -> Path | None:
    found = archive_entry(entrypoint, recorded_code)
    return found[0] if found else None


def scope_code_identity() -> dict:
    return code_identity("p25")
