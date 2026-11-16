"""Stage source identities and verified replay across interface migrations."""

import os
from pathlib import Path
import sys

from lewm_tta.runtime import file_hash, object_hash, read_json
from .catalog import ROOT


def frontend_sources(backend):
    sources = sorted((ROOT / "codes/experiments").glob("*.py")) + [
        ROOT / "codes/experiment.py", ROOT / "codes/run.sh", ROOT / "codes/cuda_env.sh"]
    if backend == "p1":
        sources += [ROOT / "codes/experiments/backends/__init__.py", ROOT / "codes/experiments/backends/p1.py"]
    if backend == "p3_core":
        sources += [ROOT / "codes/run-p3-core.sh"]
    return sources


def backend_identity(backend):
    library = ROOT / "codes/lewm_tta"
    sources = sorted(library.glob("*.py")) + frontend_sources(backend)
    sources += sorted((library / "preprocessing").rglob("*.py"))
    sources += sorted((library / "shifted").rglob("*.py"))
    if backend == "p3_core":
        for name in ("core", "interleave", "rollout", "planning", "proposals", "outcomes", "simulation", "collection", "appearance"):
            sources += sorted((library / name).rglob("*.py"))
    if backend in ("p2", "p25", "p2_followup", "p3a", "p3b", "p3c", "p3_temporal", "p3_interleave", "rank_coverage", "sim"):
        sources += sorted((library / "rollout").rglob("*.py"))
    if backend in ("p3a", "p3b", "p3c", "p3_temporal", "p3_interleave", "rank_coverage", "sim"):
        sources += sorted((library / "planning").rglob("*.py"))
    if backend in ("p3b", "p3c", "p3_interleave", "sim"):
        sources += sorted((library / "proposals").rglob("*.py"))
    if backend == "p3_interleave":
        sources += sorted((library / "interleave").rglob("*.py"))
    if backend == "p3_success":
        for name in ("outcomes", "rollout", "simulation", "collection", "appearance"):
            sources += sorted((library / name).rglob("*.py"))
    if backend == "rank_coverage":
        sources += sorted((library / "rank_coverage").rglob("*.py"))
    if backend in ("sim", "collect"):
        sources += sorted((library / "simulation").rglob("*.py"))
    if backend in ("p2", "p25", "p2_followup", "p3a", "p3b", "p3c", "p3_temporal", "p3_interleave", "sim", "collect"):
        sources += sorted((library / "collection").rglob("*.py"))
    if backend in ("p2", "p25", "p2_followup", "p3a", "p3b", "p3c", "p3_temporal", "p3_interleave", "sim"):
        sources += sorted((library / "appearance").rglob("*.py"))
    if backend not in ("p1", "p2", "p25", "p2_followup", "p3a", "p3b", "p3c", "p3_temporal", "p3_interleave", "p3_core", "p3_success", "rank_coverage", "sim", "collect"):
        raise ValueError(f"Unknown runtime backend: {backend}")
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(set(sources))}


def archive_entry(backend, recorded_code, *, root=None):
    workspace = ROOT if root is None else Path(root)
    for manifest_path in sorted((workspace / "artifacts/code-archives").glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        entry = manifest.get("entrypoints", {}).get(backend)
        if entry is None or entry.get("code") != recorded_code:
            continue
        directory = manifest_path.parent.resolve()
        files = manifest["files"]
        if manifest["archive_id"] != object_hash(files) or any(files.get(name) != digest for name, digest in recorded_code.items()):
            raise ValueError("Archived source metadata checksum changed")
        for name, digest in files.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != digest:
                raise ValueError(f"Archived source changed: {name}")
        script = (directory / entry["script"]).resolve()
        if not script.is_relative_to(directory) or entry["script"] not in recorded_code:
            raise ValueError("Archive entrypoint is absent from its recorded source identity")
        prefix = entry.get("arguments", [])
        if not isinstance(prefix, list) or any(not isinstance(item, str) for item in prefix):
            raise ValueError("Invalid archive entrypoint arguments")
        return script, prefix
    return None


def archived_command(backend, recorded_code):
    found = archive_entry(backend, recorded_code)
    return [sys.executable, "-X", "faulthandler", "-u", str(found[0]), *found[1]] if found else None


def replay_recorded_code(backend, source, arguments):
    recorded = source.get("code")
    if not recorded or recorded == backend_identity(backend):
        return
    command = archived_command(backend, recorded)
    if command is None:
        raise ValueError("Recorded source differs and has no verified archive; refusing to relabel the report")
    print(f"Using verified archived {backend} implementation for this report.", flush=True)
    os.execv(sys.executable, [*command, *arguments])
