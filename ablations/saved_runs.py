"""Small model-free helpers for immutable saved-run ablations."""

from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=None)
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def checked(path):
    value = read(path)
    require(value.get("fingerprint") == identity({k: v for k, v in value.items() if k != "fingerprint"}),
            f"Changed manifest: {path}")
    return value


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, f"Existing metadata differs; choose another output directory: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def record_rows(report):
    path = Path(report["records_file"])
    require(path.is_file() and sha(path) == report["records_sha256"], f"Missing/changed records: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def executor(report, backend):
    """Use exactly the measured source; never quietly mix old/new WM code."""
    code = report["code"]
    if all((ROOT/name).is_file() and sha(ROOT/name) == digest for name, digest in code.items()):
        return [sys.executable, "-X", "faulthandler", "-u", str(ROOT/"codes/experiment.py"), "legacy", backend]
    for path in sorted((ROOT/"artifacts/code-archives").glob("*/manifest.json")):
        archive = read(path)
        entry = archive.get("entrypoints", {}).get(backend)
        if not entry or entry.get("code") != code:
            continue
        files = archive["files"]
        require(archive["archive_id"] == identity(files) and all(files.get(n) == d for n, d in code.items()),
                f"Invalid archive identity: {path}")
        for name, digest in files.items():
            file = (path.parent/name).resolve()
            require(file.is_relative_to(path.parent.resolve()) and file.is_file() and sha(file) == digest,
                    f"Changed archived file: {file}")
        require(entry["script"] in code, "Archive entrypoint is not recorded")
        return [sys.executable, "-X", "faulthandler", "-u", str(path.parent/entry["script"]), *entry.get("arguments", [])]
    raise ValueError("The B32 reference has no matching execution source/archive")
