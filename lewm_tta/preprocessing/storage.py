"""Filesystem-aware space guards, including safe credit for resumable assets."""

from pathlib import Path
import shutil
import subprocess

from ..runtime import file_hash, object_hash, package_versions, read_json, write_json
from .hdf5 import raw_dataset_path

GIB = 2**30
METADATA_ALLOWANCE = 64 * 1024**2


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def allocated(path):
    return path.stat().st_blocks * 512 if path.exists() else 0


def cache_reservation(config, inspection):
    """Only credit partial assets from the same source/checkpoint/builders."""
    checkpoint = Path(config["paths"]["checkpoint"])
    library = Path(__file__).resolve().parents[1]
    builders = [library / name for name in ("data.py", "model.py", "observation_cache.py", "frame_bank.py",
                                            "indexed_keys.py", "retrieval.py", "cache.py")]
    builders += sorted(Path(__file__).parent.glob("*.py"))
    commit = subprocess.run(["git", "-C", config["paths"]["repo"], "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return {"version": 1, "source": inspection["source"], "data": config["data"],
            "paths": config["paths"], "packages": package_versions(),
            "lewm_commit": commit,
            "checkpoint": {name: file_hash(checkpoint / name) for name in ("config.json", "weights.pt")},
            "native_code": {name: file_hash(Path(config['paths']['repo']) / name)
                            for name in ("jepa.py", "module.py", "utils.py")},
            "retrieval": config["retrieval"], "payload": inspection["cache"],
            "builders": {str(path.relative_to(library)): file_hash(path) for path in builders}}


def cache_credit(config, inspection):
    directory = Path(config["paths"]["cache_dir"])
    record = directory / ".storage-reservation.json"
    if inspection["source"] is None or not record.exists():
        return 0
    expected = cache_reservation(config, inspection)
    saved = read_json(record)
    if saved.get("fingerprint") != object_hash(expected) or saved.get("identity") != expected:
        raise ValueError(f"Cache reservation differs from this source/config/builders: {record}. "
                         "Use a new --cache-root, or restore the original configuration.")
    return min(inspection["cache"]["total_bytes"], sum(allocated(path) for path in directory.rglob("*.npy")))


def reserve_cache(config, inspection):
    directory = Path(config["paths"]["cache_dir"])
    record = directory / ".storage-reservation.json"
    identity = cache_reservation(config, inspection)
    value = {"fingerprint": object_hash(identity), "identity": identity}
    if record.exists() and read_json(record) != value:
        raise ValueError(f"Cache reservation changed: {record}")
    write_json(record, value)


def storage_plan(jobs, *, include_cache=True, include_extraction=True, keep_free_gib=6,
                 scratch_keep_free_gib=10):
    """Group requests by st_dev, including when two path prefixes share a volume."""
    volumes = {}

    def request(path, amount, reserve, label):
        parent = existing_parent(path)
        key = parent.stat().st_dev
        group = volumes.setdefault(key, {"filesystem_example": str(parent),
            "available_bytes": shutil.disk_usage(parent).free, "additional_bytes": 0,
            "reserve_bytes": 0, "requests": []})
        group["additional_bytes"] += max(0, amount)
        group["reserve_bytes"] = max(group["reserve_bytes"], int(reserve * GIB))
        group["requests"].append({"purpose": label, "path": str(path), "additional_bytes": max(0, amount)})

    for config, item in jobs:
        if include_extraction and item["extraction_bytes"]:
            target = raw_dataset_path(config)
            partial = target.with_name(target.name + ".partial")
            receipt = target.with_name(target.name + ".extraction.json")
            credit = 0
            if partial.exists() and receipt.exists() and read_json(receipt).get("header") == item["archive"]:
                credit = min(item["extraction_bytes"], allocated(partial))
            request(target, item["extraction_bytes"] - credit, scratch_keep_free_gib, f"{item['dataset']}: HDF5")
        if include_cache:
            needed = item["cache"]["total_bytes"] - cache_credit(config, item)
            request(config["paths"]["cache_dir"], needed, keep_free_gib, f"{item['dataset']}: cache")
            request(config["paths"].get("experiment_dir", config["paths"]["cache_dir"]), METADATA_ALLOWANCE,
                    keep_free_gib, f"{item['dataset']}: metadata/canonical view")
    result = list(volumes.values())
    for group in result:
        group["sufficient"] = group["available_bytes"] >= group["additional_bytes"] + group["reserve_bytes"]
    return result


def require_space(volumes):
    for group in volumes:
        if not group["sufficient"]:
            raise ValueError(f"Insufficient space on {group['filesystem_example']}: "
                f"{group['additional_bytes']/GIB:.2f} GiB additional + {group['reserve_bytes']/GIB:.1f} GiB reserve "
                f"> {group['available_bytes']/GIB:.2f} GiB available. "
                "Select fewer --datasets, or use --cache-root / --scratch-root on a larger filesystem.")
