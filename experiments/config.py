"""Resolve inherited profiles into immutable, standalone engine inputs."""

from copy import deepcopy
from pathlib import Path

from lewm_tta.runtime import object_hash, read_json, write_json
from .catalog import ROOT


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def migrated_path(path):
    path = Path(path)
    if path.is_file():
        return path.resolve()
    # Historical configs refer to the former flat config directory.
    if path.parent.resolve() == (ROOT / "codes/configs").resolve():
        legacy = ROOT / "codes/legacy/configs" / path.name
        if legacy.is_file():
            return legacy
    raise FileNotFoundError(path)


def read_profile(path, stack=()):
    path = migrated_path(Path(path).resolve())
    if path in stack:
        raise ValueError("Configuration inheritance contains a cycle")
    current = read_json(path)
    parent = current.pop("extends", None)
    base = read_profile(path.parent / parent, (*stack, path)) if parent else {}
    return merge(base, current)


def resolve(path, *, dataset=None, cache_only=False):
    value = read_profile(path)
    if dataset is not None:
        from .datasets import bind_dataset
        value = bind_dataset(value, dataset)
    for key in ("p1_config", "p3_config"):
        if key in value:
            target = Path(value[key])
            value[key] = str(migrated_path(target if target.is_absolute() else ROOT / target))
    if "query_dataset" in value:
        value["query_dataset"] = query_dataset_settings(value["query_dataset"])
    if cache_only:
        value["cache_only"] = True
    return value


def query_dataset_settings(settings):
    """Freeze a collection profile's current paths into the engine input/lock."""
    if "collection_config" not in settings:
        return deepcopy(settings)
    if set(settings)-{"collection_config", "regime", "cache_dir", "batch_size"}:
        raise ValueError("Collection-bound query settings accept collection_config, regime, cache_dir and batch_size")
    collection = read_profile(ROOT / settings["collection_config"])
    regime = settings.get("regime", "source")
    if regime not in collection["regimes"]:
        raise ValueError(f"Unknown collection regime: {regime}")
    directory = (ROOT / collection["output_dir"] / regime).resolve()
    return {"manifest": str(directory / "queries.json"),
            "cache_dir": str((ROOT / settings.get("cache_dir", directory / "cache")).resolve()),
            "batch_size": settings.get("batch_size", 128)}


def materialize(value):
    """Engine loaders consume full JSON; this cache never changes an old file."""
    path = ROOT / "artifacts/experiment-configs" / f"{object_hash(value)}.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Resolved configuration checksum mismatch: {path}")
    else:
        write_json(path, value)
    return path
