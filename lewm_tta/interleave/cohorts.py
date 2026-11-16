"""Same episode-safe tasks, separate stage manifest and output namespace."""

from pathlib import Path

from ..planning.cohorts import manifest_value as planning_manifest
from ..rollout.resources import checked_json
from ..runtime import object_hash, write_json
from .config import STAGE, validate_config


def manifest_value(config):
    value = planning_manifest(config)
    value.pop("fingerprint")
    value["stage"] = STAGE
    value["fingerprint"] = object_hash(value)
    return value


def prepare(config):
    validate_config(config)
    value = manifest_value(config)
    path = Path(config["experiment_dir"])/"manifest.json"
    if path.exists() and checked_json(path) != value:
        raise ValueError("Interleave cohort changed; choose a new experiment directory")
    if not path.exists():
        write_json(path, value)
    for split, cohort in value["splits"].items():
        print(f"{STAGE} {split}: {cohort['queries']} queries / {cohort['episodes']} episodes", flush=True)
    print(f"Manifest: {path}", flush=True)
    return value


def load_manifest(config):
    value = checked_json(Path(config["experiment_dir"])/"manifest.json")
    if value != manifest_value(config):
        raise ValueError("Interleave manifest/config/source mismatch; run compatible prepare")
    return value
