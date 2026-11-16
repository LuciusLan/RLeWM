"""Reuse P3's episode-safe tasks and caches, with a distinct experiment identity."""

from pathlib import Path

from ...rollout.resources import checked_json
from ...runtime import object_hash, write_json
from ..cohorts import manifest_value as planning_manifest
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
    if path.exists():
        if checked_json(path) != value:
            raise ValueError("Temporal planning cohort changed; use a separate experiment directory")
    else:
        write_json(path, value)
    print(f"Manifest: {path}; reuses existing P1 observations and cuVS assets.", flush=True)
    for split, cohort in value["splits"].items():
        print(f"{split}: {cohort['queries']} contexts / {cohort['episodes']} episodes", flush=True)
    return value


def load_manifest(config):
    value = checked_json(Path(config["experiment_dir"])/"manifest.json")
    if value != manifest_value(config):
        raise ValueError("Temporal planning cohort/config mismatch; run compatible prepare")
    return value
