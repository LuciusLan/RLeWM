"""Reuse the cached planning task definition without changing P3-a manifests."""

from pathlib import Path

from ..planning.cohorts import manifest_value as planning_manifest
from ..rollout.resources import checked_json
from ..runtime import object_hash, write_json
from .config import stage_name, validate_config


def manifest_value(config):
    value = planning_manifest(config)
    value.pop("fingerprint")
    value["stage"] = stage_name(config)
    value["fingerprint"] = object_hash(value)
    return value


def prepare(config):
    validate_config(config)
    value = manifest_value(config)
    path = Path(config["experiment_dir"])/"manifest.json"
    if path.exists():
        if checked_json(path) != value:
            raise ValueError("Proposal cohort changed; choose a new experiment directory")
    else:
        write_json(path, value)
    for split, cohort in value["splits"].items():
        print(f"{stage_name(config)} {split}: {cohort['queries']} queries / {cohort['episodes']} episodes", flush=True)
    print(f"Manifest: {path}\nUses existing observed actions/latents and context columns of cuVS keys; no new index payload.", flush=True)
    return value


def load_manifest(config):
    value = checked_json(Path(config["experiment_dir"])/"manifest.json")
    if value != manifest_value(config):
        raise ValueError("Proposal manifest/config/source mismatch; run compatible prepare")
    return value
