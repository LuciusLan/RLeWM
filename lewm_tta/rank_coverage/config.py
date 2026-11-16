"""Coverage protocol and immutable prepared-data metadata; no model imports."""

from __future__ import annotations

import math
from pathlib import Path

from ..config import ROOT, load_config as load_p1
from ..runtime import file_hash, object_hash, read_json, write_json


MODES = ("recorded-action", "provisional-actions")


def load_config(path: Path) -> dict:
    value = read_json(path)
    for key in ("p1_config", "p3_config", "experiment_dir"):
        value[key] = str((ROOT / value[key]).resolve())
    value["p1"] = load_p1(Path(value["p1_config"]))
    source = read_json(Path(value["p3_config"]))
    fields = ("provisional_actions", "initial_std", "support_scan_limit", "raw_action_low", "raw_action_high")
    value["provisional"] = {key: source["planning"][key] for key in fields}
    value["p3_config_sha256"] = file_hash(Path(value["p3_config"]))
    value.setdefault("support_size", value["p1"]["adaptation"]["support_size"])
    value.setdefault("alpha", value["p1"]["adaptation"]["alpha"])
    validate_config(value)
    return value


def validate_config(value: dict) -> None:
    for key in ("percentages", "prefix_sizes", "query_modes", "seeds"):
        if not value[key] or len(set(value[key])) != len(value[key]):
            raise ValueError(f"{key} must contain distinct values")
    if value["percentages"] != sorted(value["percentages"]) or 100 not in value["percentages"] or any(
            not math.isfinite(p) or not 0 < p <= 100 for p in value["percentages"]):
        raise ValueError("Use sorted percent values in (0,100], including the 100% baseline")
    if value["prefix_sizes"] != sorted(value["prefix_sizes"]) or any(type(k) is not int or k < 1 for k in value["prefix_sizes"]):
        raise ValueError("prefix_sizes must be sorted positive integers")
    if set(value["query_modes"]) - set(MODES) or any(type(seed) is not int or seed < 0 for seed in value["seeds"]):
        raise ValueError("Invalid query mode or provisional-action seeds")
    if type(value["support_size"]) is not int or value["support_size"] < 2 or not 0 < value["alpha"] < 1:
        raise ValueError("This bidirectional pilot requires B>=2 and 0<alpha<1")
    selection = value["query_selection"]
    if any(type(selection[k]) is not int or selection[k] < 1 for k in ("windows_per_episode", "cohort_horizon")) or selection["seed"] < 0:
        raise ValueError("Invalid horizon-safe query selection")
    if value["p1"]["data"]["memory_windows"] != "all":
        raise ValueError("The pilot requires prepared full-memory source assets")
    if Path(value["experiment_dir"]) == Path(value["p1"]["paths"]["experiment_dir"]):
        raise ValueError("Use a separate coverage experiment directory")
    evaluation = value["evaluation"]
    if any(type(evaluation[k]) is not int or evaluation[k] < 0 for k in ("bootstrap_samples", "bootstrap_seed", "log_every")):
        raise ValueError("Invalid statistics settings")
    if type(evaluation["cpu_threads"]) is not int or evaluation["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    planning = value["provisional"]
    if any(type(planning[k]) is not int or planning[k] < 1 for k in ("provisional_actions", "support_scan_limit")):
        raise ValueError("Invalid provisional-action count or scan limit")
    if not math.isfinite(planning["initial_std"]) or planning["initial_std"] <= 0:
        raise ValueError("Provisional action std must be positive and finite")


def checked_json(path: Path) -> dict:
    value = read_json(path)
    if value.get("fingerprint") != object_hash({key: item for key, item in value.items() if key != "fingerprint"}):
        raise ValueError(f"Metadata checksum mismatch: {path}")
    return value


def source_metadata(value: dict) -> tuple[dict, dict]:
    directory = Path(value["p1"]["paths"]["experiment_dir"])
    source, prepared = checked_json(directory / "manifest.json"), checked_json(directory / "prepared.json")
    if source["version"] != 2 or source["data_config"] != value["p1"]["data"] or prepared["manifest"] != source["fingerprint"]:
        raise ValueError("P1 config, source manifest and prepared assets must match")
    return source, prepared


def cohort_spec(value: dict, source: dict, prepared: dict) -> dict:
    return {"p1_manifest_fingerprint": source["fingerprint"], "p1_prepared_fingerprint": prepared["fingerprint"],
            "query_selection": value["query_selection"]}


def prepare(value: dict) -> dict:
    # Existing selector is shared verbatim with P2.5/P3; no model is instantiated.
    from ..rollout.cohorts import sample_cohort
    source, prepared = source_metadata(value)
    selection = value["query_selection"]
    manifest = {"stage": "rank-coverage", "version": 1, "specification": cohort_spec(value, source, prepared),
                "splits": {name: sample_cohort(source, name, selection["cohort_horizon"], selection)
                           for name in ("validation", "test")},
                "targets_scored": False, "parameter_updates": 0}
    manifest["fingerprint"] = object_hash(manifest)
    path = Path(value["experiment_dir"]) / "manifest.json"
    if path.exists():
        if checked_json(path) != manifest:
            raise ValueError("Coverage cohort changed; choose a new experiment directory")
    else:
        write_json(path, manifest)
    print("Coverage cohort:", {name: (row["queries"], row["episodes"]) for name, row in manifest["splits"].items()}, flush=True)
    print("Prepared metadata only:", path, flush=True)
    return manifest


def code_identity() -> dict:
    from experiments.provenance import backend_identity
    return backend_identity("rank_coverage")
