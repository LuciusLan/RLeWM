"""Explicit settings for the additive, offline P3 interleave treatment."""

import math
from pathlib import Path

from ..planning.config import load_config as load_planning, validate_config as validate_planning
from ..runtime import read_json

STAGE = "P3-interleaved-cem"
ARMS = {"baseline": (False, False), "rca": (True, False),
        "proposals": (False, True), "combined": (True, True)}
SUPPORT_METHODS = ("uniform", "forward", "reverse", "fr-fixed", "fr-refresh")
ADAPTATION_SCHEDULES = ("pooled-depth", "sequential-depth")


def adaptation_schedule(config):
    # Absence retains the meaning of every historical resolved config.
    return config["interleave"].get("adaptation_schedule", "pooled-depth")


def checkpoint_updates(config, updates=None):
    """Requested gradient steps per checkpoint, rather than per depth."""
    updates = config["adaptation"]["updates"] if updates is None else updates
    positions = len(config["interleave"]["adaptation_depths"]) if adaptation_schedule(config) == "sequential-depth" else 1
    return positions*updates


def validate_depth_updates(record, config, updates=None):
    """Verify the sequential schedule from receipts without running a model."""
    if adaptation_schedule(config) != "sequential-depth":
        return
    stages = record.get("depths", [])
    if not stages:
        if record["actual_updates"] or (record["requested_updates"] and not record["skip_reason"]):
            raise ValueError("Sequential checkpoint is missing its depth audit")
        return
    updates = config["adaptation"]["updates"] if updates is None else updates
    if ([stage["depth"] for stage in stages] != config["interleave"]["adaptation_depths"]
            or record["requested_updates"] != checkpoint_updates(config, updates)):
        raise ValueError("Sequential checkpoint depth/update budget differs")
    version, presentations = record["model_version_before"], 0
    flattened = []
    for stage in stages:
        rows = [row for row in record["rounds"] if row.get("depth") == stage["depth"]]
        expected = list(range(version+1, version+len(rows)+1))
        if (stage["requested_updates"] != updates or stage["actual_updates"] != len(rows) or len(rows) > updates
                or stage["model_version_before"] != version
                or stage["reference_prefix_model_version"] != version
                or stage["reference_prefix_blocks"] != stage["depth"]
                or stage["update_indices"] != expected
                or [row["update"] for row in rows] != expected
                or [row["model_version"] for row in rows] != expected
                or any(pair["depth"] != stage["depth"] for pair in stage["adaptation_pairs"])
                or (len(rows) < updates and not stage["skip_reason"])):
            raise ValueError("Sequential depth audit has stale prefixes or invalid optimizer updates")
        version += len(rows)
        size = sum(len(row["support"]) for row in rows)
        if stage["model_version_after"] != version or stage["support_presentations"] != size:
            raise ValueError("Sequential depth version/support counts differ")
        presentations += size
        flattened.extend(rows)
    if (flattened != record["rounds"] or record["model_version_after"] != version
            or record["actual_updates"] != len(flattened)
            or record["support_presentations"] != presentations):
        raise ValueError("Sequential checkpoint accounting differs from its depth audit")


def validate_config(config):
    validate_planning(config)
    if config.get("comparison") != "interleaved-planning":
        raise ValueError("Interleave requires comparison=interleaved-planning")
    if config.get("reuse"):
        raise ValueError("Interleave never imports fixed-model P3 results")
    settings, planning = config["interleave"], config["planning"]
    if adaptation_schedule(config) not in ADAPTATION_SCHEDULES:
        raise ValueError("Unknown checkpoint adaptation schedule")
    arms = settings["arms"]
    if not arms or len(arms) != len(set(arms)) or set(arms)-set(ARMS) or "baseline" not in arms:
        raise ValueError("Choose distinct interleave arms including baseline")
    seeds = config["p1"]["evaluation"]["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Choose distinct nonempty evaluation seeds")
    checkpoints = settings["checkpoints"]
    if (not isinstance(checkpoints, list) or any(type(j) is not int or not 1 <= j < planning["iterations"] for j in checkpoints)
            or checkpoints != sorted(set(checkpoints))):
        raise ValueError("Checkpoints must be sorted unique completed-iteration counts, leaving another iteration")
    for key in ("adaptation_depths", "continuation_depths"):
        values = settings[key]
        minimum = 0 if key == "adaptation_depths" else 1
        if (not values or any(type(h) is not int or not minimum <= h < planning["horizon"] for h in values)
                or values != sorted(set(values))):
            raise ValueError(f"interleave.{key} must contain sorted distinct valid depths")
    for key in ("references", "fragments_per_pair"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"interleave.{key} must be positive")
    if settings["references"] > planning["elites"]:
        raise ValueError("Reference count cannot exceed the elite budget")
    if settings["support_method"] not in SUPPORT_METHODS or settings["reference_source"] not in ("elite", "random"):
        raise ValueError("Unknown support method or reference source")
    if settings["continuation_context"] not in ("imagined", "observed"):
        raise ValueError("continuation_context must be imagined or observed (matched reinjection control)")
    if settings["initial_source"] not in ("gaussian", "context-fragments", "uniform-fragments", "goal-context-fragments"):
        raise ValueError("Unknown initial proposal source")
    alpha = config["adaptation"]["alpha"]
    channels = 2 if settings["support_method"].startswith("fr-") and 0 < alpha < 1 else 1
    pairs = settings["support_pairs"]
    if pairs is not None and (type(pairs) is not int or not 1 <= pairs <= config["adaptation"]["support_size"]//channels):
        raise ValueError("support_pairs must leave at least one support slot per pair/direction, or be null")
    for key in ("injection_slots",):
        if type(settings[key]) is not int or not 0 <= settings[key] < planning["population"]:
            raise ValueError("Injection slots must preserve population slot zero")
    for key in ("noise_std", "goal_weight"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"interleave.{key} must be finite and nonnegative")
    p = config["proposals"]
    for key in ("fragments", "max_fragments_per_episode", "max_scan_candidates", "context_chunk_rows",
                "goal_calibration_pairs"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"proposals.{key} must be positive")
    for key in ("min_start_step_gap", "goal_calibration_seed", "injection_slots"):
        if type(p[key]) is not int or p[key] < 0:
            raise ValueError(f"proposals.{key} must be nonnegative")
    if p["injection_slots"] >= planning["population"] or (p["injection_slots"] and p["injection_slots"] < p["fragments"]):
        raise ValueError("Initial injection needs one slot per fragment and must preserve the mean")
    for key in ("noise_std", "goal_weight", "goal_distance_floor"):
        if not math.isfinite(p[key]) or p[key] < (0 if key != "goal_distance_floor" else 1e-30):
            raise ValueError(f"Invalid proposals.{key}")
    replay = settings["replay"]
    if replay["policy"] not in ("current", "current-plus-replay"):
        raise ValueError("Unknown replay policy")
    if type(replay["capacity"]) is not int or replay["capacity"] < 1:
        raise ValueError("Replay capacity must be positive")
    if not math.isfinite(replay["fraction"]) or not 0 <= replay["fraction"] < 1:
        raise ValueError("Replay fraction must be in [0,1)")
    if replay["policy"] == "current-plus-replay" and replay["capacity"] < config["adaptation"]["support_size"]:
        raise ValueError("Replay capacity must hold at least one complete support batch")
    # An existing stage manifest can never be overwritten by this variant.
    manifest = Path(config["experiment_dir"])/"manifest.json"
    if manifest.is_file() and read_json(manifest).get("stage") != STAGE:
        raise ValueError("Choose a separate interleave experiment directory; existing P3 artifacts are immutable")


def load_config(path):
    config = load_planning(path)
    if "seeds" in config["evaluation"]:
        config["p1"]["evaluation"]["seeds"] = list(config["evaluation"]["seeds"])
    validate_config(config)
    return config


def code_identity():
    from experiments.provenance import backend_identity
    return backend_identity("p3_interleave")
