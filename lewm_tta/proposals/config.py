"""Proposal-only and factorial experiment settings share the same primitives."""

import math
from pathlib import Path

from ..config import ROOT
from ..planning.config import load_config as load_planning, validate_config as validate_planning

SOURCES = ("gaussian", "uniform-fragments", "context-fragments", "goal-context-fragments")


def goal_settings(config):
    return {"goal_weight": 1.0, "goal_calibration_pairs": 8192,
            "goal_calibration_seed": 20261026, "goal_distance_floor": 1e-8, **config["proposals"]}


def stage_name(config):
    return "P3b-offline-proposals" if config["comparison"] == "proposal-only" else "P3c-offline-factorial"


def validate_config(config):
    validate_planning(config)
    from .reuse import validate_reuse
    validate_reuse(config)
    if config["comparison"] not in ("proposal-only", "factorial"):
        raise ValueError("comparison must be proposal-only or factorial")
    if config["comparison"] == "proposal-only" and config["adaptation"]["methods"] != ["frozen"]:
        raise ValueError("P3b proposal-only uses Frozen; a later factorial config can include shared TTT")
    if config["comparison"] == "factorial" and len(config["adaptation"]["methods"]) < 2:
        raise ValueError("Factorial comparison requires Frozen plus at least one TTT method")
    settings = config["proposals"]
    sources = settings["sources"]
    if not sources or len(set(sources)) != len(sources) or set(sources)-set(SOURCES) or "gaussian" not in sources:
        raise ValueError(f"Include gaussian and choose distinct proposal sources from {SOURCES}")
    if config["comparison"] == "factorial" and len(sources) < 2:
        raise ValueError("Factorial comparison requires Gaussian plus at least one fragment source")
    for key in ("fragments", "max_fragments_per_episode", "max_scan_candidates", "context_chunk_rows"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"proposals.{key} must be a positive integer")
    for key in ("injection_slots", "min_start_step_gap"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"proposals.{key} must be a nonnegative integer")
    slots = settings["injection_slots"]
    if slots >= config["planning"]["population"] or (slots and slots < settings["fragments"]):
        raise ValueError("Injection needs at least one slot per fragment and must preserve the mean slot; zero disables injection")
    if settings["max_scan_candidates"] < settings["fragments"]:
        raise ValueError("max_scan_candidates must cover the requested fragments")
    if not math.isfinite(settings["noise_std"]) or settings["noise_std"] < 0:
        raise ValueError("Proposal noise_std must be finite and nonnegative")
    goal = goal_settings(config)
    if not math.isfinite(goal["goal_weight"]) or goal["goal_weight"] < 0:
        raise ValueError("Goal weight must be finite and nonnegative")
    if not math.isfinite(goal["goal_distance_floor"]) or goal["goal_distance_floor"] <= 0:
        raise ValueError("Goal distance floor must be finite and positive")
    for key, minimum in (("goal_calibration_pairs", 1), ("goal_calibration_seed", 0)):
        if type(goal[key]) is not int or goal[key] < minimum:
            raise ValueError(f"proposals.{key} must be an integer >= {minimum}")
    for existing in ("p3a-offline-q4", "p2", "p25-q4", "p2-followup-q4"):
        if Path(config["experiment_dir"]) == ROOT/"artifacts"/existing:
            raise ValueError("Use a separate proposal experiment directory")
    other = "p3b-offline-q4" if config["comparison"] == "factorial" else "p3c-offline-q4"
    if Path(config["experiment_dir"]) == ROOT/"artifacts"/other:
        raise ValueError("P3b and P3c require separate experiment directories")


def load_config(path):
    config = load_planning(path)
    from .reuse import normalize_reuse
    normalize_reuse(config)
    validate_config(config)
    return config


def code_identity():
    from experiments.provenance import backend_identity
    return backend_identity("p3b")
