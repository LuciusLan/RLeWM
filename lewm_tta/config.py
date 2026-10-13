"""Resolved P1 configuration; experimental settings stay out of model code."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .runtime import read_json


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("frozen", "uniform", "state", "forward", "reverse", "fr-fixed", "fr-refresh", "rank-consensus")


def load_config(path: Path) -> dict:
    def merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base
    def read_inherited(filename, stack):
        filename = Path(filename).resolve()
        if filename in stack:
            raise ValueError("Configuration inheritance contains a cycle")
        current = read_json(filename)
        parent = current.pop("extends", None)
        base = read_inherited(filename.parent / parent, stack + [filename]) if parent else {}
        return merge(base, current)
    config = read_inherited(path, [])
    for key, value in config["paths"].items():
        config["paths"][key] = str((ROOT / value).resolve()) if not Path(value).is_absolute() else value
    config["data"]["inspected_reports"] = [
        str((ROOT / path).resolve()) if not Path(path).is_absolute() else path
        for path in config["data"]["inspected_reports"]
    ]
    config["data"]["inspected_report_globs"] = [
        str(ROOT / path) if not Path(path).is_absolute() else path
        for path in config["data"].get("inspected_report_globs", [])
    ]
    if config["data"].get("split_from"):
        config["data"]["split_from"] = str((ROOT / config["data"]["split_from"]).resolve())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    data, adaptation, retrieval = config["data"], config["adaptation"], config["retrieval"]
    for key in ("history", "frameskip", "validation_episodes", "test_episodes"):
        if data[key] < 1:
            raise ValueError(f"data.{key} must be positive")
    if data["memory_windows"] != "all" and (not isinstance(data["memory_windows"], int) or data["memory_windows"] < 0):
        raise ValueError("data.memory_windows must be a nonnegative integer or 'all'")
    if data["windows_per_query_episode"] != "all" and (
        not isinstance(data["windows_per_query_episode"], int) or data["windows_per_query_episode"] < 1):
        raise ValueError("data.windows_per_query_episode must be positive or 'all'")
    if data["windows_per_query_episode"] == "all" and data["memory_windows"] != "all":
        raise ValueError("All-query evaluation requires the full-cache profile")
    if data["split_seed"] < 0:
        raise ValueError("Memory size and split seed must be nonnegative")
    if config.get("cache", {}).get("checkpoint_rows", 8192) < 1:
        raise ValueError("cache.checkpoint_rows must be positive")
    methods = adaptation["methods"]
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"Choose distinct methods from {METHODS}")
    if adaptation["updates"] < 0 or adaptation["support_size"] < 1 or adaptation["learning_rate"] <= 0:
        raise ValueError("Invalid gradient/support/learning-rate budget")
    if any(method.startswith("fr-") for method in methods) and adaptation["support_size"] < 2:
        raise ValueError("Mixed-channel methods require a support budget of at least two")
    if not 0 <= adaptation["alpha"] <= 1 or adaptation["anchor_weight"] < 0 or adaptation["clip_norm"] <= 0:
        raise ValueError("Invalid channel mixture, parameter anchor, or clipping norm")
    if retrieval["pool_size"] < adaptation["support_size"]:
        raise ValueError("Retrieval pool_size must cover the support budget")
    if min(retrieval["beta_action"], retrieval["beta_successor"]) < 0 or retrieval["distance_floor"] <= 0:
        raise ValueError("Invalid distance coefficients or floor")
    if min(retrieval["calibration_pairs"], retrieval["chunk_size"]) < 1:
        raise ValueError("Calibration and search batch sizes must be positive")
    cap = retrieval.get("max_windows_per_episode")
    if cap is not None and (not isinstance(cap, int) or cap < 1):
        raise ValueError("retrieval.max_windows_per_episode must be positive or null")
    if not isinstance(retrieval.get("min_start_step_gap", 0), int) or retrieval.get("min_start_step_gap", 0) < 0:
        raise ValueError("retrieval.min_start_step_gap must be a nonnegative integer")
    evaluation = config["evaluation"]
    if not evaluation["seeds"] or min(evaluation["seeds"]) < 0 or len(set(evaluation["seeds"])) != len(evaluation["seeds"]):
        raise ValueError("Evaluation seeds must be distinct and nonnegative")
    if evaluation["bootstrap_samples"] < 0 or evaluation["harm_tolerance"] < 0 or evaluation["log_every"] < 0:
        raise ValueError("Invalid evaluation settings")
    if not 0 < evaluation["accurate_quantile"] < 1:
        raise ValueError("accurate_quantile must lie strictly between zero and one")
