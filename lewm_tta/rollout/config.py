"""Shared rollout settings for original, scope and iteration comparisons."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from ..config import ROOT, load_config as load_p1, validate_config as validate_p1
from ..runtime import read_json
from .record_io import record_mode


P2_STAGE = "P2-recorded-rollout"
SCOPE_STAGE = "P2.5-scope-comparison"
SCHEDULE_STAGE = "P2-followup-iteration"
SCOPES = ("last-block", "predictor", "predictor-action")


def is_scope_comparison(config: dict) -> bool:
    return "scopes" in config


def is_schedule_comparison(config: dict) -> bool:
    return "schedules" in config


def schedule_names(config: dict) -> list:
    return config["schedule_names"] if is_schedule_comparison(config) else [None]


def updates_at(schedule: dict, position: int) -> int:
    return schedule["updates"] if position == 0 or schedule["placement"] == "each-step" else 0


def update_budget(schedule: dict, horizon: int) -> int:
    return sum(updates_at(schedule, position) for position in range(horizon))


def settings_for_schedule(settings: dict, schedule: dict | None = None) -> dict:
    """Return an independent settings mapping; an explicit schedule LR wins."""
    result = dict(settings)
    rate = schedule.get("learning_rate", settings["learning_rate"]) if schedule is not None else settings["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Schedule learning_rate must be a finite positive number")
    result["learning_rate"] = float(rate)
    return result


def cohort_horizon(config: dict) -> int:
    return config.get("cohort_horizon", max(config["horizons"]))


def scope_names(config: dict) -> list[str]:
    return config.get("scopes", ["last-block"])


def stage_name(config: dict) -> str:
    if is_schedule_comparison(config):
        return SCHEDULE_STAGE
    return SCOPE_STAGE if is_scope_comparison(config) else P2_STAGE


def entrypoint(config: dict) -> str:
    if is_schedule_comparison(config):
        return "p2_followup"
    return "p25" if is_scope_comparison(config) else "p2"


def load_config(path: Path) -> dict:
    config = read_json(path)
    for key in ("p1_config", "experiment_dir"):
        config[key] = str((ROOT/config[key]).resolve())
    config["p1"] = load_p1(Path(config["p1_config"]))
    validate_config(config)
    return config


def effective_p1(config: dict) -> dict:
    result = deepcopy(config["p1"])
    result["adaptation"].update(config["adaptation"])
    result["evaluation"].update({key: config["evaluation"][key]
                                for key in ("bootstrap_samples", "log_every")})
    return result


def validate_config(config: dict) -> None:
    record_mode(config)
    p1 = effective_p1(config)
    validate_p1(p1)
    from ..collection.contracts import validate_evaluation_config
    validate_evaluation_config(config)
    if p1["data"]["memory_windows"] != "all":
        raise ValueError("Rollout evaluation requires the prepared full observation cache")
    if "frozen" not in p1["adaptation"]["methods"]:
        raise ValueError("Frozen is required for paired rollout comparisons")
    if type(p1["adaptation"]["updates"]) is not int:
        raise ValueError("adaptation.updates must be an integer")
    if not math.isfinite(p1["adaptation"]["learning_rate"]):
        raise ValueError("adaptation.learning_rate must be finite")
    horizons = config["horizons"]
    if (not horizons or horizons != sorted(set(horizons)) or horizons[0] != 1
            or any(type(h) is not int or h < 1 for h in horizons)):
        raise ValueError("horizons must be distinct increasing positive integers including 1")
    for key in ("bootstrap_samples", "log_every"):
        if type(config["evaluation"][key]) is not int or config["evaluation"][key] < 0:
            raise ValueError(f"evaluation.{key} must be a nonnegative integer")
    limit = config["evaluation"]["query_limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("query_limit must be null or a positive integer")
    if Path(config["experiment_dir"]) == Path(config["p1"]["paths"]["experiment_dir"]):
        raise ValueError("Rollouts must use a separate experiment directory from the P1 asset source")
    if is_schedule_comparison(config):
        if not is_scope_comparison(config):
            raise ValueError("Iteration comparisons require explicit scopes and q4 query_selection")
        if type(cohort_horizon(config)) is not int or cohort_horizon(config) < max(3, max(horizons)):
            raise ValueError("cohort_horizon must cover the evaluation and H3 acceptance fixtures")
        catalog, selected = config["schedules"], config["schedule_names"]
        if (not isinstance(catalog, dict) or not catalog or not isinstance(selected, list)
                or not selected or any(not isinstance(name, str) for name in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("Provide a schedule catalog and distinct nonempty schedule_names")
        if set(selected)-set(catalog):
            raise ValueError("Unknown selected schedule")
        for name, schedule in catalog.items():
            if (not isinstance(name, str) or not name or not isinstance(schedule, dict)
                    or not {"placement", "updates", "persistence"} <= set(schedule)
                    or set(schedule) - {"placement", "updates", "persistence", "learning_rate"}):
                raise ValueError("Each schedule requires placement, updates and persistence; learning_rate is optional")
            settings_for_schedule(p1["adaptation"], schedule)
            if schedule["placement"] not in ("initial", "each-step"):
                raise ValueError("Schedule placement must be initial or each-step")
            if type(schedule["updates"]) is not int or schedule["updates"] < 0:
                raise ValueError("Schedule updates must be a nonnegative integer")
            if schedule["persistence"] not in ("trajectory", "position"):
                raise ValueError("Schedule persistence must be trajectory or position")
            if schedule["placement"] == "initial" and schedule["persistence"] != "trajectory":
                raise ValueError("Initial-only schedules must preserve their adapted model for the trajectory")
        if "updates" in config["adaptation"]:
            raise ValueError("For follow-up experiments, put update counts in schedules, not adaptation.updates")
        if Path(config["experiment_dir"]) in (ROOT/"artifacts/p2", ROOT/"artifacts/p25-q4"):
            raise ValueError("The follow-up must preserve the existing P2/P2.5 directories")
    if is_scope_comparison(config):
        scopes = config["scopes"]
        if not scopes or len(set(scopes)) != len(scopes) or set(scopes)-set(SCOPES):
            raise ValueError(f"Choose distinct scopes from {SCOPES}")
        selection = config["query_selection"]
        if type(selection["windows_per_episode"]) is not int or selection["windows_per_episode"] < 1:
            raise ValueError("query_selection.windows_per_episode must be positive")
        if type(selection["seed"]) is not int or selection["seed"] < 0:
            raise ValueError("query_selection.seed must be a nonnegative integer")
        if Path(config["experiment_dir"]) == ROOT/"artifacts/p2":
            raise ValueError("Scope comparisons must preserve the existing P2 experiment directory")


def manifest_spec(config: dict) -> dict:
    # Preserve the existing P2/P2.5 manifests and cache source identities.
    if config.get("query_dataset") and is_scope_comparison(config):
        return {"p1_experiment_dir": config["p1"]["paths"]["experiment_dir"],
                "horizons": config["horizons"], "cohort_horizon": cohort_horizon(config),
                "query_selection": config["query_selection"], "selection_version": 2,
                "selection": "reuse the fresh collection's prespecified starts; no stage resampling"}
    if is_schedule_comparison(config):
        return {"p1_experiment_dir": config["p1"]["paths"]["experiment_dir"],
                "horizons": config["horizons"], "cohort_horizon": cohort_horizon(config),
                "query_selection": config["query_selection"], "selection_version": 1,
                "selection": "sample at cohort_horizon, independently of schedule and evaluated horizon"}
    if is_scope_comparison(config):
        return {"p1_experiment_dir": config["p1"]["paths"]["experiment_dir"],
                "horizons": config["horizons"], "query_selection": config["query_selection"],
                "selection_version": 1,
                "selection": "sample distinct maximum-horizon-safe starts per source episode"}
    return {"horizons": config["horizons"],
            "p1_experiment_dir": config["p1"]["paths"]["experiment_dir"]}
