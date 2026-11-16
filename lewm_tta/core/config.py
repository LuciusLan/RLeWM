"""Four-cell core design, with the historical six-cell comparison available."""

from copy import deepcopy
from math import isfinite

from ..interleave.config import validate_config as validate_interleave

STAGE = "P3-core-planning"
CELLS = {
    ("gaussian", "baseline"): "gaussian/frozen",
    ("gaussian", "rca"): "gaussian/checkpoint",
    ("retrieval", "baseline"): "init/frozen",
    ("retrieval", "rca"): "init/checkpoint",
    ("retrieval", "proposals"): "init-mid/frozen",
    ("retrieval", "combined"): "init-mid/checkpoint",
}


def mid_search_enabled(config):
    # Saved six-cell configs predate this flag; never reinterpret their matrix.
    enabled = config["core"].get("mid_search", True)
    if type(enabled) is not bool:
        raise ValueError("core.mid_search must be a boolean")
    return enabled


def cells(config):
    return {key: label for key, label in CELLS.items()
            if mid_search_enabled(config) or key[1] not in ("proposals", "combined")}


def groups(config):
    result = {}
    selected = cells(config)
    for name in ("gaussian", "retrieval"):
        arms = [arm for group, arm in selected if group == name]
        value = deepcopy(config)
        value["interleave"]["arms"] = arms
        value["interleave"]["initial_source"] = "gaussian" if name == "gaussian" else config["core"]["initial_source"]
        if not mid_search_enabled(config):
            value["interleave"]["injection_slots"] = 0
        result[name] = value
    return result


def primary_endpoint(config):
    return "final" if config["core"]["control_budget"] == 25 else "ever"


def validate(config):
    validate_interleave(config)
    if not config.get("query_dataset") or "core" not in config:
        raise ValueError("Core comparison requires fresh queries and core settings")
    settings, planning = config["core"], config["planning"]
    seeds = config["p1"]["evaluation"]["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Core planner seeds must be distinct nonnegative integers")
    budgets = settings["iterations"]
    if (not budgets or budgets != sorted(set(budgets)) or any(type(j) is not int or j < 1 for j in budgets)
            or budgets[-1] != planning["iterations"]):
        raise ValueError("Core stopping iterations must be sorted, distinct, positive and end at maximum J")
    if not config["interleave"]["checkpoints"]:
        raise ValueError("Core comparison needs at least one adaptation checkpoint")
    if settings["control_budget"] not in (25, 50) or planning["horizon"]*config["p1"]["data"]["frameskip"] != 25:
        raise ValueError("Core supports H*frameskip=25 and --control-budget 25 or 50")
    if settings["initial_source"] not in ("context-fragments", "goal-context-fragments"):
        raise ValueError("Core initialization must retrieve action fragments")
    if not config["proposals"]["injection_slots"]:
        raise ValueError("Core requires an active retrieval initialization budget")
    if mid_search_enabled(config) and not config["interleave"]["injection_slots"]:
        raise ValueError("Six-cell core requires an active mid-search injection budget")
    if config["adaptation"]["updates"] < 1 or config["interleave"]["support_method"] == "uniform":
        raise ValueError("Core checkpoint adaptation must use retrieved supervision and positive updates")
    if settings["runtime_budget_hours"] <= 0 or not isfinite(settings["runtime_budget_hours"]):
        raise ValueError("Core runtime budget must be positive and finite")
    if settings["pusht_success"] not in ("native-goal", "coverage") or not 0 < settings["coverage_threshold"] <= 1:
        raise ValueError("Invalid PushT physical success criterion")
    if any(not isfinite(settings[key]) or settings[key] < 0 for key in ("state_atol", "rgb_mae_atol")):
        raise ValueError("Reconstruction tolerances must be finite and nonnegative")
    if (type(settings["bootstrap_samples"]) is not int or settings["bootstrap_samples"] < 0
            or type(settings["bootstrap_seed"]) is not int or settings["bootstrap_seed"] < 0
            or any(not isfinite(x) or not 0 < x <= 1 for x in settings["success_targets"])
            or any(not isfinite(x) or x <= 0 for x in settings["deadlines"])):
        raise ValueError("Invalid confidence/target/deadline settings")
    for value in groups(config).values():
        validate_interleave(value)
