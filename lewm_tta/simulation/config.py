"""Simulator protocol settings; native model/retrieval configuration stays shared."""

import math
from pathlib import Path

from ..config import ROOT
from ..proposals.config import load_config as load_proposals, validate_config as validate_proposals

STAGE = "simulator-evaluation"


def load_config(path):
    config = load_proposals(path)
    if config["simulation"].get("source_report"):
        config["simulation"]["source_report"] = str((ROOT/config["simulation"]["source_report"]).resolve())
    validate_config(config)
    return config


def validate_config(config):
    validate_proposals(config)
    settings = config["simulation"]
    task = settings.get("task", "future-goal")
    if task not in ("future-goal", "green-target"):
        raise ValueError("simulation.task must be future-goal or green-target")
    from .environments import environment_class
    environment_class(settings["backend"]).validate_config(settings, config["planning"])
    for key in ("episodes", "image_size", "max_raw_steps", "execute_blocks"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"simulation.{key} must be a positive integer")
    if type(settings["cohort_seed"]) is not int or settings["cohort_seed"] < 0:
        raise ValueError("simulation.cohort_seed must be a nonnegative integer")
    if not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]) or any(type(x) is not int or x < 0 for x in settings["seeds"]):
        raise ValueError("Provide distinct nonnegative simulator seeds")
    if config["evaluation"]["records"] not in ("compact", "none") or config["evaluation"]["call_records"] not in ("compact", "none"):
        raise ValueError("Simulator records/call_records support compact or none")
    frameskip = config["p1"]["data"]["frameskip"]
    green = task == "green-target"
    fresh = bool(config.get("query_dataset")) or green
    if (settings["max_raw_steps"] % frameskip or settings["execute_blocks"] > config["planning"]["horizon"]
            or (not fresh and settings["execute_blocks"] != 1)):
        raise ValueError("Execution must be block-aligned and within the planning horizon; historical pilot uses one block")
    if fresh and settings.get("source_report"):
        raise ValueError("Fresh cases must not import old source-corpus plans/reports")
    if green:
        if config["evaluation"]["records"] != "compact":
            raise ValueError("Green validation/test requires compact episode records for paired analysis and locking; call records may be disabled")
        if config.get("query_dataset") or settings["backend"] != "pusht":
            raise ValueError("Green PushT uses independent native reset cases, without query_dataset")
        if type(settings.get("test_episodes")) is not int or settings["test_episodes"] < 1:
            raise ValueError("Provide positive simulation.test_episodes")
        if settings.get("split", "validation") not in ("validation", "test"):
            raise ValueError("Green split must be validation or test")
        coverage = settings.get("success_coverage")
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 < coverage <= 1:
            raise ValueError("success_coverage must lie in (0,1]")
        agent = settings.get("goal_agent_position")
        if (not isinstance(agent, list) or len(agent) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 32 <= x <= 480 for x in agent)):
            raise ValueError("goal_agent_position requires two native-scene coordinates in [32,480]")
        from .environments.fresh_pusht import validate_shift
        validate_shift(settings.get("shift"))
        if config["p1"]["data"]["history"] != 3 or frameskip != 5:
            raise ValueError("Released PushT green benchmark uses H3 and frameskip5")
    if config["scopes"] != ["last-block"] or config["adaptation"]["methods"] != ["frozen", "uniform", "fr-fixed"]:
        raise ValueError("The pilot fixes last-block scope and the three prespecified model rows")
    if config["proposals"]["sources"] != ["gaussian", "context-fragments"]:
        raise ValueError("The pilot fixes Gaussian/context-fragment proposal columns")
    if config.get("reuse"):
        raise ValueError("Simulator replay reads source plans; MPC must not import offline model forecasts or searches")
    if settings["runtime_budget_hours"] is not None and (isinstance(settings["runtime_budget_hours"], bool)
            or not isinstance(settings["runtime_budget_hours"], (int, float))
            or not math.isfinite(settings["runtime_budget_hours"]) or settings["runtime_budget_hours"] <= 0):
        raise ValueError("runtime_budget_hours must be null or positive")
    if settings.get("source_report") and Path(config["experiment_dir"]) == Path(settings["source_report"]).parent.parent.parent:
        raise ValueError("Keep simulator artifacts separate from completed offline results")
