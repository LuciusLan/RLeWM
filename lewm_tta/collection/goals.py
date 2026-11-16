"""Observed-future goal protocols, independent of collection behavior and MPC."""

from pathlib import Path

from ..runtime import write_json
from .manifest import checked, sealed


def protocols(config):
    return config.get("goal_protocols", {"default": {"goal_offset_steps": config["goal_offset_steps"]}})


def max_offset(config):
    return max(item["goal_offset_steps"] for item in protocols(config).values())


def available_starts(config):
    return config["raw_steps"]+1-(config["history"]-1)*config["frameskip"]-max_offset(config)


def validate_protocols(config):
    values = protocols(config)
    if not values or not any(item["goal_offset_steps"] == config["goal_offset_steps"] for item in values.values()):
        raise ValueError("A declared goal protocol must match the primary query goal offset")
    for name, item in values.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Goal protocol names must be simple names")
        if set(item)-{"goal_offset_steps", "eval_budget", "source"}:
            raise ValueError("Unsupported goal protocol option")
        if type(item["goal_offset_steps"]) is not int or item["goal_offset_steps"] < 1:
            raise ValueError("Goal offsets count positive native outer environment steps")
        if "eval_budget" in item and (type(item["eval_budget"]) is not int or item["eval_budget"] < 1):
            raise ValueError("MPC execution budgets must be positive")


def export_catalog(config, query):
    """Rows refer to actual stored frames/states; never goals rendered by teleporting."""
    if "goal_protocols" not in config:
        return
    from ..simulation.environments import fresh_environment_class
    adapter = fresh_environment_class(config["backend"])
    offsets = query["episode_metadata"]["offsets"]
    rows = config["raw_steps"]+1
    cases = []
    for split, selection in query["evaluation_windows"].items():
        for start in selection["raw_start_rows"]:
            episode = int(start)//rows
            current = start+(config["history"]-1)*config["frameskip"]
            cases.append({"case_id": f"{query['cases'][episode]['case_id']}:window:{start-offsets[episode]}",
                          "split": split, "episode_id": query["episode_metadata"]["ids"][episode],
                          "history_start_row": start, "current_row": current,
                          "goal_rows": {name: current+item["goal_offset_steps"] for name, item in protocols(config).items()}})
    path = Path(query["source"]["path"]).with_name("goal-protocols.json")
    value = sealed({
        "version": 1, "query_manifest": query["fingerprint"], "dataset": query["source"]["path"],
        "backend": config["backend"], "protocols": protocols(config), "cases": cases,
        "native_goal": adapter.goal_contract(),
        "note": "Primary queries.json uses goal_offset_steps; alternative catalog goals require a runner that explicitly consumes this catalog. Native task targets are not substituted for observed future RGB."})
    if path.exists():
        if checked(path) != value:
            raise ValueError("Exported goal catalog changed; refusing to overwrite it")
        return
    write_json(path, value)
