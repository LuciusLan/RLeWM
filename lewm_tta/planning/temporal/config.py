"""Pure configuration and nominal budgets for the expensive temporal ablation."""

from pathlib import Path

from ..config import load_config as load_planning, validate_config as validate_planning


STAGE = "P3-temporal-cem"
MODES = ("frozen", "shared-initial", "shared-depth", "candidate-temporal")
SELECTORS = ("uniform", "forward", "reverse", "fr-fixed")


def valid_seeds(values):
    return (isinstance(values, list) and bool(values)
            and all(type(seed) is int and seed >= 0 for seed in values)
            and len(set(values)) == len(values))


def seeds_for_mode(config: dict, mode: str) -> list[int]:
    if mode not in MODES:
        raise ValueError("Unknown temporal mode")
    evaluation = config["evaluation"]
    fallback = evaluation.get("seeds", config["p1"]["evaluation"]["seeds"])
    return list(evaluation.get("seeds_by_mode", {}).get(mode, fallback))


def seed_schedule(config: dict) -> list[tuple[int, list[str]]]:
    """Run Frozen first for each used seed, followed by that seed's active arms."""
    modes = ["frozen"]+[mode for mode in config["temporal"]["modes"] if mode != "frozen"]
    return [(seed, active) for seed in config["p1"]["evaluation"]["seeds"]
            if (active := [mode for mode in modes if seed in seeds_for_mode(config, mode)])]


def override_seeds(config: dict, seeds: list[int]) -> None:
    """An explicit CLI --seeds selects the same seeds for every active mode."""
    if not valid_seeds(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    config["evaluation"]["seeds"] = list(seeds)
    config["p1"]["evaluation"]["seeds"] = list(seeds)
    config["evaluation"].pop("seeds_by_mode", None)


def load_config(path: Path) -> dict:
    config = load_planning(path)
    if "seeds" in config["evaluation"]:
        if not valid_seeds(config["evaluation"]["seeds"]):
            raise ValueError("evaluation.seeds must be distinct nonnegative integers")
        config["p1"]["evaluation"]["seeds"] = list(config["evaluation"]["seeds"])
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    evaluation = config["evaluation"]
    global_seeds = evaluation.get("seeds", config["p1"]["evaluation"]["seeds"])
    if not valid_seeds(global_seeds):
        raise ValueError("evaluation.seeds must be distinct nonnegative integers")
    validate_planning(config)
    if config.get("comparison") != "temporal-planning":
        raise ValueError("Require comparison=temporal-planning")
    settings = config["temporal"]
    if set(settings) != {"modes", "selector", "persistence"}:
        raise ValueError("temporal requires modes, selector and persistence")
    modes = settings["modes"]
    if not isinstance(modes, list) or not modes or any(not isinstance(mode, str) for mode in modes):
        raise ValueError("temporal.modes must be a nonempty list")
    if len(set(modes)) != len(modes) or set(modes)-set(MODES) or "frozen" not in modes:
        raise ValueError(f"Include frozen and choose distinct temporal modes from {MODES}")
    if settings["selector"] not in SELECTORS:
        raise ValueError(f"temporal.selector must be one of {SELECTORS}")
    if settings["selector"] not in config["adaptation"]["methods"]:
        raise ValueError("Include temporal.selector in adaptation.methods")
    if settings["persistence"] not in ("trajectory", "position"):
        raise ValueError("temporal.persistence must be trajectory or position")
    if "seeds" in config["evaluation"] and config["evaluation"]["seeds"] != config["p1"]["evaluation"]["seeds"]:
        raise ValueError("Resolved temporal seeds must agree with the effective P1 evaluation seeds")
    overrides = evaluation.get("seeds_by_mode", {})
    if not isinstance(overrides, dict) or set(overrides)-set(MODES):
        raise ValueError(f"evaluation.seeds_by_mode must map known modes from {MODES} to seed lists")
    for mode, seeds in overrides.items():
        if not valid_seeds(seeds) or set(seeds)-set(global_seeds):
            raise ValueError(f"evaluation.seeds_by_mode.{mode} must be a nonempty distinct subset of evaluation.seeds")
    frozen_seeds = set(seeds_for_mode(config, "frozen"))
    if any(set(seeds_for_mode(config, mode))-frozen_seeds for mode in modes):
        raise ValueError("Frozen seeds must cover every selected mode's seeds for paired comparisons")


def nominal_budget(config: dict, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError("Unknown temporal mode")
    planning = config["planning"]
    sequences = planning["population"]*planning["iterations"]+1
    phases = {"frozen": 0, "shared-initial": 1, "shared-depth": planning["horizon"],
              "candidate-temporal": sequences*planning["horizon"]}[mode]
    updates = phases*config["adaptation"]["updates"]
    return {"cem_sequence_evaluations": sequences,
            "cem_transition_predictions": sequences*planning["horizon"],
            "adaptation_positions": phases, "requested_gradient_updates": updates,
            "requested_support_presentations": updates*config["adaptation"]["support_size"],
            "summed_learning_rate": updates*config["adaptation"]["learning_rate"]}


def code_identity():
    from experiments.provenance import backend_identity
    return backend_identity("p3_temporal")
