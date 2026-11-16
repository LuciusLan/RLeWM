"""P3-a offline configuration and provenance; no changes to earlier stages."""

from pathlib import Path
import math

from ..config import ROOT, load_config as load_p1, validate_config as validate_p1
from ..runtime import read_json
from ..rollout.config import effective_p1, SCOPES
from ..rollout.record_io import record_mode

STAGE = "P3a-offline-cem"
METHODS = ("frozen", "uniform", "forward", "reverse", "fr-fixed")
LIMITS = [
    "Independent planning calls from recorded contexts; no execution or simulator reset.",
    "Goal is the recorded observation H blocks ahead; no intermediate future observation enters planning.",
    "Only the recorded action sequence has observed outcomes. Other plan costs are model predictions.",
    "Action agreement is descriptive, not success, coverage, regret, or true candidate ranking.",
    "Frozen rescoring measures model disagreement; the Frozen predictor is not an oracle.",
    "Test episodes were inspected in earlier stages; this is not a fresh episode holdout.",
    "Cold-start calls do not measure MPC warm starts, replanning, or matched-wall-clock performance.",
]


def evaluation_limits(config):
    limits = list(LIMITS)
    if config.get("query_dataset"):
        limits[5] = "Fresh simulator queries; exact released-source state audit passed; unknown additional pretraining data cannot be audited."
        if config["p1"]["data"].get("dataset_id") == "tworooms":
            limits[5] = "Fresh simulator queries; complete-trajectory/source duplicate audit passed. TwoRoom 2D point collisions are diagnostic; unknown pretraining data cannot be audited."
    return limits


def load_config(path: Path) -> dict:
    config = read_json(path)
    for key in ("p1_config", "experiment_dir"):
        config[key] = str((ROOT/config[key]).resolve())
    config["p1"] = load_p1(Path(config["p1_config"]))
    config["adaptation"] = effective_p1(config)["adaptation"]
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    record_mode(config)
    p1 = effective_p1(config)
    validate_p1(p1)
    from ..collection.contracts import validate_evaluation_config
    validate_evaluation_config(config)
    if p1["data"]["memory_windows"] != "all":
        raise ValueError("P3-a requires the existing full observation cache")
    methods = p1["adaptation"]["methods"]
    if "frozen" not in methods or set(methods)-set(METHODS):
        raise ValueError(f"Include Frozen and choose methods from {METHODS}")
    if len(config["scopes"]) != 1 or config["scopes"][0] not in SCOPES:
        raise ValueError("Use exactly one adaptation scope per P3-a comparison")
    a, p, e = p1["adaptation"], config["planning"], config["evaluation"]
    for key in ("horizon", "population", "iterations", "elites", "scoring_batch_size",
                "provisional_actions", "support_scan_limit"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"planning.{key} must be a positive integer")
    if not 2 <= p["elites"] <= p["population"]:
        raise ValueError("Require 2 <= elites <= population (sample standard deviation)")
    if type(a["updates"]) is not int or type(a["support_size"]) is not int:
        raise ValueError("Support and update budgets must be integers")
    # At least one slot for every action/direction; a deliberate balanced pilot.
    channels = 2 if "fr-fixed" in methods and 0 < a["alpha"] < 1 else 1
    if a["support_size"] < channels*p["provisional_actions"]:
        raise ValueError("Support budget must cover each provisional action/direction")
    if p["support_scan_limit"] < a["support_size"]:
        raise ValueError("support_scan_limit must cover the unique support budget")
    for key in ("initial_std", "minimum_std"):
        if not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"planning.{key} must be finite and nonnegative")
    if p["initial_std"] <= 0 or p["minimum_std"] > p["initial_std"]:
        raise ValueError("Require 0 <= minimum_std <= initial_std, initial_std > 0")
    low, high = p["raw_action_low"], p["raw_action_high"]
    if (not low or len(low) != len(high) or any(
            not math.isfinite(x) or not math.isfinite(y) or x >= y for x, y in zip(low, high))):
        raise ValueError("Provide matching finite raw action bounds with low < high")
    for key in ("learning_rate", "clip_norm", "anchor_weight", "alpha"):
        if not math.isfinite(a[key]):
            raise ValueError(f"adaptation.{key} must be finite")
    for key in ("bootstrap_samples", "log_every"):
        if type(e[key]) is not int or e[key] < 0:
            raise ValueError(f"evaluation.{key} must be a nonnegative integer")
    if e["query_limit"] is not None and (type(e["query_limit"]) is not int or e["query_limit"] < 1):
        raise ValueError("query_limit must be null or a positive integer")
    for key in ("windows_per_episode", "seed"):
        value = config["query_selection"][key]
        if type(value) is not int or value < (1 if key == "windows_per_episode" else 0):
            raise ValueError(f"Invalid query_selection.{key}")
    if any(type(seed) is not int for seed in p1["evaluation"]["seeds"]):
        raise ValueError("Evaluation seeds must be integers")
    if Path(config["experiment_dir"]) == Path(p1["paths"]["experiment_dir"]):
        raise ValueError("Use a separate P3 experiment directory")


def code_identity() -> dict:
    from experiments.provenance import backend_identity
    return backend_identity("p3a")
