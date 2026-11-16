"""Bind shared P2/P3 protocols to a dataset without copying stage configs."""

from copy import deepcopy

from lewm_tta.runtime import read_json
from .catalog import ROOT


DATASETS = ("pusht", "tworooms", "reacher", "cube")


def dataset_profile(name):
    if name not in DATASETS or name == "pusht":
        raise ValueError("Native task profiles are tworooms, reacher and cube")
    value = read_json(ROOT/f"codes/configs/datasets/{name}.json")
    if value["dataset_id"] != name:
        raise ValueError("Dataset profile ID differs from its name")
    return value


def experiment_kind(config):
    if "core" in config:
        return "p3-core"
    if "planning" not in config and "horizons" in config:
        return "p2"
    return {"factorial": "p3", "proposal-only": "p3-proposals",
            "temporal-planning": "p3-temporal", "interleaved-planning": "p3-interleave"}.get(config.get("comparison"))


def bind_dataset(config, name):
    """Dataset selection changes sources/bounds/output, retaining scientific knobs."""
    if name not in DATASETS:
        raise ValueError(f"Unknown evaluation dataset: {name}")
    from .config import read_profile
    if not config.get("query_dataset") or experiment_kind(config) is None:
        raise ValueError("--dataset requires a fresh P2/P3 profile; use --variant fresh, fresh-temporal or fresh-proposals")
    if name == "pusht":
        p1 = read_profile(ROOT/config["p1_config"])
        if p1["data"].get("dataset_id", "pusht") != name:
            raise ValueError("The profile does not use PushT memory")
        return deepcopy(config)
    task = dataset_profile(name)
    value = deepcopy(config)
    value["evaluation_dataset"] = name
    value["p1_config"] = task["p1_config"]
    if value.get("comparison") == "interleaved-planning":
        rates = value["interleave"].get("dataset_learning_rates", {})
        if name in rates:
            value["adaptation"]["learning_rate"] = rates[name]
        seeds = value["interleave"].get("dataset_seeds", {})
        if name in seeds:
            value["evaluation"]["seeds"] = list(seeds[name])
            if "core" in value:
                value["evaluation"]["seeds"] = value["evaluation"]["seeds"][:1]
    value["experiment_dir"] = str(ROOT/task["experiment_root"]/experiment_kind(value))
    collection_config = value.get("core", {}).get("collection_by_dataset", {}).get(name, task["collection_config"])
    value["query_dataset"] = {"collection_config": collection_config, "regime": "source",
                              "batch_size": config["query_dataset"].get("batch_size", 128)}
    # Never import PushT reports or its latent cache into a new task.
    if "reuse" in value:
        value["reuse"] = {}
    if "planning" in value:
        value["planning"].update(raw_action_low=task["raw_action_low"], raw_action_high=task["raw_action_high"])
    collection = read_profile(ROOT/collection_config)
    p1 = read_profile(ROOT/task["p1_config"])
    if (collection["backend"] != name or p1["data"].get("dataset_id") != name
            or (ROOT/collection["reference_dataset"]).resolve() != (ROOT/p1["paths"]["dataset"]).resolve()
            or any(collection[key] != p1["data"][key] for key in ("history", "frameskip"))):
        raise ValueError("Dataset profile must bind matching source-memory and fresh-collection profiles")
    return value
