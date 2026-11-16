"""Model-free source/checkpoint/query contracts shared by P2 and P3."""

from pathlib import Path

from ..runtime import read_json


def validate_evaluation_config(config):
    from ..cache_loading import cache_only
    cache_only(config)
    p1 = config["p1"]
    task = p1["data"].get("dataset_id")
    if p1["data"].get("memory_only") and not config.get("query_dataset"):
        raise ValueError("Memory-only P1 profiles require independent query_dataset episodes for P2/P3")
    if config.get("evaluation_dataset") and config["evaluation_dataset"] != task:
        raise ValueError("Selected evaluation dataset differs from the P1 memory task")
    width = p1.get("preprocessing", {}).get("raw_action_dim")
    if width is None:
        return  # Keep historical PushT protocols and synthetic fixtures unchanged.
    checkpoint = read_json(Path(p1["paths"]["checkpoint"])/"config.json")
    if (checkpoint["predictor"]["num_frames"] != p1["data"]["history"]
            or checkpoint["action_encoder"]["input_dim"] != width*p1["data"]["frameskip"]):
        raise ValueError("P1 checkpoint history/action-block width differs from this task")
    if "planning" in config and any(len(config["planning"][key]) != width for key in ("raw_action_low", "raw_action_high")):
        raise ValueError(f"P3 raw action bounds must have {width} coordinates for {task}; select the correct --dataset")


def validate_query_task(config, memory_source, query):
    """Reject a same-shaped but unrelated task before encoding or adaptation."""
    p1 = config["p1"]
    task = p1["data"].get("dataset_id")
    source_task = memory_source["data_config"].get("dataset_id")
    if task != source_task:
        raise ValueError("P1 memory task differs from its prepared source manifest")
    if task is not None and query.get("backend") != task:
        raise ValueError(f"Fresh query backend {query.get('backend')!r} differs from P1 memory task {task!r}")
    # Older PushT exports lack a backend field. New non-PushT exports must never
    # be used with the old untagged PushT source profile.
    if task is None and query.get("backend") not in (None, "pusht"):
        raise ValueError("Native-task fresh queries cannot use PushT memory/checkpoint")
    width = query["source"]["schema"]["action"]["shape"][-1]
    expected = memory_source["source"]["schema"]["action"]["shape"][-1]
    if width != expected:
        raise ValueError("Fresh and memory datasets have different raw action widths")
