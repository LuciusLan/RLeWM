"""Read individual committed episodes, including before full-cohort export."""

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from ..collection.config import load_config
from ..collection.manifest import checked, load_manifest
from ..collection.storage import episode_paths
from ..runtime import file_hash, object_hash, write_json
from .media import Panel, index_page, save_gallery


def resolved_config(path, loader):
    """Honor inherited profiles without changing/materializing experiment assets."""
    import json
    import tempfile
    from experiments.config import resolve
    with tempfile.TemporaryDirectory(prefix="lewm-preview-config-") as temporary:
        resolved = Path(temporary)/"config.json"
        resolved.write_text(json.dumps(resolve(path)))
        return loader(resolved)


def load_episode(config_path, split, episode, regime, start_step=None, window_index=None, goal_protocol=None):
    config = resolved_config(Path(config_path), load_config)
    manifest = load_manifest(config)
    if regime not in config["regimes"]:
        raise ValueError(f"Unknown collection regime: {regime}")
    matched = [case for case in manifest["cases"] if case["split"] == split and case["index"] == episode]
    if len(matched) != 1:
        raise ValueError("Episode index is outside the configured split")
    case = matched[0]
    path, metadata_path = episode_paths(config, regime, case)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Collect this episode first: {path}")
    metadata = checked(metadata_path)
    if (metadata["case"] != case or metadata["regime"] != regime
            or metadata["collection_manifest"] != manifest["fingerprint"] or file_hash(path) != metadata["sha256"]):
        raise ValueError("Episode metadata/content does not match the collection")
    from ..collection.goals import available_starts, protocols
    available = available_starts(config)
    goal_offset = config["goal_offset_steps"]
    if goal_protocol is not None:
        if goal_protocol not in protocols(config):
            raise ValueError(f"Unknown goal protocol: {goal_protocol}")
        goal_offset = protocols(config)[goal_protocol]["goal_offset_steps"]
    seed = int(object_hash([case["case_id"], "windows"])[:16], 16)
    starts = (metadata["quality"]["selected_start_steps"] if config.get("quality") else
              np.sort(np.random.default_rng(seed).choice(available, config["windows_per_episode"], replace=False)).tolist())
    if start_step is None:
        window_index = 0 if window_index is None else window_index
        if not 0 <= window_index < len(starts):
            raise ValueError("window-index is outside this episode's reserved query windows")
        start_step = int(starts[window_index])
    elif window_index is not None:
        raise ValueError("Choose start-step or window-index, not both")
    if not 0 <= start_step < available:
        raise ValueError(f"start-step must lie in [0,{available-1}] for a complete history/H5 future")
    current = start_step+(config["history"]-1)*config["frameskip"]
    goal = current+goal_offset
    return {"config": config, "manifest": manifest, "case": case, "metadata": metadata, "path": path,
            "regime": regime, "start": start_step, "current": current, "goal": goal,
            "reserved_starts": starts, "fps": float(metadata["environment"]["control_hz"])}


def output_directory(path, mode):
    from ..config import ROOT
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = ROOT/"artifacts/previews"/f"{mode}-{stamp}"
    else:
        path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def preview(config_path, split, episodes, regime, output=None, *, start_step=None, window_index=None, every=1, scale=2, goal_protocol=None):
    if not episodes or len(set(episodes)) != len(episodes) or any(index < 0 for index in episodes):
        raise ValueError("Choose distinct nonnegative episode indices")
    # Validate selected source assets before allocating an output directory.
    loaded = [load_episode(config_path, split, index, regime, start_step, window_index, goal_protocol) for index in episodes]
    directory = output_directory(output, "collection")
    galleries, records = [], []
    for item in loaded:
        episode = item["case"]["index"]
        name = f"{split}-{episode:05d}"
        with h5py.File(item["path"], "r") as source:
            frames = list(source["pixels"][:])
            states = source["state64"][:] if "state64" in source else source["state"][:]
            actions = source["action"][:-1]
        whole = save_gallery(directory, name,
            [Panel("Collected exploration", frames, list(range(len(frames))))],
            fps=item["fps"], every=every, scale=scale)
        current, goal = item["current"], item["goal"]
        query = save_gallery(directory, name+"-query", [
            Panel(f"Recorded +{goal-current} control continuation", frames[current:goal+1], list(range(current, goal+1))),
            Panel("Supplied future goal", [frames[goal]], [goal], fixed=True)],
            fps=item["fps"], every=every, scale=scale)
        from ..simulation.environments import fresh_environment_class
        adapter = fresh_environment_class(item["config"]["backend"])
        if hasattr(adapter, "goal_diagnostics_rows"):
            with h5py.File(item["path"], "r") as source:
                diagnostic = adapter.goal_diagnostics_rows(source, current, goal)
        else:
            diagnostic = adapter.goal_diagnostics(states[current], states[goal], item["config"]["regimes"][regime])
        record = {"episode": episode, "split": split, "case_id": item["case"]["case_id"],
                  "collection_manifest": item["manifest"]["fingerprint"], "source": str(item["path"]),
                  "source_sha256": item["metadata"]["sha256"], "regime": regime,
                  "policy": item["config"]["policy"], "reserved_query_start_steps": item["reserved_starts"],
                  "history_start_step": item["start"], "current_step": current, "goal_step": goal, "goal_protocol": goal_protocol,
                  "history_steps": [item["start"]+i*item["config"]["frameskip"] for i in range(item["config"]["history"])],
                  "goal_diagnostics": diagnostic, "raw_action_min": actions.min(0).tolist(),
                  "behavior_quality": item["metadata"].get("quality"),
                  "accepted_attempt": item["metadata"].get("accepted_attempt", 0),
                  "raw_action_max": actions.max(0).tolist(), "whole_episode": whole, "example_query": query}
        write_json(directory/(name+".json"), record)
        records.append(record)
        galleries.extend([(name+": full exploratory trajectory", whole), (name+": one fixed observed-future query", query)])
    description = ("Stored simulator frames from scripted exploration, not CEM plans. "
                   f"Each query fixes its supplied goal to the observed frame {loaded[0]['goal']-loaded[0]['current']} raw actions ahead. "
                   "The native scene target may differ from that separate goal. No model/simulator rerun.")
    index_page(directory, "Fresh episode validity previews", galleries, description)
    write_json(directory/"preview.json", {"mode": "collection-frames", "episodes": records,
                                          "description": description, "benchmark_result": False})
    print(f"Previews: {directory/'index.html'}")
    return directory
