"""Model-free checks for fresh native navigation, control and manipulation data."""

import hashlib
from pathlib import Path

import h5py
import numpy as np

from experiments.progress import progress
from ..runtime import file_hash, write_json
from .goals import protocols
from .manifest import checked, sealed


def trajectory_key(states, actions):
    digest = hashlib.sha256()
    for value in (states, actions):
        canonical = np.array(value, dtype="<f4", order="C", copy=True)
        canonical[canonical == 0] = 0
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def goal_diagnostics(adapter, stream, current, goal, shift):
    if hasattr(adapter, "goal_diagnostics_rows"):
        return adapter.goal_diagnostics_rows(stream, current, goal)
    return adapter.goal_diagnostics(stream["state64"][current], stream["state64"][goal], shift)


def summarize_goals(goals):
    return {"queries": len(goals), "initially_successful": sum(item["initial_success"] for item in goals),
            "metrics": {key: {"mean": float(np.mean([item[key] for item in goals])),
                              "median": float(np.median([item[key] for item in goals]))}
                        for key in goals[0] if key != "initial_success"},
            "note": "diagnostics only; all prespecified episodes/windows retained, including low-motion and initially successful goals"}


def check_native(config, manifest, adapter, regimes=None):
    from .pipeline import state_keys
    regimes = regimes or list(config["regimes"])
    if set(regimes)-set(config["regimes"]):
        raise ValueError("Unknown collection regime")
    import hdf5plugin  # noqa: F401 -- native source compression
    reference = Path(config["reference_dataset"])
    with h5py.File(reference, "r") as source:
        from ..preprocessing.hdf5 import view_record
        view_record(source)
        states = adapter.reference_state(source)
        if states.shape[1] != adapter.collection_schema()["state_width"] or not np.isfinite(states).all():
            raise ValueError("Reference physical state columns do not match this task")
        offsets, lengths = source["ep_offset"][:], source["ep_len"][:]
        metadata_column = "episode_idx" if "episode_idx" in source else "ep_idx"
        original_ids = set(source[metadata_column][offsets.tolist()].tolist())
        reference_initial = set(item.tobytes() for item in state_keys(states[offsets]))
        reference_keys = np.unique(state_keys(states))
        reference_trajectories = set()
        for offset, length in zip(offsets, lengths):
            if length == config["raw_steps"]+1:
                reference_trajectories.add(trajectory_key(states[offset:offset+length], source["action"][offset:offset+length-1]))
    del states
    audits = []
    for regime in regimes:
        query = checked(Path(config["output_dir"])/regime/"queries.json")
        if (query["collection_manifest"] != manifest["fingerprint"]
                or file_hash(Path(query["source"]["path"])) != query["dataset_sha256"]):
            raise ValueError("Collection or exported virtual dataset identity changed")
        overlap = initial_overlap = duplicate_episodes = count = 0
        seen_initial, seen_episodes = set(), set()
        for record in progress(query["episode_files"], config=config, desc=f"Audit {regime}", unit="episode"):
            if (record["case"]["episode_id"] in original_ids or file_hash(Path(record["path"])) != record["sha256"]):
                raise ValueError("Episode namespace collision or changed payload")
            if config["policy"]["name"] == "sac":
                for key in ("checkpoint", "normalizer"):
                    asset = record["behavior_assets"][key]
                    if asset["sha256"] != file_hash(Path(asset["path"])):
                        raise ValueError("SAC collection assets changed")
            with h5py.File(record["path"], "r") as stream:
                state, actions = stream["state"][:], stream["action"][:]
                schema = adapter.collection_schema()
                if (state.shape != (config["raw_steps"]+1, schema["state_width"])
                        or actions.shape != (len(state), schema["action_width"])
                        or not np.isfinite(state).all() or not np.isfinite(actions).all()
                        or (np.abs(actions[:-1]) > 1+1e-6).any() or np.any(actions[-1] != 0)
                        or not stream["action_valid"][:-1].all() or stream["action_valid"][-1]):
                    raise ValueError("Invalid native state/action alignment or final action sentinel")
                if not np.array_equal(stream["state64"][:].astype(np.float32), state):
                    raise ValueError("Float64 replay state disagrees with cached float32 state")
                for name, shape in schema["fields"].items():
                    value = stream[name][:]
                    if value.shape != (len(state), *shape) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid native replay field: {name}")
                if "physics_time" in stream and not (np.diff(stream["physics_time"][:]) > 0).all():
                    raise ValueError("Simulator clock did not advance monotonically; possible implicit reset")
                initial = state_keys(state[:1])[0].tobytes()
                if initial in seen_initial:
                    raise ValueError("Duplicate initial physical states within/across validation and test")
                seen_initial.add(initial)
                initial_overlap += int(initial in reference_initial)
                key = trajectory_key(state, actions[:-1])
                duplicate_episodes += int(key in reference_trajectories or key in seen_episodes)
                seen_episodes.add(key)
                keys = state_keys(state)
                positions = np.searchsorted(reference_keys, keys)
                valid = positions < len(reference_keys)
                overlap += int((reference_keys[positions[valid]] == keys[valid]).sum())
                count += len(state)
        with h5py.File(query["source"]["path"], "r") as stream:
            goals = {name: {split: summarize_goals([
                goal_diagnostics(adapter, stream, start+(config["history"]-1)*config["frameskip"],
                    start+(config["history"]-1)*config["frameskip"]+item["goal_offset_steps"], config["regimes"][regime])
                for start in selection["raw_start_rows"]]) for split, selection in query["evaluation_windows"].items()}
                for name, item in protocols(config).items()}
        # A matching 2D navigation point can occur naturally (wall clipping),
        # unlike an identical full physical trajectory and action sequence.
        passed = duplicate_episodes == 0 and (config["backend"] == "tworooms" or initial_overlap == 0)
        audit = sealed({"passed": passed, "collection_manifest": manifest["fingerprint"],
            "query_manifest": query["fingerprint"], "regime": regime, "backend": config["backend"],
            "source_dataset": str(reference.resolve()), "source_size": reference.stat().st_size,
            "source_mtime_ns": reference.stat().st_mtime_ns, "collected_states": count,
            "exact_released_state_matches": overlap, "exact_released_initial_state_matches": initial_overlap,
            "exact_duplicate_complete_episodes": duplicate_episodes, "goal_protocol_diagnostics": goals,
            "split_rule": "prespecified disjoint reset/action seeds; no evaluated-model success filtering",
            "scope": "float32 exact physical states and complete equal-length state/action trajectories against the released source; TwoRoom point collisions are descriptive only; no perceptual-near-duplicate or unknown-pretraining guarantee"})
        path = Path(config["output_dir"])/regime/"audit.json"
        write_json(path, audit)
        audits.append(audit)
        print(f"{regime}: {count:,} observations; {duplicate_episodes} duplicate episodes. Audit: {path}")
        if not passed:
            raise ValueError("Native fresh/source duplicate audit failed; inspect audit.json")
    return audits
