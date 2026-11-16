"""Resumable CPU collection and model-free provenance/duplicate audits."""

import fcntl
from pathlib import Path

import h5py
import numpy as np

from experiments.progress import progress
from experiments.provenance import backend_identity
from ..runtime import file_hash, write_json
from .manifest import checked, load_manifest, sealed
from .storage import EpisodeWriter, episode_paths, export_regime


def collect(config, split="all", regimes=None, limit=None):
    from ..simulation.environments import make_collection_environment
    manifest = load_manifest(config)
    wanted = [case for case in manifest["cases"] if split == "all" or case["split"] == split]
    if limit is not None:
        wanted = wanted[:limit]
    regimes = regimes or list(config["regimes"])
    if not regimes or set(regimes)-set(config["regimes"]):
        raise ValueError("Choose configured regime names")
    implementation = backend_identity("collect")
    lock_path = Path(config["output_dir"])/"collect.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another collector owns this output directory") from error
        for regime in regimes:
            environment = make_collection_environment(config, regime)
            try:
                description = environment.description()
                for case in progress(wanted, config=config, desc=f"Collect {regime}/{split}", unit="episode"):
                    path, metadata = episode_paths(config, regime, case)
                    if metadata.exists():
                        previous = checked(metadata)
                        if (previous["case"] != case or previous["collection_manifest"] != manifest["fingerprint"]
                                or previous["environment"] != description or previous["code"] != implementation
                                or previous["sha256"] != file_hash(path)):
                            raise ValueError("Committed episode/source changed; use a new output directory")
                        continue
                    details = {}
                    if config.get("quality"):
                        from .attempts import collect_quality_episode
                        attempt, attempts = collect_quality_episode(environment, config, manifest, regime, case, path, implementation)
                        details = {"execution": attempt["execution"], "quality": attempt["quality"],
                                   "accepted_attempt": attempt["attempt"], "attempts": attempts}
                    else:
                        observation = environment.reset(case["reset_seed"])
                        rng = np.random.default_rng(case["action_seed"])
                        writer = EpisodeWriter(path, config, case, environment.collection_schema())
                        try:
                            writer.append(observation)
                            for step in range(config["raw_steps"]):
                                action = environment.collection_action(observation, rng, step, config["policy"])
                                observation = environment.step(action)
                                if observation.truncated:
                                    raise ValueError("Collector truncated before the prespecified horizon")
                                writer.append(observation, action)
                            writer.commit()
                        finally:
                            writer.close()
                    record = sealed({"collection_manifest": manifest["fingerprint"], "regime": regime,
                        "case": case, "path": str(path.resolve()), "sha256": file_hash(path),
                        "raw_steps": config["raw_steps"], "observations": config["raw_steps"]+1,
                        "environment": description, "code": implementation,
                        "outcome_filtering": False, "behavior_quality_filtering": bool(config.get("quality")),
                        **({"behavior_assets": environment.behavior_assets()} if hasattr(environment, "behavior_assets") else {}), **details})
                    write_json(metadata, record)
            finally:
                environment.close()
            export_regime(config, manifest, regime)


def state_keys(states):
    """An explicit exact float32 state test, not a nearest-neighbour novelty claim."""
    value = np.array(states, dtype=np.dtype("<f4"), order="C", copy=True)
    value[value == 0] = 0  # Canonicalize signed zeros before byte comparison.
    return value.view(np.dtype((np.void, value.dtype.itemsize*value.shape[1]))).ravel()


def check(config, regimes=None):
    """Read/check only. No simulator stepping, encoder, predictor or adaptation."""
    manifest = load_manifest(config)
    from ..simulation.environments import fresh_environment_class
    adapter = fresh_environment_class(config["backend"])
    if config["backend"] != "pusht":
        from .native_audit import check_native
        return check_native(config, manifest, adapter, regimes)
    regimes = regimes or list(config["regimes"])
    if set(regimes)-set(config["regimes"]):
        raise ValueError("Unknown regime")
    import hdf5plugin  # noqa: F401 -- source dataset may use plugin compression.
    reference = Path(config["reference_dataset"])
    with h5py.File(reference, "r") as original:
        reference_states = np.unique(state_keys(original["state"][:]))
        original_ids = set(original["episode_idx"][original["ep_offset"][:].tolist()].tolist())
    audit = []
    for regime in regimes:
        query = checked(Path(config["output_dir"])/regime/"queries.json")
        if query["collection_manifest"] != manifest["fingerprint"]:
            raise ValueError("Query export belongs to another collection")
        if file_hash(Path(query["source"]["path"])) != query["dataset_sha256"]:
            raise ValueError("Exported dataset changed")
        seen_initial, overlap, states_count, final_actions = set(), 0, 0, 0
        quality_rows, selected_by_split = [], {split: [] for split in config["splits"]}
        for record in progress(query["episode_files"], config=config, desc=f"Audit {regime}", unit="episode"):
            path = Path(record["path"])
            if file_hash(path) != record["sha256"]:
                raise ValueError("Episode contents changed")
            if record["case"]["episode_id"] in original_ids:
                raise ValueError("Fresh and released episode ID namespaces collide")
            with h5py.File(path, "r") as stream:
                states, actions = stream["state"][:], stream["action"][:]
                if (len(states) != config["raw_steps"]+1 or not np.isfinite(states).all()
                        or not np.isfinite(actions).all() or np.abs(actions[:-1]).max() > 1+1e-6
                        or not stream["action_valid"][:-1].all() or stream["action_valid"][-1]
                        or np.any(actions[-1] != 0)):
                    raise ValueError("Invalid state/action alignment or terminal action sentinel")
                if (config["policy"]["name"] == "bounded-contact"
                        and (np.linalg.norm(actions[:-1], axis=1) > config["policy"]["max_action_norm"]+1e-6).any()):
                    raise ValueError("Collected controls exceed the declared bounded-policy action norm")
                initial = state_keys(states[:1])[0].tobytes()
                if initial in seen_initial:
                    raise ValueError("Repeated initial state across validation/test episodes")
                seen_initial.add(initial)
                keys = state_keys(states)
                positions = np.searchsorted(reference_states, keys)
                valid = positions < len(reference_states)
                overlap += int((reference_states[positions[valid]] == keys[valid]).sum())
                states_count += len(states)
                final_actions += 1
                if config.get("quality"):
                    from .quality_audit import audit_episode
                    quality_rows.append({"split": record["case"]["split"],
                        **audit_episode(config, record, stream, adapter, config["regimes"][regime])})
                    offset = query["episode_metadata"]["offsets"][final_actions-1]
                    selected_by_split[record["case"]["split"]].extend(
                        offset+start for start in record["quality"]["selected_start_steps"])
                    if query["cases"][final_actions-1] != record["execution"]:
                        raise ValueError("Replay seed differs from the accepted collector attempt")
        if config.get("quality"):
            for split, starts in selected_by_split.items():
                if starts != query["evaluation_windows"][split]["raw_start_rows"]:
                    raise ValueError("Exported queries differ from the quality-selected windows")
        with h5py.File(query["source"]["path"], "r") as stream:
            goals = []
            for selection in query["evaluation_windows"].values():
                for start in selection["raw_start_rows"]:
                    current = start+(config["history"]-1)*config["frameskip"]
                    state, goal = stream["state"][current], stream["state"][current+config["goal_offset_steps"]]
                    goals.append(adapter.goal_diagnostics(state, goal, config["regimes"][regime]))
        value = sealed({"passed": overlap == 0, "collection_manifest": manifest["fingerprint"],
            "query_manifest": query["fingerprint"], "regime": regime,
            "source_dataset": str(reference.resolve()), "source_size": reference.stat().st_size,
            "source_mtime_ns": reference.stat().st_mtime_ns, "reference_unique_float32_states": len(reference_states),
            "collected_states": states_count, "exact_released_state_matches": overlap,
            "episodes_with_terminal_action_sentinel": final_actions,
            "q4_goal_diagnostics": {"queries": len(goals),
                "initially_successful": sum(item["initial_success"] for item in goals),
                "metrics": {key: {"mean": float(np.mean([item[key] for item in goals])),
                                  "median": float(np.median([item[key] for item in goals]))}
                            for key in goals[0] if key != "initial_success"},
                "note": ("declared contact/motion/nontrivial-goal eligibility applied before query selection"
                         if config.get("quality") else "diagnostics only; no filtering based on these values")},
            "behavior_quality": ({"thresholds": config["quality"], "by_split": {
                split: {"accepted_episodes": sum(row["split"] == split for row in quality_rows),
                        "attempts": sum(row["attempts"] for row in quality_rows if row["split"] == split),
                        "eligible_windows": sum(row["eligible_windows"] for row in quality_rows if row["split"] == split),
                        "candidate_windows": sum(row["candidate_windows"] for row in quality_rows if row["split"] == split),
                        "minimum_clearance": min(row["minimum_clearance"] for row in quality_rows if row["split"] == split),
                        "attempt_rejection_counts": {reason: sum(row["rejection_counts"].get(reason, 0) for row in quality_rows if row["split"] == split)
                            for reason in ("incomplete_horizon", "episode_visibility", "insufficient_eligible_windows", "native_truncation")},
                        "window_failure_counts": {reason: sum(row["window_failure_counts"][reason] for row in quality_rows if row["split"] == split)
                            for reason in ("visibility", "no_contact", "low_motion", "initial_success")}}
                for split in config["splits"]}} if config.get("quality") else None),
            "test_validation_split": "prespecified disjoint case IDs and seed streams; no evaluated-planner outcome filtering",
            "scope": "exact full seven-coordinate float32 states against the released corpus; not perceptual or near-duplicate exclusion and not an audit of unknown pretraining data"})
        path = Path(config["output_dir"])/regime/"audit.json"
        write_json(path, value)
        audit.append(value)
        print(f"{regime}: {states_count:,} states; {overlap} exact released-source matches. Audit: {path}")
        if overlap:
            raise ValueError("Fresh/source exact state overlap detected; inspect the audit before evaluation")
    return audit
