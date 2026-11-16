"""Finite, resumable behavior-quality rejection sampling; never runs a model."""

from pathlib import Path

import numpy as np

from ..runtime import file_hash, object_hash, write_json
from .manifest import attempt_case, checked, sealed
from .quality import assess_episode, geometry_visible, select_starts
from .storage import EpisodeWriter


def collect_quality_episode(environment, config, manifest, regime, case, path, implementation):
    directory = path.parent.parent/"attempts"/path.stem
    directory.mkdir(parents=True, exist_ok=True)
    attempts = []
    identity = {"collection_manifest": manifest["fingerprint"], "case_id": case["case_id"],
                "regime": regime, "code_fingerprint": object_hash(implementation),
                "environment_fingerprint": object_hash(environment.description())}
    for index in range(config["quality"]["max_attempts"]):
        execution = attempt_case(case, index)
        metadata = directory/f"{index:03d}.json"
        if metadata.exists():
            previous = checked(metadata)
            if (any(previous.get(key) != value for key, value in identity.items())
                    or previous["execution"] != execution or previous["attempt"] != index
                    or file_hash(Path(previous["measurements"])) != previous["measurements_sha256"]):
                raise ValueError(f"Collection attempt changed: {metadata}")
            if previous["quality"]["accepted"]:
                # Recover a crash between the data/attempt and episode commits.
                if not path.is_file() or file_hash(path) != previous["episode_sha256"]:
                    raise ValueError("Accepted attempt payload missing or changed")
                return previous, attempts+[previous]
            attempts.append(previous)
            continue
        observation = environment.reset(execution["reset_seed"])
        environment.begin_collection(config["policy"])
        rng = np.random.default_rng(execution["action_seed"])
        states, bounds, contacts, actions = [], [], [], []
        writer = EpisodeWriter(path, config, case, environment.collection_schema())
        try:
            for step in range(config["raw_steps"]+1):
                measurement = environment.collection_measurement()
                extent = measurement["extent"]
                states.append(observation.state.copy())
                bounds.append(measurement["bounds"])
                contacts.append(measurement["contacts"])
                writer.append(observation, actions[-1] if actions else None, measurement=measurement)
                if (not geometry_visible(measurement["bounds"], extent, config["quality"]["visibility_margin"])
                        or observation.truncated or step == config["raw_steps"]):
                    break
                action = environment.collection_action(observation, rng, step, config["policy"])
                actions.append(action.copy())
                observation = environment.step(action)
            quality = assess_episode(config, states, bounds, contacts, extent,
                lambda state, goal: environment.goal_diagnostics(state, goal, config["regimes"][regime]))
            if observation.truncated:
                quality["accepted"] = False
                quality["reasons"].append("native_truncation")
            if quality["accepted"]:
                quality["selected_start_steps"] = select_starts(case, quality["eligible_start_steps"], config["windows_per_episode"])
                writer.commit()
        finally:
            writer.close()
        # Small physical traces for *every* attempt; discarded RGB is never
        # published. Preserve all rejection reasons without per-step JSON logs.
        trace = directory/f"{index:03d}.npz"
        temporary = trace.with_suffix(".partial.npz")
        np.savez_compressed(temporary, states=np.asarray(states), actions=np.asarray(actions),
                            bounds=np.asarray(bounds), contacts=np.asarray(contacts))
        temporary.replace(trace)
        record = sealed({**identity, "attempt": index, "execution": execution, "quality": quality,
                         "measurements": str(trace.resolve()), "measurements_sha256": file_hash(trace),
                         "episode_sha256": file_hash(path) if quality["accepted"] else None})
        write_json(metadata, record)
        attempts.append(record)
        if quality["accepted"]:
            return record, attempts
    last = attempts[-1]["quality"]
    raise ValueError(f"{case['case_id']}: quality attempt cap ({len(attempts)}) exhausted. "
                     f"Last rejection: {last['reasons']}; window failures: {last['window_failure_counts']}. "
                     f"Inspect {directory}; change collection settings in a new/empty root, not the reserved test rules.")
