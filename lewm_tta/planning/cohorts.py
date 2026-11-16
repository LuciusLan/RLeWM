"""Episode-safe offline tasks. Planner inputs have no recorded future actions."""

from dataclasses import dataclass
from pathlib import Path

from ..model import Context
from ..runtime import object_hash, write_json
from ..rollout.cohorts import sample_cohort
from ..rollout.resources import checked_json, source_metadata
from .config import STAGE, evaluation_limits, validate_config


def manifest_value(config):
    source, prepared = source_metadata(config)
    memory_source = source
    if config.get("query_dataset"):
        from ..collection.queries import query_source
        source = query_source(config, memory_source)
    horizon = config["planning"]["horizon"]
    value = {"version": 1, "stage": STAGE, "horizon": horizon,
             "selection": config["query_selection"],
             "p1_manifest_fingerprint": memory_source["fingerprint"],
             "p1_prepared_fingerprint": prepared["fingerprint"],
             "history": source["data_config"]["history"],
             "frameskip": source["data_config"]["frameskip"],
             "goal_rule": "observed latent at current_row + horizon * frameskip",
             "splits": {split: sample_cohort(source, split, horizon, config["query_selection"])
                        for split in ("validation", "test")}, "limits": evaluation_limits(config)}
    if config.get("query_dataset"):
        value["query_source_fingerprint"] = source["fingerprint"]
        value["query_selection_protocol"] = "reuse the fresh collection's prespecified starts; no stage resampling"
        value["query_provenance"] = "new simulator episodes, independent of retrieval memory; exact source-state audit passed"
    value["fingerprint"] = object_hash(value)
    return value


def prepare(config):
    validate_config(config)
    value = manifest_value(config)
    path = Path(config["experiment_dir"])/"manifest.json"
    if path.exists():
        if checked_json(path) != value:
            raise ValueError("P3 cohort changed; choose a new experiment directory")
    else:
        write_json(path, value)
    for split, cohort in value["splits"].items():
        print(f"P3-a {split}: {cohort['queries']} queries / {cohort['episodes']} episodes", flush=True)
    print(f"Manifest: {path}\nReuses P1 caches and cuVS assets; no encoding or simulator.", flush=True)
    return value


def load_manifest(config):
    value = checked_json(Path(config["experiment_dir"])/"manifest.json")
    if value != manifest_value(config):
        raise ValueError("P3 manifest/config/source mismatch; prepare a compatible cohort")
    return value


@dataclass(frozen=True)
class PlanningTask:
    context: Context
    metadata: dict
    current_row: int
    goal_row: int


def planning_task(bank, index, horizon):
    # Do not use bank.query(): it also materializes the recorded next action.
    start = int(bank.items.starts[index])
    episode = int(bank.episode_positions[index])
    current = start + (bank.history-1)*bank.frameskip
    goal_row = current + horizon*bank.frameskip
    end = int(bank.items.offsets[episode]) + bank.manifest["episode_metadata"]["lengths"][episode]
    if goal_row >= end:
        raise ValueError("Planning goal crosses an episode boundary")
    context = Context(bank.contexts[index:index+1], bank.past_actions[index:index+1])
    return PlanningTask(context, bank.items[index], current, goal_row)


def goal_latent(bank, task):
    """Task-provided goal, available to CEM but never passed into TTT."""
    return bank.store.latents[task.goal_row:task.goal_row+1].detach()
