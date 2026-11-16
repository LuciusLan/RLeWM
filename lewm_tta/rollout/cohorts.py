"""Common observed trajectories and explicit P2-filter/q4-sample cohort policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..frame_bank import FrameLatentBank
from ..runtime import object_hash, write_json
from .config import cohort_horizon, is_scope_comparison, manifest_spec, stage_name, validate_config
from .resources import checked_json, source_metadata


def horizon_rows(start: int, history: int, frameskip: int, horizon: int):
    current = start + (history - 1) * frameskip
    return np.arange(current, current + horizon * frameskip, dtype=np.int64), \
        current + np.arange(1, horizon + 1, dtype=np.int64) * frameskip


def filter_cohort(source: dict, split: str, horizon: int) -> dict:
    metadata = source["episode_metadata"]
    offsets = np.asarray(metadata["offsets"], dtype=np.int64)
    lengths = np.asarray(metadata["lengths"], dtype=np.int64)
    history, frameskip = (source["data_config"][key] for key in ("history", "frameskip"))
    selection = source["evaluation_windows"][split]
    if selection["mode"] == "all":
        starts = np.concatenate([np.arange(offsets[ep], offsets[ep] + lengths[ep] - (history+1)*frameskip + 1)
                                 for ep in source["episodes"][split]])
    else:
        starts = np.asarray(selection["raw_start_rows"], dtype=np.int64)
    if len(np.unique(starts)) != len(starts):
        raise ValueError("Source evaluation subset contains repeated windows")
    episodes = np.searchsorted(offsets, starts, side="right") - 1
    if not np.isin(episodes, source["episodes"][split]).all():
        raise ValueError("Source query windows belong to a different split")
    final_rows = starts + (history - 1 + horizon) * frameskip
    valid = (starts >= offsets[episodes]) & (final_rows < offsets[episodes] + lengths[episodes])
    selected = starts[valid]
    if not len(selected):
        raise ValueError(f"No {split} queries support horizon {horizon}")
    return {"raw_start_rows": selected.tolist(), "queries": len(selected),
            "episodes": len(np.unique(episodes[valid])), "source_queries": len(starts),
            "dropped_for_short_future": int((~valid).sum()),
            "selection": "filter the existing P1 query subset for the longest horizon; no resampling",
            "per_episode": {str(ep): {"retained": int(((episodes == ep) & valid).sum()),
                                      "source": int((episodes == ep).sum())}
                            for ep in source["episodes"][split]}}


def sample_cohort(source: dict, split: str, horizon: int, selection: dict) -> dict:
    if source.get("external_queries"):
        # Collection already reserved the exact windows, before model outcomes.
        # Preserve those windows across P2/P3 and the corresponding previews.
        value = filter_cohort(source, split, horizon)
        if value["dropped_for_short_future"] or any(
                item["retained"] != selection["windows_per_episode"] for item in value["per_episode"].values()):
            raise ValueError("Fresh reserved windows do not match the requested horizon/q count; use a compatible collection manifest")
        value.update(selection="use the collection's prespecified query windows without resampling",
                     windows_per_episode=selection["windows_per_episode"], query_windows_may_overlap=True)
        return value
    history, frameskip = (source["data_config"][key] for key in ("history", "frameskip"))
    metadata = source["episode_metadata"]
    count, seed = selection["windows_per_episode"], selection["seed"]
    starts, per_episode = [], {}
    for episode in source["episodes"][split]:
        # Intersect full cached-window validity with final-target availability.
        length, offset = int(metadata["lengths"][episode]), int(metadata["offsets"][episode])
        available = min(length-(history+1)*frameskip+1,
                        length-(history-1+horizon)*frameskip)
        if available < count:
            raise ValueError(f"Episode {episode} has only {available} horizon-safe windows; requested {count}")
        rng = np.random.default_rng([seed, 0 if split == "validation" else 1, int(episode)])
        selected = np.sort(rng.choice(available, size=count, replace=False)) + offset
        starts.extend(selected.tolist())
        per_episode[str(episode)] = {"retained": count, "eligible": available}
    if not starts or len(set(starts)) != len(starts):
        raise ValueError("Empty or duplicate P2.5 query windows")
    return {"raw_start_rows": starts, "queries": len(starts), "episodes": len(per_episode),
            "windows_per_episode": count, "per_episode": per_episode,
            "selection": "sample after maximum-horizon filtering; no padding or replacement",
            "query_windows_may_overlap": True}


def prepare(config: dict, quiet: bool = False) -> dict:
    validate_config(config)
    source, prepared = source_metadata(config)
    memory_source = source
    if config.get("query_dataset"):
        from ..collection.queries import query_source
        source = query_source(config, memory_source)
    scoped = is_scope_comparison(config)
    splits = {split: (sample_cohort(source, split, cohort_horizon(config), config["query_selection"])
                      if scoped else filter_cohort(source, split, max(config["horizons"])))
              for split in ("validation", "test")}
    value = {"version": 1, "stage": stage_name(config), "specification": manifest_spec(config),
        "p1_manifest_fingerprint": memory_source["fingerprint"],
        "p1_prepared_fingerprint": prepared["fingerprint"],
        "horizons": config["horizons"], "history": source["data_config"]["history"],
        "frameskip": source["data_config"]["frameskip"], "splits": splits,
        "test_status": ("Same P1 test episodes already inspected; not a fresh episode holdout" if scoped else
                        "reuses P1 test episodes already inspected at P1; not a fresh episode holdout")}
    if not scoped:
        value["protocol"] = "adapt on the first recorded action block, then freeze through the recorded action sequence"
    if config.get("query_dataset"):
        value["query_source_fingerprint"] = source["fingerprint"]
        value["test_status"] = "fresh simulator episodes reserved before collection; no evaluated-model success filtering; source-overlap audit required"
    value["fingerprint"] = object_hash(value)
    path = Path(config["experiment_dir"])/"manifest.json"
    if path.exists():
        if checked_json(path) != value:
            raise ValueError("Rollout cohort changed; choose another experiment directory")
    else:
        write_json(path, value)
    if not quiet:
        for split, cohort in splits.items():
            print(f"{stage_name(config)} {split}: {cohort['queries']} queries / {cohort['episodes']} episodes", flush=True)
        detail = ("Original memory/normalization/retrieval assets are reused; new query frames encode at first check/evaluate."
                  if config.get("query_dataset") else "Existing observations/actions and full-memory retrieval keys are reused.")
        print(f"Manifest: {path}\n{detail}", flush=True)
    return value


def load_manifest(config: dict) -> dict:
    value = checked_json(Path(config["experiment_dir"])/"manifest.json")
    source, prepared = source_metadata(config)
    if (value["stage"] != stage_name(config) or value["specification"] != manifest_spec(config)
            or value["p1_manifest_fingerprint"] != source["fingerprint"]
            or value["p1_prepared_fingerprint"] != prepared["fingerprint"]):
        raise ValueError("Rollout cohort differs from config/prepared source; run compatible prepare")
    if config.get("query_dataset"):
        from ..collection.queries import query_source
        if value.get("query_source_fingerprint") != query_source(config, source)["fingerprint"]:
            raise ValueError("Fresh query cohort changed")
    return value


def evaluation_bank(resources, manifest: dict, split: str, limit: int | None = None):
    bank = resources.queries(split)
    starts = np.asarray(manifest["splits"][split]["raw_start_rows"], dtype=np.int64)
    if limit is not None:
        starts = starts[:limit]
    result = FrameLatentBank(bank.store, bank.manifest, split, starts, bank.identity)
    if bank.manifest.get("external_queries"):
        from ..collection.queries import namespace_bank
        return namespace_bank(result)
    return result


def rollout_inputs(bank, index: int, horizon: int):
    """Return only observed context and future actions, without future latents."""
    query = bank.query(index)
    start = int(bank.items.starts[index])
    action_rows, target_rows = horizon_rows(start, bank.history, bank.frameskip, horizon)
    ep = int(bank.episode_positions[index])
    end = int(bank.items.offsets[ep]) + int(bank.manifest["episode_metadata"]["lengths"][ep])
    if int(target_rows[-1]) >= end:
        raise ValueError("Rollout crosses an episode boundary")
    rows = torch.as_tensor(action_rows, device=bank.store.actions.device)
    actions = bank.store.actions[rows].reshape(1, horizon, -1)
    torch.testing.assert_close(actions[:, 0], query.action, rtol=0, atol=0)
    return query, actions, target_rows


def observed_targets(bank, target_rows: np.ndarray):
    """Evaluator-only: called after adaptation and open-loop prediction finish."""
    indices = torch.as_tensor(target_rows, device=bank.store.latents.device)
    return bank.store.latents[indices].unsqueeze(0)
