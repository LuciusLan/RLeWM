"""Create immutable partitions from cached episode metadata and exposure records."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import glob
from pathlib import Path
import random

from ..runtime import file_hash, object_hash, read_json, write_json


def validate_parent(manifest: dict) -> None:
    expected = object_hash({k: v for k, v in manifest.items() if k != "fingerprint"})
    if manifest.get("version") != 2 or manifest.get("fingerprint") != expected:
        raise ValueError("Require a checksummed full-cache version-2 parent manifest")
    metadata, source = manifest["episode_metadata"], manifest["source"]
    lengths, offsets, ids = (metadata[k] for k in ("lengths", "offsets", "ids"))
    if not lengths or len(lengths) != len(offsets) or len(lengths) != len(ids):
        raise ValueError("Episode metadata lengths differ")
    total = 0
    for length, offset in zip(lengths, offsets):
        if type(length) is not int or length < 1 or offset != total:
            raise ValueError("Episode metadata does not partition contiguous source rows")
        total += length
    if total != source["rows"] or len(lengths) != source["episodes"] or len(set(ids)) != len(ids):
        raise ValueError("Episode metadata differs from source dimensions or IDs")
    if source["metadata_sha256"] != object_hash([lengths, offsets, ids]):
        raise ValueError("Source episode metadata checksum mismatch")
    span = (manifest["data_config"]["history"] + 1) * manifest["data_config"]["frameskip"]
    eligible = {ep for ep, length in enumerate(lengths) if length >= span}
    groups = [manifest["episodes"][name] for name in ("memory", "validation", "test")]
    combined = [ep for group in groups for ep in group]
    if len(combined) != len(set(combined)) or set(combined) != eligible:
        raise ValueError("Parent splits must partition eligible episodes without overlap")


def _query_positions(record: dict, offsets: list[int], rows: int) -> set[int]:
    """Read query IDs, never support IDs nested inside adaptation traces."""
    positions = set(record.get("query_episode_positions", []))
    for row in record.get("records", []):
        if isinstance(row, dict):
            source = row.get("source", row)
            if isinstance(source, dict) and "source_episode_position" in source:
                positions.add(source["source_episode_position"])
    for split in ("validation", "test"):
        group = record.get("episodes", {}).get(split, [])
        if isinstance(group, list):
            positions.update(group)
        details = record.get("splits", {}).get(split, {})
        for raw in details.get("raw_start_rows", []) if isinstance(details, dict) else []:
            if type(raw) is not int or not 0 <= raw < rows:
                raise ValueError("Historical query row lies outside the source")
            positions.add(bisect_right(offsets, raw) - 1)
    if any(type(ep) is not int or not 0 <= ep < len(offsets) for ep in positions):
        raise ValueError("Historical query episode lies outside the source")
    return positions


def collect_exposure(parent: dict, data: dict, manifest_globs: list[str]) -> tuple[set[int], list[dict]]:
    """Conservatively exclude scored and previously reserved query episodes."""
    seen = set(parent.get("excluded_inspected_query_episodes", []))
    seen.update(data.get("historical_inspected_episodes", []))
    seen.update(parent["episodes"]["validation"])
    seen.update(parent["episodes"]["test"])
    filenames = {str(Path(p).resolve()) for p in data.get("inspected_reports", []) if Path(p).exists()}
    for pattern in data.get("inspected_report_globs", []) + manifest_globs:
        filenames.update(str(Path(p).resolve()) for p in glob.glob(pattern))
    sources = []
    for filename in sorted(filenames):
        path = Path(filename)
        record = read_json(path)
        positions = _query_positions(record, parent["episode_metadata"]["offsets"], parent["source"]["rows"])
        seen.update(positions)
        sources.append({"path": str(path), "sha256": file_hash(path), "query_episode_count": len(positions)})
    return seen, sources


def resplit_manifest(parent: dict, data: dict, excluded: set[int], parent_path: Path) -> tuple[dict, dict]:
    """Keep validation episodes; select a new independent episode-level test pool."""
    validate_parent(parent)
    data = deepcopy(data)
    if data["memory_windows"] != "all":
        raise ValueError("Episode resplitting requires the full-cache profile")
    if any(data[key] != parent["data_config"][key] for key in ("history", "frameskip")):
        raise ValueError("Resplitting must preserve the cached temporal contract")
    validation = sorted(parent["episodes"]["validation"])
    if data["validation_episodes"] != len(validation):
        raise ValueError("This operation preserves the exact existing validation episode IDs")
    test_n, seed, q = data["test_episodes"], data["split_seed"], data["windows_per_query_episode"]
    if type(test_n) is not int or test_n < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Test count must be positive and split seed nonnegative")
    if q != "all" and (type(q) is not int or q < 1):
        raise ValueError("Windows per episode must be positive or 'all'")
    metadata = deepcopy(parent["episode_metadata"])
    span = (data["history"] + 1) * data["frameskip"]
    eligible = {ep for ep, length in enumerate(metadata["lengths"]) if length >= span}
    excluded = set(excluded) | set(validation) | set(parent["episodes"]["test"])
    if any(type(ep) is not int or not 0 <= ep < len(metadata["lengths"]) for ep in excluded):
        raise ValueError("Excluded query episode lies outside the source")
    candidates = sorted(eligible - excluded)
    if len(candidates) < test_n or len(eligible) <= test_n + len(validation):
        raise ValueError("Insufficient eligible unscored/unreserved episodes for the new test")
    test = sorted(random.Random(seed).sample(candidates, test_n))
    episodes = {"memory": sorted(eligible - set(validation) - set(test)), "validation": validation, "test": test}
    counts = {name: sum(metadata["lengths"][ep] - span + 1 for ep in group) for name, group in episodes.items()}
    selections = {}
    for split in ("validation", "test"):
        if q == "all":
            selections[split] = {"mode": "all", "count": counts[split]}
            continue
        starts = []
        for ep in episodes[split]:
            available = metadata["lengths"][ep] - span + 1
            query_seed = int(object_hash([seed, split, ep, "query-windows"])[:16], 16)
            local = sorted(random.Random(query_seed).sample(range(available), min(q, available)))
            starts.extend(metadata["offsets"][ep] + start for start in local)
        selections[split] = {"mode": "subset", "raw_start_rows": starts, "count": len(starts)}
    provenance = {"operation": "preserve validation, replace test, use remaining eligible episodes as memory",
                  "selection_algorithm": "sorted eligible episode positions; Python random.Random(seed).sample",
                  "old_test_episode_positions": sorted(parent["episodes"]["test"]),
                  "test_exclusion": "previously scored or reserved query episodes; prior support use is not excluded",
                  "new_test_previously_in_parent_memory": sorted(set(test) & set(parent["episodes"]["memory"])),
                  "limits": ["New test episodes were available as retrieval support in earlier development runs",
                             "Released-checkpoint pretraining overlap is unknown"]}
    manifest = {"version": 2, "layout": parent["layout"], "data_config": data,
                "source": deepcopy(parent["source"]), "episodes": episodes, "episode_metadata": metadata,
                "windows": {name: {"mode": "all", "count": count, "span": span, "start_stride": 1}
                            for name, count in counts.items()}, "evaluation_windows": selections,
                "split_origin": {"path": str(Path(parent_path).resolve()), "fingerprint": parent["fingerprint"]},
                "excluded_inspected_query_episodes": sorted(excluded),
                "checkpoint_overlap": parent["checkpoint_overlap"], "resplit_provenance": provenance}
    manifest["fingerprint"] = object_hash(manifest)
    validate_parent(manifest)
    audit = {"manifest_fingerprint": manifest["fingerprint"], "parent_fingerprint": parent["fingerprint"],
             "episode_counts": {name: len(group) for name, group in episodes.items()},
             "all_window_counts": counts,
             "evaluation_query_counts": {name: selection["count"] for name, selection in selections.items()},
             "excluded_prior_query_episodes": len(excluded), "candidate_test_episodes": len(candidates),
             "validation_episode_ids_preserved": validation == sorted(parent["episodes"]["validation"]),
             "new_test_disjoint_from_prior_queries": not bool(set(test) & excluded),
             "new_test_disjoint_from_memory": not bool(set(test) & set(episodes["memory"])),
             "minimum_episode_length": min(metadata["lengths"][ep] for ep in test),
             "new_test_previously_in_memory": len(provenance["new_test_previously_in_parent_memory"]),
             "normalization_and_retrieval": "Must be prepared on the new memory; never reuse old fitted assets",
             "observation_latents": "Reusable with the same source/checkpoint/frozen preprocessing identity",
             "model_or_simulator_run": False, "provenance": provenance}
    return manifest, audit


def write_profile(config: dict, manifest: dict, audit: dict, config_path: Path, split_directory: Path) -> None:
    """Refuse to overwrite any profile or prepared experiment."""
    split_directory, config_path = Path(split_directory), Path(config_path)
    experiment = Path(config["paths"]["experiment_dir"])
    if split_directory.exists() or config_path.exists() or experiment.exists():
        raise FileExistsError("New split/config/experiment paths must not exist; reuse the saved profile or choose new paths")
    if config["data"] != manifest["data_config"]:
        raise ValueError("Profile configuration and manifest data contract differ")
    if Path(config["data"]["split_from"]).resolve() != (split_directory / "manifest.json").resolve():
        raise ValueError("Profile must import the saved new split manifest")
    write_json(split_directory / "manifest.json", manifest)
    write_json(split_directory / "audit.json", audit)
    write_json(config_path, config)
