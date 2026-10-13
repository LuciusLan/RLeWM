"""Episode-safe windows, frozen preprocessing, and the native HDF5 reader."""

from __future__ import annotations

from pathlib import Path
import glob

import h5py
import hdf5plugin  # noqa: F401 -- register the source file's compression filters
import numpy as np
import torch

from .runtime import object_hash, read_json, write_json


def inspect_source(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path).resolve(strict=True)
    with h5py.File(path, "r", swmr=True) as source:
        lengths = source["ep_len"][:].astype(np.int64)
        offsets = source["ep_offset"][:].astype(np.int64)
        rows = source["pixels"].shape[0]
        if len(lengths) != len(offsets) or not len(lengths) or np.any(lengths < 1):
            raise ValueError("Invalid episode metadata")
        if not np.array_equal(offsets, np.cumsum(np.r_[0, lengths[:-1]])) or int(lengths.sum()) != rows:
            raise ValueError("Episodes must partition the source rows contiguously")
        if any(source[name].shape[0] != rows for name in ("action", "episode_idx", "step_idx")):
            raise ValueError("Source columns have inconsistent row counts")
        episode_ids = source["episode_idx"][offsets.tolist()]
        if len(set(episode_ids.tolist())) != len(episode_ids):
            raise ValueError("Source episode IDs must be unique")
        schema = {name: {"shape": list(source[name].shape), "dtype": str(source[name].dtype)}
                  for name in ("pixels", "action", "episode_idx", "step_idx")}
    stat = path.stat()
    identity = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "schema": schema, "episodes": len(lengths), "rows": rows,
                "metadata_sha256": object_hash([lengths.tolist(), offsets.tolist(), episode_ids.tolist()]),
                "checksum_scope": "episode metadata, schema, file size and mtime; not full image payload"}
    return identity, lengths, offsets, episode_ids


def inspected_episodes(data_config: dict, own_test_manifest: str | None = None) -> set[int]:
    inspected = set(data_config["historical_inspected_episodes"])
    def include(report):
        # Repeating a frozen test configuration is permitted; its episodes
        # remain excluded from newly constructed splits in other experiments.
        if report.get("split") == "test" and report.get("manifest_fingerprint") == own_test_manifest:
            return
        for record in report.get("records", []):
            inspected.add(int(record["source"]["source_episode_position"]))
        inspected.update(report.get("query_episode_positions", []))
    for filename in data_config["inspected_reports"]:
        path = Path(filename)
        if path.exists():
            report = read_json(path)
            include(report)
    for pattern in data_config.get("inspected_report_globs", []):
        for filename in glob.glob(pattern):
            report = read_json(Path(filename))
            include(report)
    return inspected


def get_manifest(config: dict) -> dict:
    """Reuse an immutable split, or create one before selecting any windows."""
    data = config["data"]
    source, lengths, offsets, episode_ids = inspect_source(Path(config["paths"]["dataset"]))
    path = Path(config["paths"]["experiment_dir"]) / "manifest.json"
    inspected = inspected_episodes(data)
    if path.exists():
        manifest = read_json(path)
        if manifest.get("fingerprint") != object_hash({key: value for key, value in manifest.items() if key != "fingerprint"}):
            raise ValueError("Manifest checksum mismatch")
        if manifest["data_config"] != data or manifest["source"] != source:
            raise ValueError("Prepared data changed; use a new experiment_dir and cache_dir")
        new_inspected = inspected_episodes(data, own_test_manifest=manifest["fingerprint"])
        if new_inspected & set(manifest["episodes"]["test"]):
            raise ValueError("A reserved test episode was inspected since preparation; create a new split")
        return manifest
    if data["memory_windows"] == "all":
        return _full_manifest(config, source, lengths, offsets, episode_ids, inspected, path)
    span = (data["history"] + 1) * data["frameskip"]
    eligible = np.flatnonzero(lengths >= span).tolist()
    rng = np.random.default_rng(data["split_seed"])
    untouched = rng.permutation([ep for ep in eligible if ep not in inspected]).tolist()
    nq = data["validation_episodes"] + data["test_episodes"]
    if len(untouched) <= nq:
        raise ValueError("Insufficient untouched episodes for validation, test, and memory")
    validation = sorted(untouched[:data["validation_episodes"]])
    test = sorted(untouched[data["validation_episodes"]:nq])
    reserved = set(validation) | set(test)
    memory = sorted(set(eligible) - reserved)
    if set(memory) & reserved or set(validation) & set(test):
        raise AssertionError("Episode split leakage")

    def item(ep: int, start: int) -> dict:
        raw_start = int(offsets[ep]) + start
        frames = list(range(raw_start, raw_start + span, data["frameskip"]))
        return {"transition_id": f"pusht:{int(episode_ids[ep])}:{start}",
                "source_episode_position": ep, "episode_id": int(episode_ids[ep]),
                "start_in_episode": start, "raw_observation_rows": frames,
                "candidate_action_rows": list(range(frames[-2], frames[-1])),
                "source": source["path"]}

    def query_windows(episodes: list[int]) -> list[dict]:
        windows = []
        for ep in episodes:
            count = min(data["windows_per_query_episode"], int(lengths[ep]) - span + 1)
            starts = rng.choice(int(lengths[ep]) - span + 1, count, replace=False)
            windows.extend(item(ep, int(start)) for start in sorted(starts.tolist()))
        return windows

    # Equal episode opportunity. Revisit episodes only after each eligible
    # episode has had one chance; never repeat a window within an episode.
    available = {ep: rng.permutation(int(lengths[ep]) - span + 1).tolist() for ep in memory}
    requested = min(data["memory_windows"], sum(len(starts) for starts in available.values()))
    selected = []
    while len(selected) < requested:
        for ep in rng.permutation(memory).tolist():
            if available[ep]:
                selected.append(item(ep, available[ep].pop()))
                if len(selected) == requested:
                    break
    manifest = {"version": 1, "data_config": data, "source": source,
                "episodes": {"memory": memory, "validation": validation, "test": test},
                "windows": {"memory": selected, "validation": query_windows(validation),
                            "test": query_windows(test)},
                "excluded_inspected_query_episodes": sorted(inspected),
                "checkpoint_overlap": "unknown; adaptation holdout is not a pretraining holdout"}
    manifest["fingerprint"] = object_hash(manifest)
    write_json(path, manifest)
    return manifest


def _full_manifest(config, source, lengths, offsets, episode_ids, inspected, path):
    """Compact rules describe every eligible window without millions of dicts."""
    data = config["data"]
    span = (data["history"] + 1) * data["frameskip"]
    eligible = np.flatnonzero(lengths >= span).tolist()
    rng = np.random.default_rng(data["split_seed"])
    parent_path = Path(data["split_from"]) if data.get("split_from") else None
    parent = None
    if parent_path is not None and parent_path.exists():
        parent = read_json(parent_path)
        if parent.get("fingerprint") != object_hash({key: value for key, value in parent.items() if key != "fingerprint"}):
            raise ValueError("Source split manifest checksum mismatch")
        if parent["source"] != source:
            raise ValueError("Source split belongs to a different dataset")
        if any(parent["data_config"][key] != data[key] for key in ("history", "frameskip")):
            raise ValueError("Source split has a different temporal protocol; use split_from=null")
        episodes = parent["episodes"]
        if len(episodes["validation"]) != data["validation_episodes"] or len(episodes["test"]) != data["test_episodes"]:
            raise ValueError("Requested episode counts differ from split_from; use split_from=null for a new split")
        if inspected & set(episodes["test"]):
            raise ValueError("Source split's test episodes were inspected; use a fresh split in a new experiment directory")
    else:
        untouched = rng.permutation([ep for ep in eligible if ep not in inspected]).tolist()
        reserved = data["validation_episodes"] + data["test_episodes"]
        if len(untouched) <= reserved:
            raise ValueError("Insufficient untouched episodes for the requested split")
        validation = sorted(untouched[:data["validation_episodes"]])
        test = sorted(untouched[data["validation_episodes"]:reserved])
        episodes = {"memory": sorted(set(eligible) - set(validation) - set(test)),
                    "validation": validation, "test": test}
    assigned = [set(episodes[name]) for name in ("memory", "validation", "test")]
    if any(assigned[i] & assigned[j] for i in range(3) for j in range(i+1, 3)) or set.union(*assigned) != set(eligible):
        raise ValueError("Episode splits must partition all eligible episodes without overlap")
    counts = {name: int(sum(int(lengths[ep]) - span + 1 for ep in group)) for name, group in episodes.items()}
    evaluation_windows = {}
    for split in ("validation", "test"):
        if data["windows_per_query_episode"] == "all":
            evaluation_windows[split] = {"mode": "all", "count": counts[split]}
        else:
            starts = []
            if parent and parent["data_config"]["windows_per_query_episode"] == data["windows_per_query_episode"]:
                if parent["version"] == 1:
                    starts = [item["raw_observation_rows"][0] for item in parent["windows"][split]]
                else:
                    starts = parent["evaluation_windows"][split]["raw_start_rows"]
            else:
                for ep in episodes[split]:
                    available = int(lengths[ep]) - span + 1
                    selected = rng.choice(available, min(data["windows_per_query_episode"], available), replace=False)
                    starts.extend(int(offsets[ep]) + int(start) for start in sorted(selected.tolist()))
            evaluation_windows[split] = {"mode": "subset", "raw_start_rows": starts, "count": len(starts)}
    manifest = {"version": 2, "layout": "shared observation latents with compact window indices",
        "data_config": data, "source": source, "episodes": episodes,
        "episode_metadata": {"lengths": lengths.tolist(), "offsets": offsets.tolist(), "ids": episode_ids.tolist()},
        "windows": {name: {"mode": "all", "count": count, "span": span, "start_stride": 1}
                    for name, count in counts.items()},
        "evaluation_windows": evaluation_windows,
        "split_origin": {"path": str(parent_path), "fingerprint": parent["fingerprint"]} if parent else None,
        "excluded_inspected_query_episodes": sorted(inspected),
        "checkpoint_overlap": "unknown; adaptation holdout is not a pretraining holdout"}
    manifest["fingerprint"] = object_hash(manifest)
    write_json(path, manifest)
    return manifest


def action_statistics(manifest: dict) -> dict:
    """Native z-score convention, fitted to raw actions in memory episodes only."""
    with h5py.File(manifest["source"]["path"], "r", swmr=True) as source:
        lengths, offsets = source["ep_len"][:], source["ep_offset"][:]
        action = source["action"][:]
    mask = np.zeros(len(action), dtype=bool)
    for ep in manifest["episodes"]["memory"]:
        mask[int(offsets[ep]):int(offsets[ep] + lengths[ep])] = True
    tensor = torch.from_numpy(action[mask])
    if len(tensor) < 2 or not torch.isfinite(tensor).all():
        raise ValueError("Action normalization requires finite memory actions")
    mean, std = tensor.mean(0), tensor.std(0, correction=1)
    if (std <= 0).any():
        raise ValueError("Degenerate memory action coordinates")
    return {"mean": mean.tolist(), "std": std.tolist(), "raw_rows": len(tensor),
            "std_correction": 1, "fit_split": "memory episodes only",
            "convention": "native ZScoreNormalizer before ordered action-block reshaping"}


class WindowReader:
    def __init__(self, manifest: dict, normalization: dict, image_size: int):
        import stable_pretraining as spt
        import stable_worldmodel as swm
        from utils import ZScoreNormalizer, get_img_preprocessor
        data = manifest["data_config"]
        self.frameskip, self.history = data["frameskip"], data["history"]
        self.span = (self.history + 1) * self.frameskip
        transform = spt.data.transforms.Compose(
            get_img_preprocessor("pixels", "pixels", image_size),
            spt.data.transforms.WrapTorchTransform(
                ZScoreNormalizer(torch.tensor(normalization["mean"]).unsqueeze(0),
                                 torch.tensor(normalization["std"]).unsqueeze(0)),
                source="action", target="action"))
        self.dataset = swm.data.HDF5Dataset(path=manifest["source"]["path"],
            frameskip=self.frameskip, num_steps=self.history + 1, transform=transform,
            keys_to_load=["pixels", "action", "episode_idx", "step_idx"], keys_to_cache=["action"])

    def batch(self, items: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        episodes = np.array([item["source_episode_position"] for item in items])
        starts = np.array([item["start_in_episode"] for item in items])
        loaded = self.dataset.load_chunk(episodes, starts, starts + self.span)
        for item, values in zip(items, loaded):
            if not torch.all(values["episode_idx"] == item["episode_id"]):
                raise AssertionError("Window crosses a source episode boundary")
            expected = torch.arange(item["start_in_episode"], item["start_in_episode"] + self.span,
                                    self.frameskip, dtype=values["step_idx"].dtype)
            torch.testing.assert_close(values["step_idx"], expected, rtol=0, atol=0)
        pixels = torch.stack([values["pixels"] for values in loaded])
        actions = torch.stack([values["action"][:self.history] for values in loaded])
        if not torch.isfinite(pixels).all() or not torch.isfinite(actions).all():
            raise ValueError("Nonfinite preprocessed window")
        return pixels, actions

    def close(self) -> None:
        if self.dataset.h5_file is not None:
            self.dataset.h5_file.close()
            self.dataset.h5_file = None


class FrameReader:
    """Read contiguous raw observations using native decoding/preprocessing."""
    def __init__(self, manifest: dict, image_size: int):
        import stable_worldmodel as swm
        from utils import get_img_preprocessor
        metadata = manifest["episode_metadata"]
        self.lengths = np.asarray(metadata["lengths"], dtype=np.int64)
        self.offsets = np.asarray(metadata["offsets"], dtype=np.int64)
        self.ids = np.asarray(metadata["ids"], dtype=np.int64)
        # load_chunk accepts variable lengths. Avoid constructing an unused
        # fixed-window clip_indices list by choosing a span exceeding every ep.
        self.dataset = swm.data.HDF5Dataset(path=manifest["source"]["path"], frameskip=1,
            num_steps=int(self.lengths.max()) + 1, keys_to_load=["pixels", "episode_idx", "step_idx"],
            transform=get_img_preprocessor("pixels", "pixels", image_size))

    def batch(self, start: int, end: int) -> torch.Tensor:
        episodes, starts, ends = [], [], []
        cursor = start
        while cursor < end:
            ep = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            stop = min(end, int(self.offsets[ep] + self.lengths[ep]))
            episodes.append(ep)
            starts.append(cursor - int(self.offsets[ep]))
            ends.append(stop - int(self.offsets[ep]))
            cursor = stop
        chunks = self.dataset.load_chunk(np.array(episodes), np.array(starts), np.array(ends))
        for ep, first, last, values in zip(episodes, starts, ends, chunks):
            if not torch.all(values["episode_idx"] == int(self.ids[ep])):
                raise AssertionError("Raw-frame episode ID mismatch")
            torch.testing.assert_close(values["step_idx"],
                torch.arange(first, last, dtype=values["step_idx"].dtype), rtol=0, atol=0)
        pixels = torch.cat([values["pixels"] for values in chunks])
        if len(pixels) != end - start or not torch.isfinite(pixels).all():
            raise ValueError("Invalid preprocessed observation batch")
        return pixels

    def close(self):
        if self.dataset.h5_file is not None:
            self.dataset.h5_file.close()
            self.dataset.h5_file = None
