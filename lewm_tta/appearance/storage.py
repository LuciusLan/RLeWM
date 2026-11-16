"""Atomic episode shards, virtual HDF5 exports and preserved query windows."""

from copy import deepcopy
import hashlib
from pathlib import Path

import h5py
import numpy as np

from ..collection.manifest import checked, sealed
from ..runtime import file_hash, write_json
from .source import shard_paths


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def committed_episode(directory, recipe, episode):
    target, receipt = shard_paths(directory, episode)
    if receipt.exists():
        record = checked(receipt)
        if (record["recipe"] != recipe["fingerprint"] or record["episode"] != episode
                or file_hash(target) != record["sha256"]):
            raise ValueError(f"Committed shifted episode changed: {target}")
        return record
    return None


class OffsetRows:
    """Global-row access over an already loaded, episode-sized label array."""
    def __init__(self, values, start):
        self.values, self.start = values, start

    def __getitem__(self, row):
        local = int(row)-self.start
        if not 0 <= local < len(self.values):
            raise IndexError("Renderer requested a frame outside this recorded episode")
        return self.values[local]


def write_episode(directory, recipe, episode, source, renderer, compression, *, batch_size=16, execution=None):
    existing = committed_episode(directory, recipe, episode)
    if existing is not None:
        return existing
    if batch_size < 1:
        raise ValueError("write batch size must be positive")
    target, receipt = shard_paths(directory, episode)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.h5")
    start, rows = episode["parent_raw_start"], episode["rows"]
    hashes = {}
    # Load only small recorded labels once per episode. RGB is never loaded here.
    # Native renderers retain their existing global-row API and exact input values.
    labels = {key: source[key][start:start+rows] for key in recipe["columns"] if key != "pixels"}
    render_source = {key: OffsetRows(values, start) for key, values in labels.items()}
    execution_metadata = {"execution": execution} if execution is not None else {}
    options = {"compression": compression}
    if compression == "gzip":
        options["compression_opts"] = 1
    if target.exists():
        # Recover a crash between committing the HDF5 and its small JSON receipt.
        # Never overwrite a committed or unrelated payload.
        with h5py.File(target, "r") as saved:
            if (saved.attrs.get("appearance_recipe") != recipe["fingerprint"]
                    or saved.attrs.get("parent_raw_start") != start
                    or saved["pixels"].shape != (rows, 224, 224, 3) or saved["pixels"].dtype != np.uint8):
                raise ValueError(f"Unrecognized uncommitted episode: {target}")
            for key, values in labels.items():
                if key != "pixels":
                    hashes[key] = array_hash(values)
                    if hashes[key] != array_hash(saved[key][:]):
                        raise ValueError(f"Uncommitted {key} differs from its parent: {target}")
        record = sealed({"recipe": recipe["fingerprint"], "episode": episode, "path": str(target.resolve()),
                         "sha256": file_hash(target), "labels_sha256": hashes, "renderer": renderer.description,
                         "recorded_controls_executed": 0, "physics_transition_steps": 0,
                         "recovered_unreceipted_payload": True, **execution_metadata})
        write_json(receipt, record)
        return record
    try:
        with h5py.File(temporary, "w", libver="latest") as output:
            for key, values in labels.items():
                output.create_dataset(key, data=values, track_times=False)
                hashes[key] = array_hash(values)
            pixels = output.create_dataset("pixels", shape=(rows, 224, 224, 3), dtype=np.uint8,
                                           chunks=(1, 224, 224, 3), track_times=False, **options)
            for first in range(0, rows, batch_size):
                count = min(batch_size, rows-first)
                block = np.empty((count, 224, 224, 3), dtype=np.uint8)
                for index in range(count):
                    image = renderer.render(render_source, start+first+index)
                    if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                        raise ValueError("Unexpected renderer RGB contract")
                    block[index] = image
                pixels[first:first+count] = block
            output.attrs["appearance_recipe"] = recipe["fingerprint"]
            output.attrs["parent_episode_id"] = episode["episode_id"]
            output.attrs["parent_raw_start"] = start
            output.flush()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    record = sealed({"recipe": recipe["fingerprint"], "episode": episode, "path": str(target.resolve()),
                     "sha256": file_hash(target), "labels_sha256": hashes, "renderer": renderer.description,
                     "recorded_controls_executed": 0, "physics_transition_steps": 0, **execution_metadata})
    write_json(receipt, record)
    return record


def source_identity(target, recipe):
    rows = sum(ep["rows"] for ep in recipe["episodes"])
    stat = target.stat()
    schema = {key: {**spec, "shape": [rows, *spec["shape"][1:]]}
              for key, spec in recipe["columns"].items() if key in ("pixels", "action", "episode_idx", "step_idx")}
    return {"path": str(target.resolve()), "rows": rows, "episodes": len(recipe["episodes"]),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "schema": schema,
            "appearance_recipe": recipe["fingerprint"]}


def export(directory, recipe, parent):
    records = [checked(shard_paths(directory, ep)[1]) for ep in recipe["episodes"]]
    target = directory/"dataset.h5"
    lengths = [ep["rows"] for ep in recipe["episodes"]]
    offsets = np.cumsum(np.r_[0, lengths[:-1]]).tolist()
    if not target.exists():
        temporary = target.with_suffix(".partial.h5")
        with h5py.File(temporary, "w", libver="latest") as output:
            label_datasets = {}
            for key, spec in recipe["columns"].items():
                shape = (sum(lengths), *spec["shape"][1:])
                if key != "pixels":
                    # Materialize small labels instead of a separate N-shard VDS
                    # for every column. This bounds later audit/cache-reader metadata.
                    label_datasets[key] = output.create_dataset(key, shape=shape, dtype=np.dtype(spec["dtype"]), track_times=False)
                    continue
                layout = h5py.VirtualLayout(shape, dtype=np.dtype(spec["dtype"]))
                for offset, episode, record in zip(offsets, recipe["episodes"], records):
                    shape = (episode["rows"], *spec["shape"][1:])
                    layout[offset:offset+episode["rows"]] = h5py.VirtualSource(record["path"], key, shape=shape)
                output.create_virtual_dataset(key, layout)
            for offset, record in zip(offsets, records):
                with h5py.File(record["path"], "r") as shard:
                    for key, values in label_datasets.items():
                        count = shard[key].shape[0]
                        values[offset:offset+count] = shard[key][:]
            if "episode_idx" not in recipe["columns"]:
                output["episode_idx"] = output["ep_idx"]
            output.create_dataset("ep_len", data=lengths)
            output.create_dataset("ep_offset", data=offsets)
            output.attrs["appearance_recipe"] = recipe["fingerprint"]
        temporary.replace(target)
    else:
        with h5py.File(target, "r") as stream:
            if stream.attrs.get("appearance_recipe") != recipe["fingerprint"]:
                raise ValueError("Existing HDF5 export belongs to a different appearance recipe")
    appearance = {"recipe": str((directory/"manifest.json").resolve()), "fingerprint": recipe["fingerprint"],
                  "parent_manifest": recipe["parent_manifest"], "parent_fingerprint": parent["fingerprint"],
                  "settings": recipe["settings"], "classification": "appearance-only; physical trajectories unchanged"}
    source = source_identity(target, recipe)
    exports = {"version": 1, "backend": recipe["backend"], "appearance": appearance,
               "source": source, "dataset_sha256": file_hash(target),
               "episode_files": [{"path": record["path"], "sha256": record["sha256"]} for record in records],
               "episode_metadata": {"lengths": lengths, "offsets": offsets,
                                    "ids": [ep["episode_id"] for ep in recipe["episodes"]]},
               "episodes": {role: [i for i, ep in enumerate(recipe["episodes"]) if ep["split"] == role]
                            for role in ("memory", "validation", "test")}}
    if parent.get("external_queries"):
        # Keep exact q windows and future offsets, remapping only compact HDF5 row offsets.
        value = deepcopy(parent)
        value.pop("fingerprint")
        value.update(exports)
        value["version"] = parent["version"]
        value["cases"] = [parent["cases"][ep["parent_episode_position"]] for ep in recipe["episodes"]]
        value["regime"] = directory.name
        value["shift"] = {"appearance": recipe["settings"]}
        selections = {}
        for role in ("validation", "test"):
            parent_starts = parent["evaluation_windows"][role]
            if parent_starts["mode"] != "subset":
                raise ValueError("Appearance query export currently requires fixed subset windows")
            starts = np.asarray(parent_starts["raw_start_rows"], dtype=np.int64)
            chosen = []
            for offset, ep in zip(offsets, recipe["episodes"]):
                if ep["split"] != role:
                    continue
                left = ep["parent_raw_start"]
                chosen.extend((starts[(starts >= left) & (starts < left+ep["rows"])]-left+offset).tolist())
            selections[role] = {"mode": "subset", "raw_start_rows": chosen, "count": len(chosen)}
        value["evaluation_windows"] = selections
        value = sealed(value)
        filename = "queries.json"
    else:
        value = sealed(exports)
        filename = "episodes.json"
    path = directory/filename
    if path.exists() and checked(path) != value:
        raise ValueError("Committed appearance export changed; use a new output directory")
    if not path.exists():
        write_json(path, value)
    return value, path
