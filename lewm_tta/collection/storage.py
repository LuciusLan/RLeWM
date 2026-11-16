"""Atomic episode commits and a native-schema HDF5 virtual concatenation."""

import json
import os
from pathlib import Path

import h5py
import numpy as np

from ..runtime import file_hash, object_hash, write_json
from .manifest import checked, sealed


def episode_paths(config, regime, case):
    stem = Path(config["output_dir"])/regime/"episodes"/f"{case['split']}-{case['index']:05d}"
    return stem.with_suffix(".h5"), stem.with_suffix(".json")


class EpisodeWriter:
    def __init__(self, path, config, case, schema):
        self.path, self.case, self.rows = path, case, config["raw_steps"]+1
        self.proprio_indices = schema["proprio_indices"]
        self.fields = schema.get("fields", {})
        path.parent.mkdir(parents=True, exist_ok=True)
        # The pipeline holds the collection-root writer lock. Leftover files
        # have no committed metadata and can be discarded after an interruption.
        for previous in path.parent.glob(path.name+".partial.*"):
            previous.unlink()
        self.temporary = path.with_name(path.name+f".partial.{os.getpid()}")
        self.file = h5py.File(self.temporary, "w", libver="latest")
        size = config["image_size"]
        compression = {"compression": config["compression"]}
        if config["compression"] == "gzip":
            compression["compression_opts"] = 1
        self.file.create_dataset("pixels", (self.rows, size, size, 3), dtype="u1", chunks=(1, size, size, 3), **compression)
        for key, width, dtype in (("state", schema["state_width"], "f4"), ("state64", schema["state_width"], "f8"),
                                  ("action", schema["action_width"], "f4"), ("proprio", len(self.proprio_indices), "f4")):
            self.file.create_dataset(key, (self.rows, width), dtype=dtype)
        self.file.create_dataset("action_valid", data=np.r_[np.ones(self.rows-1, dtype=np.uint8), np.uint8(0)])
        for key, shape in self.fields.items():
            self.file.create_dataset(key, (self.rows, *shape), dtype="f8")
        if config.get("quality"):
            self.file.create_dataset("quality_bounds", (self.rows, 2, 4), dtype="f8")
            self.file.create_dataset("quality_contacts", (self.rows,), dtype="i8")
            self.file.attrs["quality_bounds"] = "native xyxy AABBs of complete agent and rotated block shapes, in that order"
            self.file.attrs["quality_contacts"] = "body-pair contact points summed over physics substeps; row r belongs to action[r-1]; row 0 is zero"
        self.next_row = 0

    def append(self, observation, action=None, *, measurement=None):
        index = self.next_row
        if index >= self.rows:
            raise ValueError("Too many observations for the reserved episode")
        if (not np.isfinite(observation.state).all() or observation.state.shape != self.file["state"].shape[1:]
                or observation.rgb.shape != self.file["pixels"].shape[1:]):
            raise ValueError("Invalid native state or RGB shape")
        for key, shape in self.fields.items():
            value = np.asarray(observation.fields[key])
            if value.shape != tuple(shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid native observation field: {key}")
            self.file[key][index] = value
        self.file["pixels"][index] = observation.rgb
        self.file["state"][index] = observation.state
        self.file["state64"][index] = observation.state
        self.file["proprio"][index] = observation.state[self.proprio_indices]
        if "quality_bounds" in self.file:
            if measurement is None:
                raise ValueError("Quality-enabled episodes require geometry/contact measurements")
            self.file["quality_bounds"][index] = measurement["bounds"]
            self.file["quality_contacts"][index] = measurement["contacts"]
        if action is not None:
            if index == 0:
                raise ValueError("An incoming action requires a previous observation")
            if np.shape(action) != self.file["action"].shape[1:] or not np.isfinite(action).all():
                raise ValueError("Invalid incoming action")
            self.file["action"][index-1] = action
        self.next_row += 1

    def commit(self):
        if self.next_row != self.rows:
            raise ValueError("Refusing to publish a partial trajectory")
        self.file["action"][-1] = 0  # Explicitly marked invalid; never used as an action target.
        self.file.attrs["action_alignment"] = "action[t] takes pixels/state[t] to pixels/state[t+1]; final action invalid"
        self.file.flush()
        self.file.close()
        self.temporary.replace(self.path)

    def close(self):
        if self.file.id.valid:
            self.file.close()
        if self.temporary.exists():
            self.temporary.unlink()


def export_regime(config, manifest, regime):
    """Only a complete regime receives an evaluation manifest; no payload copies."""
    paths = [episode_paths(config, regime, case) for case in manifest["cases"]]
    if any(not path.is_file() or not metadata.is_file() for path, metadata in paths):
        return None
    episodes = [checked(metadata) for _, metadata in paths]
    for record, (path, _) in zip(episodes, paths):
        if (record["collection_manifest"] != manifest["fingerprint"] or record["regime"] != regime
                or record["sha256"] != file_hash(path)):
            raise ValueError("Episode identity or contents changed")
    rows = config["raw_steps"]+1
    offsets = np.arange(len(paths), dtype=np.int64)*rows
    target = Path(config["output_dir"])/regime/"dataset.h5"
    target_manifest = target.with_name("queries.json")
    # An exported view is immutable. Verify its dependencies rather than rewriting
    # it on resume, which would invalidate observation-cache file identities.
    if target_manifest.exists():
        previous = checked(target_manifest)
        if previous["collection_manifest"] != manifest["fingerprint"] or previous["episode_files"] != episodes:
            raise ValueError("Exported collection changed")
        if file_hash(target) != previous["dataset_sha256"]:
            raise ValueError("Exported virtual dataset changed")
        from .goals import export_catalog
        export_catalog(config, previous)
        return previous
    temporary = target.with_suffix(".partial.h5")
    with h5py.File(temporary, "w", libver="latest") as output:
        keys = ["pixels", "state", "state64", "action", "proprio", "action_valid"]
        from ..simulation.environments import fresh_environment_class
        keys += list(fresh_environment_class(config["backend"]).collection_schema().get("fields", {}))
        if config.get("quality"):
            keys += ["quality_bounds", "quality_contacts"]
        for key in keys:
            with h5py.File(paths[0][0], "r") as first:
                shape, dtype = first[key].shape, first[key].dtype
            layout = h5py.VirtualLayout((rows*len(paths), *shape[1:]), dtype)
            for index, (path, _) in enumerate(paths):
                layout[index*rows:(index+1)*rows] = h5py.VirtualSource(str(path.resolve()), key, shape=shape)
            output.create_virtual_dataset(key, layout)
        output.create_dataset("ep_len", data=np.full(len(paths), rows, dtype=np.int64))
        output.create_dataset("ep_offset", data=offsets)
        output.create_dataset("episode_idx", data=np.repeat([case["episode_id"] for case in manifest["cases"]], rows))
        output.create_dataset("step_idx", data=np.tile(np.arange(rows, dtype=np.int64), len(paths)))
        output.attrs["collection_manifest"] = manifest["fingerprint"]
        output.attrs["regime"] = regime
        output.attrs["shift"] = json.dumps(config["regimes"][regime], sort_keys=True)
    temporary.replace(target)
    stat = target.stat()
    with h5py.File(target, "r") as stream:
        action_width = stream["action"].shape[1]
    source = {"path": str(target.resolve()), "rows": rows*len(paths), "episodes": len(paths),
              "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "schema": {"action": {"shape": [rows*len(paths), action_width], "dtype": "float32"}},
              "collection_manifest": manifest["fingerprint"], "regime": regime}
    splits = {split: [i for i, case in enumerate(manifest["cases"]) if case["split"] == split]
              for split in ("validation", "test")}
    selections = {}
    from .goals import available_starts, export_catalog
    available = available_starts(config)
    for split, indices in splits.items():
        starts = []
        for index in indices:
            if config.get("quality"):
                quality = episodes[index]["quality"]
                if not quality["accepted"]:
                    raise ValueError("Cannot export an ineligible episode")
                chosen = quality["selected_start_steps"]
                if (len(chosen) != config["windows_per_episode"] or len(set(chosen)) != len(chosen)
                        or not set(chosen).issubset(quality["eligible_start_steps"])):
                    raise ValueError("Invalid quality-selected window indices")
            else:
                seed = int(object_hash([manifest["cases"][index]["case_id"], "windows"])[:16], 16)
                chosen = np.random.default_rng(seed).choice(available, config["windows_per_episode"], replace=False)
            starts.extend((offsets[index]+np.sort(chosen)).tolist())
        selections[split] = {"mode": "subset", "raw_start_rows": starts, "count": len(starts)}
    extra = ({"backend": config["backend"], "goal_offset_steps": config["goal_offset_steps"],
              "goal_protocols": config["goal_protocols"]} if "goal_protocols" in config else {})
    value = sealed({**extra, "version": 2 if config.get("quality") else 1, "external_queries": True, "collection_manifest": manifest["fingerprint"],
        "collection_manifest_path": str((Path(config["output_dir"])/"manifest.json").resolve()),
        "dataset_sha256": file_hash(target), "episode_files": episodes, "source": source,
        "data_config": {"history": config["history"], "frameskip": config["frameskip"]},
        "episodes": {"memory": [], **splits}, "episode_metadata": {"lengths": [rows]*len(paths),
            "offsets": offsets.tolist(), "ids": [case["episode_id"] for case in manifest["cases"]]},
        "evaluation_windows": selections,
        "cases": [record.get("execution", record["case"]) for record in episodes], "regime": regime,
        "shift": config["regimes"][regime], "collection_policy": config["policy"],
        "quality_constraints": config.get("quality"),
        "goal_rule": f"observed future RGB/state exactly {config['goal_offset_steps']} raw actions after the current observation",
        "pretraining_limit": "new simulator trajectories; no claim about unknown additional pretraining corpora; exact released-source overlap must be audited"})
    write_json(target_manifest, value)
    export_catalog(config, value)
    print(f"Exported {len(paths)} episodes: {target_manifest}")
    return value
