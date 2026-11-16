"""Immutable parent identity, split reservation and compact row mappings."""

from importlib.metadata import distribution, version
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- register released pixel compression
import numpy as np

from ..collection.manifest import checked, sealed
from ..runtime import file_hash


def implementation():
    library = Path(__file__).resolve().parents[1]
    files = list(Path(__file__).parent.glob("*.py")) + [
        library/"runtime.py", library/"collection/manifest.py",
        library/"preprocessing/hdf5.py",
        library/"simulation/environments/reacher_rendering.py"]
    native = distribution("stable-worldmodel")
    names = ("envs/pusht/env.py", "envs/utils.py", "envs/dmcontrol/reacher.py", "envs/dmcontrol/dmcontrol.py")
    native_files = {str(name): file_hash(native.locate_file(name)) for name in native.files or []
                    if any(str(name).endswith("stable_worldmodel/" + suffix) for suffix in names)}
    if len(native_files) != len(names):
        raise ValueError("Cannot fingerprint the installed stable-worldmodel renderer sources")
    return {"code": {str(path.relative_to(library)): file_hash(path) for path in sorted(files)},
            "native_code": native_files,
            "packages": {name: version(name) for name in
                         ("stable-worldmodel", "h5py", "numpy", "Pillow", "pygame", "pymunk", "dm-control", "mujoco", "opencv-python")}}


def load_parent(path, backend):
    path = Path(path).resolve(strict=True)
    parent = checked(path)
    if parent.get("appearance"):
        raise ValueError("Start from the original parent cohort, rather than nesting appearance shifts")
    task = parent.get("backend", parent.get("data_config", {}).get("dataset_id", "pusht"))
    if task != backend:
        raise ValueError(f"Parent task {task!r} differs from --dataset {backend!r}")
    source = Path(parent["source"]["path"]).resolve(strict=True)
    stat = source.stat()
    if stat.st_size != parent["source"]["bytes"] or stat.st_mtime_ns != parent["source"]["mtime_ns"]:
        raise ValueError("Parent HDF5 changed or moved; restore/re-prepare its original manifest")
    if parent.get("dataset_sha256") and file_hash(source) != parent["dataset_sha256"]:
        raise ValueError("Parent HDF5 checksum changed")
    for episode in parent.get("episode_files", []):
        if file_hash(Path(episode["path"])) != episode["sha256"]:
            raise ValueError("A parent episode payload changed")
    metadata = parent["episode_metadata"]
    lengths, offsets, ids = (np.asarray(metadata[name], dtype=np.int64) for name in ("lengths", "offsets", "ids"))
    if (not len(lengths) or len(lengths) != len(offsets) or len(ids) != len(lengths)
            or np.any(lengths < 1) or len(set(ids.tolist())) != len(ids)
            or not np.array_equal(offsets, np.cumsum(np.r_[0, lengths[:-1]]))):
        raise ValueError("Invalid parent episode partition")
    with h5py.File(source, "r") as stream:
        # Canonical native source views reference a large raw dependency.
        from ..preprocessing.hdf5 import view_record
        view = view_record(stream)
        if not np.array_equal(stream["ep_len"][:], lengths) or not np.array_equal(stream["ep_offset"][:], offsets):
            raise ValueError("Parent episode metadata differs from HDF5")
        ep_key = "episode_idx" if "episode_idx" in stream else "ep_idx"
        if not np.array_equal(stream[ep_key][offsets.tolist()], ids):
            raise ValueError("Parent episode IDs differ from HDF5")
        rows = int(lengths.sum())
        columns = {}
        for key, value in stream.items():
            if key in ("ep_len", "ep_offset"):
                continue
            if not isinstance(value, h5py.Dataset) or not value.shape or value.shape[0] != rows or value.dtype.hasobject:
                raise ValueError(f"Unsupported non-row-aligned parent column: {key}")
            columns[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if columns["pixels"]["shape"][1:] != [224, 224, 3] or columns["pixels"]["dtype"] != "uint8":
            raise ValueError("Expected LeWM RGB224 uint8 observations")
        required = ("state",) if backend == "pusht" else ("qpos", "qvel")
        if any(name not in stream for name in ("action", "step_idx", *required)):
            raise ValueError("Parent lacks recorded actions, step metadata or renderable physical states")
    dependencies = {"source": {"path": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                    "native_view": view,
                    "episode_files": [{"path": ep["path"], "sha256": ep["sha256"]} for ep in parent.get("episode_files", [])]}
    return parent, columns, dependencies


def specification(args, settings):
    parent, columns, dependencies = load_parent(args.source_manifest, args.dataset)
    external = bool(parent.get("external_queries"))
    split = args.split or ("all" if external else "memory")
    if (external and split == "memory") or (not external and split != "memory"):
        raise ValueError("Released trajectories may supply memory only; use fresh queries.json for validation/test")
    allowed = ("validation", "test") if external and split == "all" else (split,)
    roles = {position: role for role in allowed for position in parent["episodes"][role]}
    selected = sorted(position for role in allowed for position in parent["episodes"][role])
    if len(set(selected)) != len(selected):
        raise ValueError("Parent episode splits overlap")
    if args.limit_episodes is not None:
        if args.limit_episodes < 1:
            raise ValueError("--limit-episodes must be positive")
        if args.limit_episodes < len(selected):
            selected = sorted(np.random.default_rng(args.episode_seed).choice(selected, args.limit_episodes, replace=False).tolist())
    if not selected:
        raise ValueError("No episodes in the requested parent split")
    if any(type(position) is not int or not 0 <= position < len(parent["episode_metadata"]["ids"]) for position in selected):
        raise ValueError("Invalid parent episode positions")
    metadata = parent["episode_metadata"]
    episodes = []
    for position in selected:
        role = roles[position]
        episodes.append({"parent_episode_position": position, "episode_id": metadata["ids"][position],
                         "parent_raw_start": metadata["offsets"][position],
                         "rows": metadata["lengths"][position], "split": role})
    value = sealed({"version": 1, "stage": "appearance-synthesis", "backend": args.dataset,
                    "parent_manifest": str(args.source_manifest.resolve()), "parent_fingerprint": parent["fingerprint"],
                    "dependencies": dependencies, "columns": columns, "settings": settings,
                    "selection": {"split": split, "limit_episodes": args.limit_episodes, "episode_seed": args.episode_seed},
                    "episodes": episodes, "implementation": implementation(),
                    "label_policy": "Recorded actions, states, validity, steps and IDs are copied without numerical changes; no controls replayed",
                    "split_policy": "All visual copies inherit their parent episode identity and split; never mix their val/test ancestors into memory"})
    return value, parent


def shard_paths(directory, episode):
    stem = f"episode-{episode['parent_episode_position']:06d}"
    return directory/"episodes"/(stem + ".h5"), directory/"episodes"/(stem + ".json")
