"""Canonical metadata and terminal-action views over immutable native HDF5s."""

import json
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from ..runtime import file_hash, object_hash

VIEW_ATTRIBUTE = "lewm_source_view"


def raw_dataset_path(config):
    return Path(config.get("preprocessing", {}).get("raw_dataset", config["paths"]["dataset"]))


def metadata_columns(source):
    """Recognize released naming conventions without inventing episode IDs."""
    episode = next((name for name in ("episode_idx", "ep_idx") if name in source), None)
    if episode is None or "step_idx" not in source:
        raise ValueError("Source requires ep_len/ep_offset, episode_idx or ep_idx, and step_idx")
    return {"episode_idx": episode, "step_idx": "step_idx"}


def action_padding(source, lengths, offsets):
    """Accept only all-NaN action placeholders at episode-terminal observations."""
    actions = source["action"][:]
    if actions.ndim != 2 or not np.issubdtype(actions.dtype, np.floating):
        raise ValueError("Native actions must be a two-dimensional floating-point array")
    valid = np.isfinite(actions).all(axis=1)
    invalid = np.flatnonzero(~valid)
    terminal = np.asarray(offsets, dtype=np.int64) + np.asarray(lengths, dtype=np.int64) - 1
    if len(invalid) and (not np.isin(invalid, terminal).all() or not np.isnan(actions[invalid]).all()):
        raise ValueError("Nonfinite actions outside all-NaN terminal padding; source requires an explicit data audit")
    audit = {"valid_rows": int(valid.sum()), "terminal_padding_rows": len(invalid),
             "policy": "exclude padding from statistics; finite storage placeholders are never observed transitions"}
    return actions, valid, audit


def view_record(source):
    """Validate a view's native dependency before treating it as a source."""
    if VIEW_ATTRIBUTE not in source.attrs:
        return None
    record = json.loads(source.attrs[VIEW_ATTRIBUTE])
    if record.get("fingerprint") != object_hash(record.get("identity")):
        raise ValueError("Canonical source view identity checksum mismatch")
    native = record["identity"]["raw_source"]
    stat = Path(native["path"]).stat()
    if stat.st_size != native["bytes"] or stat.st_mtime_ns != native["mtime_ns"]:
        raise ValueError("Native HDF5 behind the canonical view changed; rerun preprocessing")
    return record


def prepare_source_view(config):
    """Link large columns, alias ep_idx, and store only cleaned small actions.

    Terminal observations remain available as states. Their absent outgoing
    actions get finite storage placeholders plus an explicit validity mask.
    No image/state payload is copied or modified.
    """
    if "raw_dataset" not in config.get("preprocessing", {}):
        return None
    from ..data import inspect_source

    raw = raw_dataset_path(config).resolve(strict=True)
    target = Path(config["paths"]["dataset"]).resolve()
    if raw == target:
        raise ValueError("Canonical source view must be separate from the immutable native HDF5")
    identity, lengths, offsets, _ = inspect_source(raw)
    specification = {"version": 1, "raw_source": identity,
                     "adapter_sha256": file_hash(Path(__file__)),
                     "action_padding": "terminal all-NaN rows only; excluded using action_valid"}
    fingerprint = object_hash(specification)
    if target.exists():
        with h5py.File(target, "r", swmr=True) as current:
            saved = view_record(current)
            if saved is None:
                raise ValueError(f"Refusing to replace an unrecognized HDF5: {target}")
            if saved["fingerprint"] == fingerprint:
                print(f"Reuse canonical source view: {target}", flush=True)
                return saved
        # A prepared manifest must never be silently rebound to a different view.
        if (Path(config["paths"]["experiment_dir"]) / "manifest.json").exists():
            raise ValueError("Canonical source adapter changed after preparation; use a new output/cache directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        with h5py.File(raw, "r", swmr=True) as native:
            columns = metadata_columns(native)
            actions, valid, audit = action_padding(native, lengths, offsets)
            if "action_valid" in native:
                raise ValueError("Native source already contains action_valid; an explicit adapter is required")
            actions[~valid] = 0  # Storage placeholder only; validity remains explicit.
            record = {"identity": specification, "fingerprint": fingerprint,
                      "metadata_columns": columns, "action_audit": audit}
            with h5py.File(temporary, "w", libver="latest") as view:
                for name in native:
                    if name != "action":
                        view[name] = h5py.ExternalLink(str(raw), f"/{name}")
                for canonical, actual in columns.items():
                    if canonical != actual:
                        view[canonical] = h5py.ExternalLink(str(raw), f"/{actual}")
                view.create_dataset("action", data=actions, track_times=False)
                view.create_dataset("action_valid", data=valid, track_times=False)
                view.attrs[VIEW_ATTRIBUTE] = json.dumps(record, sort_keys=True)
                view.flush()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        # Stable metadata identity when this small facade is restored later.
        os.utime(temporary, ns=(identity["mtime_ns"], identity["mtime_ns"]))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Canonical source: {columns['episode_idx']} -> episode_idx; "
          f"{audit['terminal_padding_rows']:,} terminal actions masked | {target}", flush=True)
    return record
