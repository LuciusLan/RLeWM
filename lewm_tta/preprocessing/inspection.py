"""Model-free checkpoint/data contracts and explicit cache size estimates."""

from pathlib import Path

from ..runtime import read_json
from .archives import archive_header
from .hdf5 import action_padding, metadata_columns, raw_dataset_path


def cache_bytes(rows, windows, history, latent_dim, raw_action_dim, frameskip):
    """Float32 observations/actions/flat keys and int64 starts, without duplication."""
    ordered = raw_action_dim * frameskip
    dimensions = {"forward": history * latent_dim + history * ordered,
                  "reverse": (history + 1) * latent_dim + (history - 1) * ordered}
    sizes = {"latents": rows * latent_dim * 4,
             "normalized_actions": rows * raw_action_dim * 4,
             "window_starts": windows * 8,
             "forward_keys": windows * dimensions["forward"] * 4,
             "reverse_keys": windows * dimensions["reverse"] * 4}
    return {"arrays": sizes, "total_bytes": sum(sizes.values()), "key_dimensions": dimensions}


def inspect_task(config):
    """Inspect metadata only. No weight deserialization, encoding or environment."""
    from ..config import validate_config
    validate_config(config)
    if not config["data"].get("memory_only"):
        raise ValueError("Multi-task preprocessing expects an explicit memory-only source profile")
    settings, data, paths = config["preprocessing"], config["data"], config["paths"]
    checkpoint = Path(paths["checkpoint"])
    native = read_json(checkpoint / "config.json")
    if not (checkpoint / "weights.pt").is_file():
        raise FileNotFoundError(checkpoint / "weights.pt")
    expected_action = settings["raw_action_dim"] * data["frameskip"]
    if native["predictor"]["num_frames"] != data["history"] or native["action_encoder"]["input_dim"] != expected_action:
        raise ValueError("Checkpoint history/action-block width does not match the task profile")
    if native["encoder"].get("pretrained", False):
        raise ValueError("Expected an offline local checkpoint with encoder.pretrained=false")
    header = archive_header(settings["archive"], settings["member"]) if Path(settings["archive"]).exists() else None
    source, summary = None, None
    target = raw_dataset_path(config)
    span = (data["history"] + 1) * data["frameskip"]
    if target.exists():
        import h5py
        import numpy as np
        from ..data import inspect_source
        source, lengths, offsets, _ = inspect_source(target)
        shape = source["schema"]["action"]["shape"]
        if len(shape) != 2 or shape[1] != settings["raw_action_dim"]:
            raise ValueError(f"Unexpected raw action shape for {data['dataset_id']}: {shape}")
        if header is not None and target.stat().st_size != header["bytes"]:
            raise ValueError("Existing source size differs from the downloaded archive")
        eligible = lengths >= span
        rows, windows = source["rows"], int((lengths[eligible] - span + 1).sum())
        if not windows:
            raise ValueError("Source has no episode-safe memory windows")
        with h5py.File(target, "r", swmr=True) as handle:
            native_columns = metadata_columns(handle)
            columns = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for name, value in handle.items() if isinstance(value, h5py.Dataset)}
            for name in native_columns.values():
                if not np.issubdtype(handle[name].dtype, np.integer):
                    raise ValueError(f"{name} must contain integer metadata")
            _, _, action_audit = action_padding(handle, lengths, offsets)
        summary = {"episodes": len(lengths), "eligible_episodes": int(eligible.sum()),
                   "short_episodes_excluded": int((~eligible).sum()), "rows": rows,
                   "length_min": int(lengths.min()), "length_max": int(lengths.max()),
                   "length_mean": float(lengths.mean()), "columns": columns,
                   "metadata_columns": native_columns, "action_audit": action_audit}
        estimated = False
    else:
        if header is None:
            raise FileNotFoundError(f"Neither source archive nor extracted HDF5 exists for {data['dataset_id']}")
        rows = settings["estimated_raw_rows"]
        # Nominal episode/row budgets are explicitly provisional. The actual
        # episode-safe count replaces them before any allocation or encoding.
        windows = max(0, rows - settings.get("estimated_episodes", 0) * (span - 1))
        estimated = True
    payload = cache_bytes(rows, windows, data["history"], native["projector"]["output_dim"],
                          settings["raw_action_dim"], data["frameskip"])
    return {"dataset": data["dataset_id"], "source": source, "source_summary": summary,
            "archive": header, "source_exists": target.exists(),
            "checkpoint": {"path": str(checkpoint), "image_size": native["encoder"]["image_size"],
                           "latent_dim": native["projector"]["output_dim"], "history": data["history"],
                           "raw_action_dim": settings["raw_action_dim"], "action_block_dim": expected_action},
            "cache": payload, "estimate_before_extraction": estimated,
            "extraction_bytes": 0 if target.exists() else header["bytes"],
            "paths": {"dataset": str(target), "canonical_dataset": paths["dataset"],
                      **{key: paths[key] for key in ("experiment_dir", "cache_dir")}},
            "environment": settings["environment"],
            "protocol": "released source corpus is retrieval memory; validation/test require independently collected episodes"}
