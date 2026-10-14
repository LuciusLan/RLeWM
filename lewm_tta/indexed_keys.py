"""Portable, resumable float32 index vectors over memory transitions only."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path

import numpy as np
import torch

from .runtime import file_hash, object_hash, read_json, write_json


FIELDS = {"context": "contexts", "past_actions": "past_actions",
          "action": "actions", "successor": "successors"}
DIRECTIONS = {"forward": ("context", "past_actions", "action"),
              "reverse": ("context", "past_actions", "successor")}


@dataclass(frozen=True)
class KeyLayout:
    dimensions: dict[str, int]
    scales: dict[str, float]
    beta_action: float
    beta_successor: float

    @classmethod
    def from_memory(cls, memory, calibration: dict, settings: dict):
        return cls({name: math.prod(getattr(memory, field).shape[1:]) for name, field in FIELDS.items()},
                   dict(calibration["scales"]), settings["beta_action"], settings["beta_successor"])

    def width(self, direction: str) -> int:
        return sum(self.dimensions[name] for name in DIRECTIONS[direction])

    def factor(self, name: str) -> float:
        weight = {"action": self.beta_action, "successor": self.beta_successor}.get(name, 1.0)
        scale, dimension = self.scales[name], self.dimensions[name]
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(weight) or weight < 0:
            raise ValueError("Index weights/scales must be finite and nonnegative/positive")
        return math.sqrt(weight / (dimension * scale)) if dimension else 0.0

    def transform(self, direction: str, components: dict[str, torch.Tensor]) -> torch.Tensor:
        blocks = []
        for name in DIRECTIONS[direction]:
            value = components[name]
            # reshape, rather than flatten, also supports empty past-action blocks.
            block = value.reshape(value.shape[0], self.dimensions[name])
            blocks.append(block.to(dtype=torch.float32) * self.factor(name))
        return torch.cat(blocks, dim=1).contiguous()

    def identity(self) -> dict:
        return {"dimensions": self.dimensions, "scales": self.scales,
                "beta_action": self.beta_action, "beta_successor": self.beta_successor,
                "metric": "sqeuclidean", "dtype": "float32",
                "rule": "concatenate blocks multiplied by sqrt(weight/(dimension*scale))"}


def key_identity(memory, layout: KeyLayout) -> dict:
    if memory.split != "memory":
        raise ValueError("Only observed memory transitions may enter a retrieval index")
    return {"version": 1, "memory_cache": memory.identity, "memory_windows": len(memory),
            "layout": layout.identity(), "generator_sha256": file_hash(Path(__file__))}


def _prepare_direction(memory, layout, direction, path, identity, batch_rows, checkpoint_rows):
    shape = (len(memory), layout.width(direction))
    values_path, status_path = path / f"{direction}.npy", path / f"{direction}-progress.json"
    if status_path.exists():
        status = read_json(status_path)
        if status["identity"] != identity:
            raise ValueError("Indexed-key identity mismatch")
    else:
        if not values_path.exists():
            temporary = values_path.with_name(f"{values_path.name}.allocating.{os.getpid()}")
            values = np.lib.format.open_memmap(temporary, mode="w+", shape=shape, dtype=np.float32)
            values.flush()
            del values
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            temporary.replace(values_path)
        status = {"identity": identity, "completed_rows": 0, "complete": False}
        write_json(status_path, status)
    values = np.load(values_path, mmap_mode="r+", allow_pickle=False)
    if values.shape != shape or values.dtype != np.float32:
        raise ValueError("Indexed-key shape/precision mismatch")
    completed = int(status["completed_rows"])
    if not 0 <= completed <= shape[0]:
        raise ValueError("Invalid indexed-key progress")
    if status["complete"]:
        if completed != shape[0] or file_hash(values_path) != status["sha256"]:
            raise ValueError("Indexed-key checksum mismatch")
        print(f"Reuse {direction} index keys: {completed:,} windows", flush=True)
        return {"path": str(values_path), "shape": list(shape), "sha256": status["sha256"],
                "file_bytes": values_path.stat().st_size}
    if completed:
        print(f"Resume {direction} index keys: {completed:,}/{shape[0]:,}", flush=True)
    with torch.no_grad():
        while completed < shape[0]:
            checkpoint_end = min(completed + checkpoint_rows, shape[0])
            for start in range(completed, checkpoint_end, batch_rows):
                end = min(start + batch_rows, checkpoint_end)
                components = {name: getattr(memory, FIELDS[name])[start:end]
                              for name in DIRECTIONS[direction]}
                keys = layout.transform(direction, components).cpu().numpy()
                if not np.isfinite(keys).all():
                    raise ValueError("Nonfinite indexed keys")
                values[start:end] = keys
            values.flush()
            with values_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            completed = checkpoint_end
            status["completed_rows"] = completed
            write_json(status_path, status)
            print(f"Index keys {direction}: {completed:,}/{shape[0]:,} ({completed/shape[0]:.1%})", flush=True)
    status.update(complete=True, sha256=file_hash(values_path))
    write_json(status_path, status)
    return {"path": str(values_path), "shape": list(shape), "sha256": status["sha256"],
            "file_bytes": values_path.stat().st_size}


def key_cache(memory, calibration: dict, settings: dict, directory: Path, *, build: bool) -> dict:
    """Row i always corresponds to memory.items[i]; no validation/test rows."""
    layout = KeyLayout.from_memory(memory, calibration, settings)
    identity = key_identity(memory, layout)
    path = Path(directory) / "retrieval" / f"cuvs-{object_hash(identity)[:16]}"
    record_path = path / "index.json"
    if not build:
        if not record_path.exists():
            raise FileNotFoundError("cuVS index keys are not prepared; run p1_run.sh prepare first")
        record = read_json(record_path)
        if record["identity"] != identity:
            raise ValueError("Prepared cuVS key identity differs from memory/scales/weights")
        for direction in DIRECTIONS:
            entry = record["arrays"][direction]
            values_path = Path(entry["path"])
            values = np.load(values_path, mmap_mode="r", allow_pickle=False)
            if (list(values.shape) != entry["shape"] or values.dtype != np.float32 or
                    values_path.stat().st_size != entry["file_bytes"] or file_hash(values_path) != entry["sha256"]):
                raise ValueError(f"Invalid {direction} index payload")
        return record
    if not len(memory):
        return {"backend": "cuvs", "empty_memory": True, "identity": identity, "arrays": {},
                "fingerprint": object_hash(identity), "numeric_payload_bytes": 0}
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arrays = {direction: _prepare_direction(memory, layout, direction, path, identity,
                  settings.get("index_batch_rows", 8192), settings.get("index_checkpoint_rows", 131072))
                  for direction in DIRECTIONS}
        record = {"backend": "cuvs", "identity": identity, "arrays": arrays,
                  "fingerprint": object_hash({"identity": identity, "arrays": arrays}),
                  "numeric_payload_bytes": len(memory) * sum(layout.width(d) for d in DIRECTIONS) * 4,
                  "storage": "portable float32 vectors; cuVS flat handles constructed once on device load",
                  "row_mapping": "unchanged memory row order; metadata retained in original window manifest"}
        write_json(record_path, record)
    return record
