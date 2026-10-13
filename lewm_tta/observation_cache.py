"""Resumable, shared projected observations on NVMe; no predictor updates."""

from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import inspect
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .data import FrameReader
from .runtime import file_hash, object_hash, read_json, write_json


def save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _encode_observations(adapter, manifest, directory, identity, batch_size, checkpoint_rows):
    """Commit complete row ranges; a restart repeats only the unfinished range."""
    shape = (manifest["source"]["rows"], adapter.latent_dim)
    status_path, values_path = directory / "progress.json", directory / "latents.npy"
    if status_path.exists():
        status = read_json(status_path)
        if status["identity"] != identity:
            raise ValueError("Observation cache identity mismatch")
    else:
        if not values_path.exists():
            allocation = values_path.with_name(f"{values_path.name}.allocating.{os.getpid()}")
            values = np.lib.format.open_memmap(allocation, mode="w+", dtype=np.float32, shape=shape)
            values.flush()
            del values
            with allocation.open("r+b") as stream:
                os.fsync(stream.fileno())
            allocation.replace(values_path)
        # A crash between array allocation and the first progress record has
        # no committed rows. The identity-specific directory makes restarting
        # at row zero safe; shape/precision are checked immediately below.
        status = {"identity": identity, "encoded_rows": 0, "complete": False,
                  "encode_device": str(adapter.device), "batch_size": batch_size}
        write_json(status_path, status)
    values = np.load(values_path, mmap_mode="r+", allow_pickle=False)
    if values.shape != shape or values.dtype != np.float32:
        raise ValueError("Observation cache array shape/precision mismatch")
    completed = int(status["encoded_rows"])
    if not 0 <= completed <= shape[0]:
        raise ValueError("Invalid observation-cache progress")
    if status["complete"]:
        if completed != shape[0] or file_hash(values_path) != status["latents_sha256"]:
            raise ValueError("Completed observation cache checksum mismatch")
        print(f"Reuse observations: {completed:,} cached frames", flush=True)
        return status
    if completed:
        print(f"Resume observations: {completed:,}/{shape[0]:,} frames already committed", flush=True)
    status.setdefault("invocations", []).append({"start_row": completed, "batch_size": batch_size,
        "checkpoint_rows": checkpoint_rows, "device": str(adapter.device)})
    write_json(status_path, status)
    reader = FrameReader(manifest, adapter.image_size)
    started, resumed_at = time.perf_counter(), completed
    try:
        while completed < shape[0]:
            checkpoint_end = min(completed + checkpoint_rows, shape[0])
            for start in range(completed, checkpoint_end, batch_size):
                end = min(start + batch_size, checkpoint_end)
                pixels = reader.batch(start, end)
                latent = adapter.encode(pixels.unsqueeze(1))[:, 0].cpu().numpy()
                if latent.shape != (end-start, adapter.latent_dim) or not np.isfinite(latent).all():
                    raise ValueError("Invalid encoded observation batch")
                values[start:end] = latent
            values.flush()
            with values_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            completed = checkpoint_end
            status["encoded_rows"] = completed
            write_json(status_path, status)
            elapsed = time.perf_counter() - started
            speed = (completed-resumed_at) / max(elapsed, 1e-9)
            remaining_minutes = (shape[0]-completed) / max(speed, 1e-9) / 60
            print(f"Observations: {completed:,}/{shape[0]:,} ({completed/shape[0]:.1%}) | "
                  f"{speed:.1f} frames/s | ETA {remaining_minutes:.1f} min", flush=True)
    finally:
        reader.close()
        values.flush()
    status.update(complete=True, latents_sha256=file_hash(values_path),
                  elapsed_seconds_this_invocation=time.perf_counter()-started)
    write_json(status_path, status)
    return status


@dataclass
class ObservationStore:
    latents: torch.Tensor
    actions: torch.Tensor
    identity: dict
    directory: Path
    _device_copies: dict = field(default_factory=dict, repr=False)

    def to(self, device):
        device = torch.device(device)
        if self.latents.device == device:
            return self
        key = str(device)
        if key not in self._device_copies:
            self._device_copies[key] = ObservationStore(self.latents.to(device), self.actions.to(device),
                                                       self.identity, self.directory)
        return self._device_copies[key]


def build_or_load_observations(adapter, manifest, normalization, directory: Path,
                               batch_size: int, checkpoint_rows: int = 8192) -> ObservationStore:
    # Selection/normalization do not change image embeddings. Bank/adaptation
    # edits do not invalidate an expensive cache of the frozen observations.
    identity = {"version": 1, "source": manifest["source"], "model": adapter.fingerprint,
        "latent_dim": adapter.latent_dim, "dtype": "float32",
        "frame_reader_sha256": hashlib.sha256(inspect.getsource(FrameReader).encode()).hexdigest(),
        "encoder_loop_sha256": hashlib.sha256(inspect.getsource(_encode_observations).encode()).hexdigest()}
    fingerprint = object_hash(identity)
    normalization_hash = object_hash(normalization)
    key = (str(Path(directory).resolve()), fingerprint, normalization_hash)
    stores = getattr(adapter, "_observation_stores", {})
    if key in stores:
        return stores[key]
    target = Path(directory) / f"observations-{fingerprint[:16]}"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is preparing this observation cache") from error
        status = _encode_observations(adapter, manifest, target, identity, batch_size, checkpoint_rows)
        actions_path = target / f"actions-{normalization_hash[:16]}.npy"
        actions_status_path = actions_path.with_suffix(".json")
        if not actions_status_path.exists():
            with h5py.File(manifest["source"]["path"], "r", swmr=True) as source:
                raw = torch.from_numpy(source["action"][:])
            mean, std = torch.tensor(normalization["mean"]), torch.tensor(normalization["std"])
            actions = ((raw - mean) / std).float().numpy()
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite normalized source actions")
            save_array(actions_path, actions)
            write_json(actions_status_path, {"normalization": normalization, "sha256": file_hash(actions_path)})
        actions_status = read_json(actions_status_path)
        if actions_status["normalization"] != normalization or file_hash(actions_path) != actions_status["sha256"]:
            raise ValueError("Normalized-action cache checksum mismatch")
    # Copy-on-write maps expose writable tensors without modifying cache files.
    latents = torch.from_numpy(np.load(target / "latents.npy", mmap_mode="c", allow_pickle=False))
    actions = torch.from_numpy(np.load(actions_path, mmap_mode="c", allow_pickle=False))
    if actions.shape != (manifest["source"]["rows"], manifest["source"]["schema"]["action"]["shape"][1]):
        raise ValueError("Invalid normalized-action cache shape")
    store_identity = {"observation_fingerprint": fingerprint, "latents_sha256": status["latents_sha256"],
                      "normalization": normalization, "actions_sha256": actions_status["sha256"],
                      "raw_frames": len(latents), "directory": str(target)}
    store = ObservationStore(latents, actions, store_identity, target)
    stores[key] = store
    adapter._observation_stores = stores
    return store
