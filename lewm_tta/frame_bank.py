"""Lazy transition views over one cached embedding per raw observation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .cache import Query
from .model import Context
from .observation_cache import build_or_load_observations, save_array
from .runtime import file_hash, object_hash, read_json, write_json


class WindowItems(Sequence):
    """Metadata materializes only for requested windows, not millions at once."""
    def __init__(self, manifest: dict, starts: np.ndarray):
        self.manifest, self.starts = manifest, starts
        metadata = manifest["episode_metadata"]
        self.offsets = np.asarray(metadata["offsets"], dtype=np.int64)
        self.episode_ids = np.asarray(metadata["ids"], dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        raw_start = int(self.starts[index])
        ep = int(np.searchsorted(self.offsets, raw_start, side="right") - 1)
        start = raw_start - int(self.offsets[ep])
        history, frameskip = self.manifest["data_config"]["history"], self.manifest["data_config"]["frameskip"]
        frames = [raw_start + step * frameskip for step in range(history+1)]
        return {"transition_id": f"pusht:{int(self.episode_ids[ep])}:{start}",
                "source_episode_position": ep, "episode_id": int(self.episode_ids[ep]),
                "start_in_episode": start, "raw_observation_rows": frames,
                "candidate_action_rows": list(range(frames[-2], frames[-1])),
                "source": self.manifest["source"]["path"]}


class WindowTensor:
    """Tensor-like indexing materializes only a search/support/query batch."""
    def __init__(self, bank, kind: str):
        self.bank, self.kind = bank, kind
        self.device = bank.store.latents.device
        self.dtype = torch.float32
        h, f = bank.history, bank.frameskip
        z, a = bank.store.latents.shape[1], bank.store.actions.shape[1] * f
        shapes = {"contexts": (len(bank), h, z), "past_actions": (len(bank), h-1, a),
                  "actions": (len(bank), a), "successors": (len(bank), z)}
        self.shape = shapes[kind]

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to(self.device)
        starts = self.bank.start_rows[index]
        scalar = starts.ndim == 0
        starts = starts.reshape(-1)
        h, f = self.bank.history, self.bank.frameskip
        if self.kind == "contexts":
            rows = starts[:, None] + torch.arange(h, device=self.device)[None] * f
            values = self.bank.store.latents[rows]
        elif self.kind == "successors":
            values = self.bank.store.latents[starts + h*f]
        elif self.kind == "past_actions":
            rows = starts[:, None] + torch.arange((h-1)*f, device=self.device)[None]
            values = self.bank.store.actions[rows].reshape(len(starts), h-1, f*self.bank.store.actions.shape[1])
        else:
            rows = starts[:, None] + (h-1)*f + torch.arange(f, device=self.device)[None]
            values = self.bank.store.actions[rows].reshape(len(starts), f*self.bank.store.actions.shape[1])
        return values[0] if scalar else values


class FrameLatentBank:
    def __init__(self, store, manifest: dict, split: str, starts: np.ndarray, identity: dict):
        self.store, self.manifest, self.split, self.identity = store, manifest, split, identity
        self.history, self.frameskip = manifest["data_config"]["history"], manifest["data_config"]["frameskip"]
        self.items = WindowItems(manifest, starts)
        positions = self.episode_positions
        if len(starts):
            if np.any(starts < 0) or np.any(positions >= len(self.items.offsets)):
                raise ValueError("Window start lies outside the source")
            lengths = np.asarray(manifest["episode_metadata"]["lengths"], dtype=np.int64)
            if np.any(starts + (self.history+1)*self.frameskip > self.items.offsets[positions] + lengths[positions]):
                raise ValueError("Window crosses an episode boundary")
            allowed = np.zeros(len(lengths), dtype=bool)
            allowed[manifest["episodes"][split]] = True
            if not allowed[positions].all():
                raise ValueError("Window belongs to a different episode split")
        self.start_rows = torch.from_numpy(starts).to(store.latents.device)
        for kind in ("contexts", "past_actions", "actions", "successors"):
            setattr(self, kind, WindowTensor(self, kind))

    def __len__(self):
        return len(self.items)

    @property
    def episode_positions(self):
        return np.searchsorted(self.items.offsets, self.items.starts, side="right") - 1

    @property
    def step_positions(self):
        return self.items.starts - self.items.offsets[self.episode_positions]

    def to(self, device):
        return FrameLatentBank(self.store.to(device), self.manifest, self.split, self.items.starts, self.identity)

    def query(self, index: int) -> Query:
        return Query(Context(self.contexts[index:index+1], self.past_actions[index:index+1]),
                     self.actions[index:index+1], self.items[index])

    def for_evaluation(self):
        selection = self.manifest["evaluation_windows"][self.split]
        if selection["mode"] == "all":
            return self
        starts = np.asarray(selection["raw_start_rows"], dtype=np.int64)
        return FrameLatentBank(self.store, self.manifest, self.split, starts, self.identity)


def build_frame_bank(adapter, manifest, normalization, split: str, directory: Path,
                     batch_size: int, checkpoint_rows: int):
    store = build_or_load_observations(adapter, manifest, normalization, directory, batch_size, checkpoint_rows)
    if store.actions.shape[1] * manifest["data_config"]["frameskip"] != adapter.action_dim:
        raise ValueError("Ordered action width differs from checkpoint")
    target = Path(directory) / f"windows-{manifest['fingerprint'][:16]}"
    target.mkdir(parents=True, exist_ok=True)
    path, metadata_path = target / f"{split}-starts.npy", target / f"{split}-index.json"
    specification = manifest["windows"][split]
    if not metadata_path.exists():
        metadata = manifest["episode_metadata"]
        span = specification["span"]
        starts = np.concatenate([np.arange(metadata["offsets"][ep],
            metadata["offsets"][ep] + metadata["lengths"][ep] - span + 1, dtype=np.int64)
            for ep in manifest["episodes"][split]]) if manifest["episodes"][split] else np.empty(0, dtype=np.int64)
        if len(starts) != specification["count"]:
            raise AssertionError("Full-window count differs from manifest")
        save_array(path, starts)
        write_json(metadata_path, {"manifest": manifest["fingerprint"], "split": split,
                                  "specification": specification, "sha256": file_hash(path)})
    metadata = read_json(metadata_path)
    if metadata["manifest"] != manifest["fingerprint"] or metadata["specification"] != specification or file_hash(path) != metadata["sha256"]:
        raise ValueError("Window-index identity/checksum mismatch")
    starts = np.load(path, mmap_mode="c", allow_pickle=False)
    if starts.shape != (specification["count"],) or starts.dtype != np.int64:
        raise ValueError("Invalid window start index")
    identity = {"version": 2, "layout": "shared observed frames and lazy window views",
                "manifest": manifest["fingerprint"], "split": split, "windows": len(starts),
                "observations": store.identity, "window_index_sha256": metadata["sha256"],
                "view_implementation_sha256": file_hash(Path(__file__))}
    return FrameLatentBank(store, manifest, split, starts, identity)
