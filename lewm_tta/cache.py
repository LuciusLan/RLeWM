"""Frozen projected-latent caches; query inputs deliberately omit query truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from .data import WindowReader
from .model import Context, LeWMAdapter
from .runtime import file_hash, object_hash


@dataclass(frozen=True)
class Query:
    context: Context
    action: torch.Tensor
    metadata: dict


@dataclass
class LatentBank:
    split: str
    contexts: torch.Tensor
    past_actions: torch.Tensor
    actions: torch.Tensor
    successors: torch.Tensor
    items: list[dict]
    identity: dict

    def __len__(self):
        return len(self.items)

    @property
    def episode_positions(self):
        return np.asarray([item.get("source_episode_position", index)
                           for index, item in enumerate(self.items)], dtype=np.int64)

    @property
    def step_positions(self):
        return np.asarray([item.get("start_in_episode", 0) for item in self.items], dtype=np.int64)

    def to(self, device) -> "LatentBank":
        return LatentBank(self.split, *(tensor.to(device) for tensor in
            (self.contexts, self.past_actions, self.actions, self.successors)), self.items, self.identity)

    def query(self, index: int) -> Query:
        return Query(Context(self.contexts[index:index+1], self.past_actions[index:index+1]),
                     self.actions[index:index+1], self.items[index])

    def for_evaluation(self):
        return self


def build_or_load_cache(adapter: LeWMAdapter, manifest: dict, normalization: dict,
                        split: str, directory: Path, batch_size: int = 32, checkpoint_rows: int = 8192):
    if manifest["version"] == 2:
        from .frame_bank import build_frame_bank
        return build_frame_bank(adapter, manifest, normalization, split, directory, batch_size, checkpoint_rows)
    identity = {"version": 1, "manifest": manifest["fingerprint"], "model": adapter.fingerprint,
                "implementation": {name: file_hash(Path(__file__).parent / name) for name in ("cache.py", "data.py")},
                "normalization": normalization, "split": split,
                "input_semantics": "H observations, H-1 past blocks, candidate block, observed successor"}
    fingerprint = object_hash(identity)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}-{fingerprint[:16]}.pt"
    if path.exists():
        contents = torch.load(path, map_location="cpu", weights_only=True)
        if contents["identity"] != identity:
            raise ValueError("Cache identity mismatch")
        bank = LatentBank(**contents)
        if bank.items != manifest["windows"][split]:
            raise ValueError("Cached transitions differ from the manifest")
        expected_shapes = ((len(bank), adapter.history, adapter.latent_dim),
                           (len(bank), adapter.history-1, adapter.action_dim),
                           (len(bank), adapter.action_dim), (len(bank), adapter.latent_dim))
        for tensor, shape in zip((bank.contexts, bank.past_actions, bank.actions, bank.successors), expected_shapes):
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("Invalid cached tensor shape, precision, or values")
        return bank
    items = manifest["windows"][split]
    reader = WindowReader(manifest, normalization, adapter.image_size) if items else None
    contexts, past, actions, successors = [], [], [], []
    started = time.perf_counter()
    try:
        for start in range(0, len(items), batch_size):
            pixels, ordered_actions = reader.batch(items[start:start+batch_size])
            latent = adapter.encode(pixels).cpu()
            contexts.append(latent[:, :adapter.history])
            successors.append(latent[:, adapter.history])
            past.append(ordered_actions[:, :adapter.history-1])
            actions.append(ordered_actions[:, adapter.history-1])
            done = min(start + batch_size, len(items))
            if done == len(items) or done % (batch_size * 16) == 0:
                print(f"Cache {split}: {done}/{len(items)} windows | {time.perf_counter()-started:.1f}s", flush=True)
    finally:
        if reader is not None:
            reader.close()
    def merge(parts, shape):
        return torch.cat(parts).contiguous() if parts else torch.empty(shape, dtype=torch.float32)
    bank = LatentBank(split, merge(contexts, (0, adapter.history, adapter.latent_dim)),
        merge(past, (0, adapter.history-1, adapter.action_dim)),
        merge(actions, (0, adapter.action_dim)), merge(successors, (0, adapter.latent_dim)), items, identity)
    if bank.actions.shape[1] != adapter.action_dim:
        raise ValueError("Raw action width times frameskip differs from checkpoint action width")
    contents = {name: getattr(bank, name) for name in bank.__dataclass_fields__}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)
    return bank
