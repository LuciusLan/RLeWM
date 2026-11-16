"""Strictly reuse P1 manifests, encodings, normalization, and cuVS key tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..adaptation import AdaptationEngine
from .. import frame_bank as frame_module
from ..frame_bank import FrameLatentBank
from ..model import LeWMAdapter
from ..observation_cache import ObservationStore
from ..cache_relocation import make_prepared_retriever as make_retriever
from ..runtime import file_hash, object_hash, read_json
from ..cache_loading import cache_only
from .config import effective_p1, scope_names
from .provenance import code_identity
from .scopes import ScopedLeWMAdapter, ScopeAdaptationEngine


def checked_json(path: Path) -> dict:
    value = read_json(path)
    if value.get("fingerprint") != object_hash({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Identity checksum mismatch: {path}")
    return value


def source_metadata(config: dict):
    root = Path(config["p1"]["paths"]["experiment_dir"])
    manifest = checked_json(root / "manifest.json")
    prepared = checked_json(root / "prepared.json")
    if config["p1"].get("memory_reference") is not None or prepared.get("memory_reference") is not None:
        if cache_only(config):
            raise ValueError("Cache-only shifted-memory references need a dedicated lineage bundle; use the raw-data mode")
        from ..shifted.reference import validate_prepared_reference
        validate_prepared_reference(config["p1"], prepared)
    if manifest["version"] != 2 or manifest["data_config"] != config["p1"]["data"]:
        raise ValueError("P1 config must match its prepared full-cache manifest")
    if prepared["manifest"] != manifest["fingerprint"]:
        raise ValueError("Prepared P1 data belongs to a different manifest")
    if manifest["source"]["path"] != config["p1"]["paths"]["dataset"]:
        raise ValueError("P1 source path differs from its cached provenance")
    assigned = [set(manifest["episodes"][split]) for split in ("memory", "validation", "test")]
    if any(assigned[a] & assigned[b] for a in range(3) for b in range(a+1, 3)):
        raise ValueError("P1 source episode partitions overlap")
    if config["p1"]["retrieval"]["backend"] == "cuvs" and not prepared.get("retrieval_assets"):
        raise ValueError("P1 cuVS assets must already be prepared")
    return manifest, prepared


def require_existing_cache(config: dict, manifest: dict, prepared: dict) -> None:
    """P2 must not turn a missing observation asset into a full re-encoding job."""
    directory = Path(config["paths"]["cache_dir"])
    observed = prepared["memory_cache"]["observations"]
    frames = directory / f"observations-{observed['observation_fingerprint'][:16]}"
    actions = f"actions-{object_hash(prepared['normalization'])[:16]}"
    windows = directory / f"windows-{manifest['fingerprint'][:16]}"
    required = [frames / "latents.npy", frames / "progress.json", frames / f"{actions}.npy",
                frames / f"{actions}.json"]
    required.extend(windows / f"{split}-{suffix}" for split in ("memory", "validation", "test")
                    for suffix in ("starts.npy", "index.json"))
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("P1 cache assets are missing; restore/complete P1 preparation before P2")
    if not read_json(frames / "progress.json").get("complete"):
        raise ValueError("P1 observation cache is incomplete; P2 will not resume its encoding")


def cached_observations(config: dict, prepared: dict, adapter) -> ObservationStore:
    identity = prepared["memory_cache"]["observations"]
    root = Path(config["paths"]["cache_dir"]) / f"observations-{identity['observation_fingerprint'][:16]}"
    latent_path = root / "latents.npy"
    action_path = root / f"actions-{object_hash(prepared['normalization'])[:16]}.npy"
    if (file_hash(latent_path) != identity["latents_sha256"]
            or file_hash(action_path) != identity["actions_sha256"]
            or identity["normalization"] != prepared["normalization"]):
        raise ValueError("P1 cached observation/action contents changed")
    latents = np.load(latent_path, mmap_mode="c", allow_pickle=False)
    actions = np.load(action_path, mmap_mode="c", allow_pickle=False)
    if (latents.shape != (identity["raw_frames"], adapter.latent_dim)
            or actions.ndim != 2 or len(actions) != len(latents)
            or latents.dtype != np.float32 or actions.dtype != np.float32):
        raise ValueError("Invalid cached observation/action shape or dtype")
    return ObservationStore(torch.from_numpy(latents), torch.from_numpy(actions), identity, root).to(adapter.device)


def cached_bank(config: dict, manifest: dict, prepared: dict, store, split: str):
    expected = prepared[{"memory": "memory_cache", "validation": "validation_cache", "test": "test_cache"}[split]]
    directory = Path(config["paths"]["cache_dir"]) / f"windows-{manifest['fingerprint'][:16]}"
    path = directory / f"{split}-starts.npy"
    metadata = read_json(directory / f"{split}-index.json")
    starts = np.load(path, mmap_mode="c", allow_pickle=False)
    if starts.shape != (expected["windows"],) or starts.dtype != np.int64:
        raise ValueError("Invalid cached window index")
    identity = {"version": 2, "layout": "shared observed frames and lazy window views",
                "manifest": manifest["fingerprint"], "split": split, "windows": len(starts),
                "observations": store.identity, "window_index_sha256": file_hash(path),
                "view_implementation_sha256": file_hash(Path(frame_module.__file__))}
    bank_class = FrameLatentBank
    if manifest["data_config"].get("dataset_id"):
        from ..preprocessing.banks import SourceFrameLatentBank, namespace_identity
        identity.update(namespace_identity())
        bank_class = SourceFrameLatentBank
    if (identity != expected or metadata["sha256"] != identity["window_index_sha256"]
            or metadata["manifest"] != manifest["fingerprint"]
            or metadata["specification"] != manifest["windows"][split]):
        raise ValueError("P1 cached window identity/implementation changed")
    return bank_class(store, manifest, split, starts, identity)


@dataclass
class Resources:
    config: dict
    adapter: LeWMAdapter
    manifest: dict
    prepared: dict
    memory: FrameLatentBank
    engine: AdaptationEngine
    _banks: dict = field(default_factory=dict, repr=False)
    external_manifest: dict | None = field(default=None, repr=False)
    external_settings: dict | None = field(default=None, repr=False)
    cache_only: bool = False
    loading_provenance: dict = field(default_factory=dict, repr=False)

    def queries(self, split: str):
        if split not in ("validation", "test"):
            raise ValueError("Queries must come from validation or test")
        if split not in self._banks:
            if self.external_manifest is None:
                self._banks[split] = cached_bank(self.config, self.manifest, self.prepared, self.memory.store, split)
            else:
                from ..collection.queries import build_query_bank
                self._banks[split] = build_query_bank(self, split)
        return self._banks[split]


def load_resources(config: dict, device: str, cpu_threads: int) -> Resources:
    p1 = effective_p1(config)
    manifest, prepared = source_metadata(config)
    if p1["retrieval"]["backend"] == "cuvs" and device != "cuda":
        raise ValueError("The full prepared cuVS source requires --device cuda")
    calibration = {key: p1["retrieval"][key] for key in ("calibration_pairs", "distance_floor")}
    if calibration != prepared["calibration_settings"]:
        raise ValueError("Source calibration settings changed")
    require_existing_cache(p1, manifest, prepared)
    if cache_only(config):
        from ..cache_loading import validate_source_receipts
        validate_source_receipts(p1, manifest, prepared)
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(p1["data"]["split_seed"])
    adapter = ScopedLeWMAdapter(Path(p1["paths"]["repo"]), Path(p1["paths"]["checkpoint"]),
                               device, scope_names(config)[0])
    if p1["data"]["history"] != adapter.history or prepared["model_fingerprint"] != adapter.fingerprint:
        raise ValueError("Prepared model/encoder identity changed")
    store = cached_observations(p1, prepared, adapter)
    if store.actions.shape[1]*manifest["data_config"]["frameskip"] != adapter.action_dim:
        raise ValueError("Cached raw actions have an incompatible ordered-block width")
    memory = cached_bank(p1, manifest, prepared, store, "memory")
    retriever = make_retriever(memory, prepared["calibration"], p1["retrieval"], Path(p1["paths"]["cache_dir"]))
    if p1["retrieval"]["backend"] == "cuvs" and retriever.assets["fingerprint"] != prepared["retrieval_assets"]["fingerprint"]:
        raise ValueError("Prepared cuVS key identity changed")
    engine = ScopeAdaptationEngine(adapter, memory, retriever, p1["adaptation"])
    result = Resources(p1, adapter, manifest, prepared, memory, engine, cache_only=cache_only(config))
    if result.cache_only:
        result.loading_provenance = {"mode": "cache-only", "source_raw_payloads_reverified": False,
                                     "source_manifest_fingerprint": manifest["fingerprint"]}
        print("Cache-only loading: verify prepared arrays and reuse the prior sealed audit; no raw-source re-audit or encoding.", flush=True)
    if config.get("query_dataset"):
        from ..collection.queries import query_source
        result.external_manifest = query_source(config, manifest)
        result.external_settings = config["query_dataset"]
        if result.cache_only:
            from ..cache_loading import validate_query_receipts
            result.loading_provenance = validate_query_receipts(result.external_settings, result.external_manifest, manifest)
    return result
