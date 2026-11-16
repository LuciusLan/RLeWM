"""Read completed query encodings without raw datasets or cache writes.

An explicit evaluation option trusts the *previous* sealed collection audit.
It never claims to re-audit missing raw data, and never regenerates a cache.
The original encoder functions stay untouched so existing cache IDs survive.
"""

from pathlib import Path
import hashlib
import inspect

from .runtime import file_hash, object_hash, read_json


def cache_only(config):
    value = config.get("cache_only", False)
    if type(value) is not bool:
        raise ValueError("cache_only must be a JSON boolean")
    return value


def checked(path):
    value = read_json(Path(path))
    if value.get("fingerprint") != object_hash({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Cache-only manifest checksum mismatch: {path}")
    return value


def validate_query_receipts(settings, query, memory_source):
    """Validate stored lineage; do not stat/open/hash any raw dataset payload."""
    if query.get("appearance"):
        raise ValueError("Cache-only appearance-derived queries need a dedicated parent-lineage bundle; use the raw-data mode")
    directory = Path(settings["manifest"]).resolve().parent
    audit = checked(directory/"audit.json")
    source = memory_source["source"]
    if (audit.get("passed") is not True or audit.get("query_manifest") != query["fingerprint"]
            or audit.get("collection_manifest") != query["collection_manifest"]
            or audit.get("source_dataset") != source["path"]
            or audit.get("source_size") != source["bytes"]
            or audit.get("source_mtime_ns") != source["mtime_ns"]):
        raise ValueError("Cache-only mode requires a passed stored query audit matching this query and P1 source manifest")
    # Standard fresh exports place queries under COLLECTION/REGIME/. Resolve
    # the copied metadata first, without rewriting its original absolute paths.
    collection_path = directory.parent/"manifest.json"
    if not collection_path.is_file():
        collection_path = Path(query["collection_manifest_path"])
    collection = checked(collection_path)
    if collection["fingerprint"] != query["collection_manifest"]:
        raise ValueError("Cache-only collection receipt belongs to another query cohort")
    if audit.get("regime") != query["regime"]:
        raise ValueError("Cache-only query audit belongs to another dynamics regime")
    if audit.get("backend") is not None and audit["backend"] != query.get("backend"):
        raise ValueError("Cache-only query audit belongs to another task")
    return {"mode": "cache-only", "source_raw_payloads_reverified": False,
            "query_raw_payloads_reverified": False,
            "prior_audit_fingerprint": audit["fingerprint"],
            "collection_manifest_fingerprint": collection["fingerprint"],
            "query_manifest_fingerprint": query["fingerprint"],
            "source_manifest_fingerprint": memory_source["fingerprint"]}


def validate_source_receipts(config, manifest, prepared):
    """Validate completion metadata before any model/GPU resource is loaded."""
    observed = prepared["memory_cache"]["observations"]
    root = Path(config["paths"]["cache_dir"])/f"observations-{observed['observation_fingerprint'][:16]}"
    status = read_json(root/"progress.json")
    actions = read_json(root/f"actions-{object_hash(prepared['normalization'])[:16]}.json")
    identity = status.get("identity", {})
    if (status.get("complete") is not True or status.get("encoded_rows") != observed["raw_frames"]
            or observed["raw_frames"] != manifest["source"]["rows"]
            or object_hash(identity) != observed["observation_fingerprint"]
            or identity.get("source") != manifest["source"]
            or identity.get("model") != prepared["model_fingerprint"]
            or status.get("latents_sha256") != observed["latents_sha256"]
            or actions.get("sha256") != observed["actions_sha256"]
            or actions.get("normalization") != prepared["normalization"]):
        raise ValueError("Cache-only P1 observation/action receipts are incomplete or do not match prepared.json")


def report_fields(resources, *, replay=False):
    if not getattr(resources, "cache_only", False):
        return {}
    return {"data_loading": {**resources.loading_provenance,
                             "query_raw_payloads_reverified": replay}}


def observation_identity(adapter, manifest):
    # This is exactly the original build_or_load_observations identity. Never
    # change the original FrameReader or encoder loop to enable read-only use.
    from .data import FrameReader
    from .observation_cache import _encode_observations
    return {"version": 1, "source": manifest["source"], "model": adapter.fingerprint,
            "latent_dim": adapter.latent_dim, "dtype": "float32",
            "frame_reader_sha256": hashlib.sha256(inspect.getsource(FrameReader).encode()).hexdigest(),
            "encoder_loop_sha256": hashlib.sha256(inspect.getsource(_encode_observations).encode()).hexdigest()}


def load_query_observations(adapter, manifest, normalization, directory):
    """Verify and map complete float32 arrays; no mkdir, locks, or fallback."""
    import numpy as np
    import torch
    from .observation_cache import ObservationStore
    identity = observation_identity(adapter, manifest)
    fingerprint = object_hash(identity)
    normalization_hash = object_hash(normalization)
    key = (str(Path(directory).resolve()), fingerprint, normalization_hash)
    stores = getattr(adapter, "_cache_only_observation_stores", {})
    if key in stores:
        return stores[key]
    target = Path(directory)/f"observations-{fingerprint[:16]}"
    latent_path = target/"latents.npy"
    actions_path = target/f"actions-{normalization_hash[:16]}.npy"
    required = (target/"progress.json", latent_path, actions_path, actions_path.with_suffix(".json"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cache-only loading requires existing complete latent AND normalized-action caches; "
                                "no encoding/rebuild is attempted. Missing: " + ", ".join(missing))
    status = read_json(target/"progress.json")
    rows = manifest["source"]["rows"]
    if (status.get("identity") != identity or status.get("complete") is not True
            or status.get("encoded_rows") != rows):
        raise ValueError("Cache-only observation cache is incomplete or its source/model/encoder identity changed")
    action_status = read_json(actions_path.with_suffix(".json"))
    if (file_hash(latent_path) != status.get("latents_sha256")
            or action_status.get("normalization") != normalization
            or file_hash(actions_path) != action_status.get("sha256")):
        raise ValueError("Cache-only latent/action checksum or source normalization mismatch")
    latents = np.load(latent_path, mmap_mode="c", allow_pickle=False)
    actions = np.load(actions_path, mmap_mode="c", allow_pickle=False)
    width = manifest["source"]["schema"]["action"]["shape"][-1]
    if (latents.shape != (rows, adapter.latent_dim) or actions.shape != (rows, width)
            or latents.dtype != np.float32 or actions.dtype != np.float32):
        raise ValueError("Cache-only observation/action shape or precision mismatch")
    store_identity = {"observation_fingerprint": fingerprint, "latents_sha256": status["latents_sha256"],
                      "normalization": normalization, "actions_sha256": action_status["sha256"],
                      "raw_frames": rows, "directory": str(target)}
    store = ObservationStore(torch.from_numpy(latents), torch.from_numpy(actions), store_identity, target)
    stores[key] = store
    adapter._cache_only_observation_stores = stores
    return store
