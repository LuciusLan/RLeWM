"""Load relocated prepared keys without rewriting their scientific identities.

Historical index manifests contain absolute payload paths. The current cache
root chooses the physical files; original metadata and payload hashes continue
to identify their contents. Key generation and numerical retrieval stay intact.
"""

from pathlib import Path

from .runtime import file_hash, object_hash, read_json


def index_array_paths(directory, record):
    """Resolve canonical payloads beneath the configured root, never the old one."""
    expected = object_hash({"identity": record["identity"], "arrays": record["arrays"]})
    if record["fingerprint"] != expected:
        raise ValueError("Prepared cuVS manifest fingerprint changed")
    root = Path(directory)/"retrieval"/f"cuvs-{object_hash(record['identity'])[:16]}"
    if set(record["arrays"]) != {"forward", "reverse"}:
        raise ValueError("Prepared cuVS manifest must contain forward/reverse arrays")
    paths = {}
    for direction, entry in record["arrays"].items():
        if Path(entry["path"]).name != f"{direction}.npy":
            raise ValueError("Unexpected prepared cuVS payload name")
        path = root/f"{direction}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared cuVS keys in the configured cache: {path}. "
                                    "Move the entire retrieval/cuvs-* directory with the cache.")
        if path.stat().st_size != entry["file_bytes"]:
            raise ValueError(f"Prepared cuVS payload size changed: {path}")
        paths[direction] = path
    return paths


def make_prepared_retriever(memory, calibration, settings, directory):
    """Use the existing retriever, resolving old absolute paths only when needed."""
    from .retrieval import make_retriever
    if settings.get("backend", "torch") != "cuvs" or not len(memory):
        return make_retriever(memory, calibration, settings, directory)
    from .indexed_keys import KeyLayout, key_identity
    layout = KeyLayout.from_memory(memory, calibration, settings)
    identity = key_identity(memory, layout)
    root = Path(directory)/"retrieval"/f"cuvs-{object_hash(identity)[:16]}"
    record = read_json(root/"index.json")
    if record["identity"] != identity:
        raise ValueError("Prepared cuVS key identity differs from memory/scales/weights")
    paths = index_array_paths(directory, record)
    if all(Path(entry["path"]).resolve() == paths[direction].resolve()
           for direction, entry in record["arrays"].items()):
        return make_retriever(memory, calibration, settings, directory)

    import numpy as np
    import torch
    from .cuvs_retrieval import CuVSRetriever
    # Preserve the exact original record/fingerprint: only file locations differ.
    # Validation remains identical to the ordinary loader, including full hashes.
    for direction, path in paths.items():
        entry = record["arrays"][direction]
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (list(values.shape) != entry["shape"] or values.dtype != np.float32
                or file_hash(path) != entry["sha256"]):
            raise ValueError(f"Invalid relocated {direction} cuVS payload: {path}")
        del values
    # Explicit empty tables construct the same native runtime without opening
    # the historical paths. Populate its normal key/index maps after validation.
    retriever = CuVSRetriever(memory, calibration, settings, directory, tables={})
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(retriever.runtime.device)
    reserve = int(settings.get("device_reserve_gib", 2)*2**30)
    required = record["numeric_payload_bytes"]
    if required+reserve > free:
        raise MemoryError(f"cuVS keys need {required/2**30:.2f} GiB plus {reserve/2**30:.2f} GiB "
                          f"reserve; {free/2**30:.2f} GiB is free")
    retriever.keys = {direction: torch.from_numpy(np.load(path, mmap_mode="c", allow_pickle=False))
                      .to(retriever.runtime.device) for direction, path in paths.items()}
    retriever.indexes = {direction: retriever.runtime.build(keys) for direction, keys in retriever.keys.items()}
    retriever.assets = record
    retriever.resolved_asset_paths = {direction: str(path) for direction, path in paths.items()}
    print(f"Using relocated cuVS keys: {root}", flush=True)
    return retriever
