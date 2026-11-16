"""External query banks share the existing frozen memory, normalization and index."""

from pathlib import Path

from ..config import ROOT
from ..runtime import file_hash
from .manifest import checked
from ..cache_loading import cache_only, validate_query_receipts


def query_source(config, memory_source):
    settings = config.get("query_dataset")
    if settings is None:
        return memory_source
    if set(settings)-{"manifest", "cache_dir", "batch_size"}:
        raise ValueError("query_dataset accepts manifest, cache_dir and batch_size only")
    if not {"manifest", "cache_dir"} <= set(settings) or type(settings.get("batch_size", 128)) is not int or settings.get("batch_size", 128) < 1:
        raise ValueError("Provide query manifest/cache_dir and a positive integer batch_size")
    value = checked((ROOT/settings["manifest"]).resolve())
    from .contracts import validate_query_task
    validate_query_task(config, memory_source, value)
    if not value.get("external_queries") or value["episodes"]["memory"]:
        raise ValueError("Fresh query data must contain evaluation episodes only")
    if any(value["data_config"][key] != memory_source["data_config"][key] for key in ("history", "frameskip")):
        raise ValueError("Query and checkpoint/memory temporal conventions disagree")
    if value.get("appearance") and not cache_only(config):
        from ..appearance.pipeline import validate_query_lineage
        validate_query_lineage(value)
    if "goal_offset_steps" in value and "planning" in config:
        if value["goal_offset_steps"] != config["planning"]["horizon"]*value["data_config"]["frameskip"]:
            raise ValueError("Offline P3 uses a horizon*frameskip goal; use a compatible primary query manifest. Alternative MPC goals are declared separately in goal-protocols.json")
    if cache_only(config):
        validate_query_receipts(settings, value, memory_source)
        return value
    verify_query_payloads(value)
    source = Path(value["source"]["path"])
    audit = checked(source.with_name("audit.json"))
    from ..shifted.reference import original_source_path
    original = original_source_path(config["p1"])
    if (not audit["passed"] or audit["query_manifest"] != value["fingerprint"]
            or audit["source_dataset"] != str(original.resolve())
            or audit["source_size"] != original.stat().st_size
            or audit["source_mtime_ns"] != original.stat().st_mtime_ns):
        raise ValueError("Run collect check against the current released source before preparing fresh queries")
    return value


def verify_query_payloads(value):
    """Raw-data validation is still mandatory for simulator replay."""
    source = Path(value["source"]["path"])
    missing = [path for path in (source, *(Path(ep["path"]) for ep in value["episode_files"]))
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fresh raw HDF5/episode payloads are required in raw-data mode and for simulator replay; "
                                "offline P2/P3 may use --cache-only. Missing: " + str(missing[0]))
    if (source.stat().st_size != value["source"]["bytes"] or source.stat().st_mtime_ns != value["source"]["mtime_ns"]
            or file_hash(source) != value["dataset_sha256"]):
        raise ValueError("Fresh HDF5 view changed")
    for episode in value["episode_files"]:
        if file_hash(Path(episode["path"])) != episode["sha256"]:
            raise ValueError("Fresh episode payload changed")


def namespace_bank(bank):
    """Keep local array offsets but isolate external episode IDs from memory IDs."""
    if not bank.manifest.get("external_queries"):
        return bank
    original = bank.items
    class ExternalItems:
        offsets = original.offsets
        starts = original.starts
        episode_ids = original.episode_ids

        def __len__(self):
            return len(original)

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self[position] for position in range(*index.indices(len(self)))]
            item = dict(original[index])
            item["local_episode_position"] = item["source_episode_position"]
            item["source_episode_position"] = -int(item["episode_id"])-1
            item["transition_id"] = f"{bank.manifest['regime']}:{bank.manifest['collection_manifest'][:16]}:{item['episode_id']}:{item['start_in_episode']}"
            item["dynamics_regime_id"] = bank.manifest["regime"]
            item["collection_manifest"] = bank.manifest["collection_manifest"]
            return item
    bank.items = ExternalItems()
    return bank


def build_query_bank(resources, split):
    import numpy as np
    from ..frame_bank import FrameLatentBank
    from ..observation_cache import build_or_load_observations
    settings = resources.external_settings
    manifest = resources.external_manifest
    directory = (ROOT/settings["cache_dir"]).resolve()
    # Only new images are encoded; original normalization and all memory/index
    # files remain unchanged. The existing encoder freezes its BN buffers.
    if getattr(resources, "cache_only", False):
        from ..cache_loading import load_query_observations
        store = load_query_observations(resources.adapter, manifest, resources.prepared["normalization"], directory)
    else:
        store = build_or_load_observations(resources.adapter, manifest, resources.prepared["normalization"],
                                           directory, settings.get("batch_size", 128), 4096)
    store = store.to(resources.adapter.device)
    starts = np.asarray(manifest["evaluation_windows"][split]["raw_start_rows"], dtype=np.int64)
    bank = FrameLatentBank(store, manifest, split, starts,
                          {"external_queries": manifest["fingerprint"], "observations": store.identity})
    return namespace_bank(bank)
