"""Freeze source statistics and trace shifted memory back to its source audit."""

from pathlib import Path

from ..collection.manifest import checked
from ..runtime import file_hash


def load_reference(config):
    reference = config.get("memory_reference")
    if reference is None:
        return None
    required = {"prepared", "prepared_sha256", "prepared_fingerprint", "source_manifest", "source_fingerprint",
                "memory_export", "memory_export_fingerprint"}
    if not isinstance(reference, dict) or set(reference) != required:
        raise ValueError("Invalid memory_reference; generate it with the shifted preparation workflow")
    if (not config["data"].get("memory_only") or config["data"].get("validation_episodes", 0)
            or config["data"].get("test_episodes", 0) or config["data"].get("split_from")):
        raise ValueError("Referenced shifted corpora must use memory-only preparation")
    path = Path(reference["prepared"])
    prepared = checked(path)
    source = checked(Path(reference["source_manifest"]))
    memory = checked(Path(reference["memory_export"]))
    if (file_hash(path) != reference["prepared_sha256"] or prepared["fingerprint"] != reference["prepared_fingerprint"]
            or source["fingerprint"] != reference["source_fingerprint"]
            or prepared["manifest"] != source["fingerprint"]
            or memory["fingerprint"] != reference["memory_export_fingerprint"]):
        raise ValueError("Frozen source statistics or shifted memory identity changed")
    if (memory.get("external_queries") or not memory["episodes"]["memory"]
            or memory["episodes"]["validation"] or memory["episodes"]["test"]
            or memory["appearance"]["parent_fingerprint"] != source["fingerprint"]
            or Path(memory["source"]["path"]).resolve() != Path(config["paths"]["dataset"]).resolve()):
        raise ValueError("Shifted memory must derive only from the declared original memory episodes")
    allowed_ids = {source["episode_metadata"]["ids"][p] for p in source["episodes"]["memory"]}
    ids = memory["episode_metadata"]["ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= allowed_ids:
        raise ValueError("Shifted retrieval pool includes non-memory or duplicate parent episode IDs")
    if any(config["data"][key] != source["data_config"][key] for key in ("history", "frameskip")):
        raise ValueError("Shifted/source temporal conventions differ")
    expected = {key: config["retrieval"][key] for key in ("calibration_pairs", "distance_floor")}
    if prepared["calibration_settings"] != expected:
        raise ValueError("Shifted memory must retain the declared source calibration settings")
    target = Path(memory["source"]["path"])
    stat = target.stat()
    if (stat.st_size != memory["source"]["bytes"] or stat.st_mtime_ns != memory["source"]["mtime_ns"]
            or file_hash(target) != memory["dataset_sha256"]):
        raise ValueError("Shifted memory HDF5 export changed")
    audit = checked(target.with_name("audit.json"))
    if (not audit.get("passed") or not audit.get("exact_labels_preserved")
            or audit.get("appearance_recipe") != memory["appearance"]["fingerprint"]
            or audit.get("parent_fingerprint") != source["fingerprint"]):
        raise ValueError("Run appearance check on shifted memory before encoding")
    return prepared


def original_source_path(config):
    if config.get("memory_reference") is None:
        return Path(config["paths"]["dataset"])
    load_reference(config)
    source = checked(Path(config["memory_reference"]["source_manifest"]))
    return Path(source["source"]["path"])


def validate_prepared_reference(config, prepared):
    if prepared.get("memory_reference") != config.get("memory_reference"):
        raise ValueError("Prepared memory uses a different normalization/calibration reference")
    reference = load_reference(config)
    if reference is not None and (prepared["normalization"] != reference["normalization"]
                                  or prepared["calibration"] != reference["calibration"]
                                  or prepared["model_fingerprint"] != reference["model_fingerprint"]):
        raise ValueError("Prepared shifted-memory model/statistics differ from the frozen source reference")
