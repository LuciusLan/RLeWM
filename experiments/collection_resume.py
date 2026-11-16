"""Resume immutable collection provenance when unrelated simulator code evolves."""

from pathlib import Path

from lewm_tta.collection.manifest import checked
from lewm_tta.runtime import object_hash, read_json
from .catalog import ROOT


def recorded_collection_code(config):
    if config is None:
        return None
    root = (ROOT/config["output_dir"]).resolve()
    identities = []
    fingerprints = set()
    # Failed attempts also pin source, even if no accepted episode exists yet.
    for regime in config["regimes"]:
        for path in sorted((root/regime/"episodes").glob("*.json")):
            value = checked(path)
            if value.get("code"):
                identities.append(value["code"])
                fingerprints.add(object_hash(value["code"]))
        for path in sorted((root/regime/"attempts").glob("*/*.json")):
            fingerprints.add(checked(path)["code_fingerprint"])
    if len(fingerprints) > 1:
        raise ValueError("Collection contains mixed execution-source identities")
    if identities:
        if any(code != identities[0] for code in identities[1:]):
            raise ValueError("Collection contains mixed execution-source identities")
        return identities[0]
    if fingerprints:
        for path in sorted((ROOT/"artifacts/code-archives").glob("*/manifest.json")):
            code = read_json(path).get("entrypoints", {}).get("collect", {}).get("code")
            if code and object_hash(code) in fingerprints:
                return code  # archived_command subsequently verifies every source file.
        raise ValueError("Attempt-only collection has no archived execution source")
    return None
