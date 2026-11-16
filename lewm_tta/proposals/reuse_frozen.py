"""Model-free verification of Frozen cells in completed proposal reports.

Only unmodified pretrained-model searches are eligible. Gaussian cells do not
depend on fragment settings; injected cells require those settings to match.
The numerical evaluation loop still checks runtime proposals, resets and costs.
"""

from itertools import product
import json
from pathlib import Path

from ..runtime import file_hash
from ..rollout.record_io import open_records
from ..collection.manifest import checked


PROPOSAL_STAGES = ("P3b-offline-proposals", "P3c-offline-factorial")
NUMERICAL_SOURCES = (
    "model.py", "adaptation.py", "retrieval.py", "cuvs_retrieval.py", "indexed_keys.py",
    "frame_bank.py", "observation_cache.py", "rollout/scopes.py", "rollout/support_updates.py",
    "rollout/inference.py", "rollout/cohorts.py", "planning/actions.py", "planning/cem.py",
    "planning/shared_ttt.py", "planning/cohorts.py", "proposals/cem.py",
    "proposals/fragments.py", "proposals/injection.py", "proposals/context_search.py",
    "proposals/providers.py", "proposals/goal_search.py",
)


def check_frozen_protocol(source, config, split, manifest):
    if (source.get("stage") not in PROPOSAL_STAGES or not source.get("completed")
            or source.get("limited_run") or source.get("split") != split):
        raise ValueError("Frozen reuse requires a completed full proposal report from the same split")
    if config["adaptation"]["methods"] != ["frozen"]:
        raise ValueError("Proposal-report reuse is restricted to Frozen-only action ablations")
    previous = source["config"]
    for field in ("planning", "scopes", "query_selection", "query_dataset"):
        if previous.get(field) != config.get(field):
            raise ValueError(f"Reused Frozen protocol differs: {field}")
    for field in ("paths", "data", "retrieval"):
        if previous["p1"][field] != config["p1"][field]:
            raise ValueError(f"Reused Frozen memory/checkpoint differs: p1.{field}")
    for field, value in config["adaptation"].items():
        if field != "methods" and previous["adaptation"].get(field) != value:
            raise ValueError(f"Reused Frozen setup differs: adaptation.{field}")
    original_manifest = checked(Path(previous["experiment_dir"]) / "manifest.json")
    if original_manifest["fingerprint"] != source["manifest_fingerprint"]:
        raise ValueError("Reused proposal source manifest changed")
    if (source["p1_prepared_fingerprint"] != manifest["p1_prepared_fingerprint"]
            or source["cohort"] != manifest["splits"][split]
            or original_manifest.get("query_source_fingerprint") != manifest.get("query_source_fingerprint")):
        raise ValueError("Reused Frozen preparation/query cohort differs")
    if (source["summary"]["queries"] != source["cohort"]["queries"]
            or source["summary"]["episodes"] != source["cohort"]["episodes"]
            or not source.get("checks") or any(value is False for value in source["checks"].values())
            or not all(source["summary"]["pairing_checks"].values())):
        raise ValueError("Reused Frozen source did not pass its full-cohort pairing checks")


def check_frozen_sources(source, path):
    from experiments.catalog import ROOT
    from experiments.provenance import archive_entry, backend_identity

    workspace = next((parent for parent in path.parents
                      if (parent / "artifacts/code-archives").is_dir()), ROOT)
    if (source.get("code") != backend_identity("p3b")
            and archive_entry("p3b", source.get("code"), root=workspace) is None):
        raise ValueError("Reused Frozen source has no verified execution archive")
    for name in NUMERICAL_SOURCES:
        relative = f"codes/lewm_tta/{name}"
        if source["code"].get(relative) != file_hash(ROOT / relative):
            raise ValueError(f"Reused Frozen numerical implementation changed: {relative}")


def reusable_sources(source, config):
    previous = source["config"]["proposals"]
    same_injection = ({key: value for key, value in previous.items() if key != "sources"}
                      == {key: value for key, value in config["proposals"].items() if key != "sources"})
    return {name for name in config["proposals"]["sources"]
            if name in previous["sources"] and (name == "gaussian" or same_injection)}


def load_frozen_cells(path, source, config, split, manifest):
    """Verify the complete source matrix, retaining just eligible Frozen cells."""
    path = Path(path).resolve()
    check_frozen_protocol(source, config, split, manifest)
    check_frozen_sources(source, path)
    if source.get("reuse", {}).get("source_reports"):
        from .reuse import verify_dependencies
        verify_dependencies(source)
    records = Path(source.get("records_file") or "")
    if (not records.is_file() or records.parent.resolve() != path.parent
            or not records.stat().st_size or file_hash(records) != source.get("records_sha256")):
        raise ValueError("Reused Frozen records are missing, empty or changed")
    previous = source["config"]
    seeds = config["p1"]["evaluation"]["seeds"]
    if not set(seeds) <= set(previous["p1"]["evaluation"]["seeds"]):
        raise ValueError("Reused Frozen source does not cover every requested seed")
    eligible, seen, rows, episodes, ids = reusable_sources(source, config), set(), {}, set(), set()
    with open_records(records) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["query"]["transition_id"], row["seed"], row["method"], row["proposal_source"]
            if key in seen:
                raise ValueError("Duplicate records in reused proposal source")
            seen.add(key)
            ids.add(key[0])
            episodes.add(row["query"]["source_episode_position"])
            if key[2] == "frozen" and key[1] in seeds and key[3] in eligible:
                if row["adaptation"]["updates"] != 0 or row["adaptation"]["parameter_displacement_mse"] != 0:
                    raise ValueError("Reused Frozen row contains parameter adaptation")
                rows[key] = row
    matrix = set(product(ids, previous["p1"]["evaluation"]["seeds"],
                         previous["adaptation"]["methods"], previous["proposals"]["sources"]))
    if (seen != matrix or len(seen) != source["record_count"]
            or len(ids) != source["cohort"]["queries"] or len(episodes) != source["cohort"]["episodes"]):
        raise ValueError("Reused proposal report has an incomplete or wrong-cohort matrix")
    if config.get("query_dataset"):
        from experiments.fresh import query_ids
        query = checked(Path(config["query_dataset"]["manifest"]))
        if ids != query_ids(query, manifest["splits"][split]["raw_start_rows"]):
            raise ValueError("Reused proposal query IDs differ from the declared fresh starts")
    if set(rows) != set(product(ids, seeds, ["frozen"], eligible)):
        raise ValueError("Reused Frozen cells are incomplete")
    for (query, seed, _, _), row in rows.items():
        baseline = rows[query, seed, "frozen", "gaussian"]
        for field in ("query", "scope", "current_raw_row", "goal_raw_row", "target_raw_rows",
                      "provisional_seed", "provisional_fingerprint", "cem_seed", "base_first_population_fingerprint",
                      "recorded_action_mse_by_block"):
            if row[field] != baseline[field]:
                raise ValueError(f"Reused Frozen proposal pairing differs: {field}")
    return rows
