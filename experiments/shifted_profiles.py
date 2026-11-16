"""Standalone stage profiles for paired source/shifted memory comparisons."""

from copy import deepcopy
from pathlib import Path

from lewm_tta.collection.manifest import checked
from lewm_tta.config import load_config as load_p1
from lewm_tta.runtime import file_hash, read_json, write_json
from .catalog import ROOT, profile
from .config import read_profile, resolve
from .fresh import JOBS


def write_once(path, value, dry_run=False):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing shifted configuration differs; choose a new --output-root: {path}")
    elif not dry_run:
        write_json(path, value)


def context(args, suite, name):
    condition = suite["conditions"][name]
    output = (ROOT/(args.output_root or suite["output_root"])/name).resolve()
    previous = read_json(output/"run-context.json") if (output/"run-context.json").exists() else {}
    raw_root = (ROOT/(args.raw_root or previous.get("raw_root", suite["raw_root"]))).resolve()
    cache_root = (ROOT/(args.cache_root or previous.get("cache_root", suite["cache_root"]))).resolve()
    p1 = load_p1((ROOT/condition["p1_config"]).resolve())
    source_root = Path(p1["paths"]["experiment_dir"])
    source_manifest = checked(source_root/"manifest.json")
    source_prepared = checked(source_root/"prepared.json")
    if source_manifest.get("version") != 2 or p1["data"]["memory_windows"] != "all":
        raise ValueError("Shifted evaluation requires the existing full-cache P1 source profile")
    if p1["retrieval"]["backend"] == "cuvs" and not source_prepared.get("retrieval_assets"):
        raise ValueError("Complete source P1 cuVS preparation before shifted evaluation")
    if source_prepared["manifest"] != source_manifest["fingerprint"]:
        raise ValueError("Prepare the original source P1 corpus before shifted evaluation")
    if p1["data"] != source_manifest["data_config"]:
        raise ValueError("Original P1 profile differs from its prepared manifest")
    collection = read_profile(ROOT/condition["collection_config"])
    if (collection["backend"] != condition["dataset"]
            or (ROOT/collection["reference_dataset"]).resolve() != Path(p1["paths"]["dataset"]).resolve()):
        raise ValueError("Condition collection and source P1 profiles refer to different tasks/datasets")
    learning_rate = args.learning_rate if args.learning_rate is not None else previous.get("learning_rate", condition["learning_rate"])
    seeds = args.seeds if args.seeds is not None else previous.get("seeds", condition["seeds"])
    p1["evaluation"]["seeds"] = list(seeds)
    settings = {"object_color": None, "agent_color": None, "goal_color": None, "background_color": None,
                "background_color2": None, "background_pattern": "preserve", "pattern_size": 64,
                **condition["appearance"]}
    paths = {"raw_root": str(raw_root), "cache_root": str(cache_root),
             "memory": str(raw_root/f"{name}-memory"), "queries": str(raw_root/f"{name}-queries"),
             "query_parent": str((ROOT/collection["output_dir"] / "source/queries.json").resolve()),
             "query_cache": str(cache_root/name/"queries"), "memory_cache": str(cache_root/name/"memory"),
             "p1_source": str(output/"configs/p1-source.json"), "p1_shifted": str(output/"configs/p1-shifted.json"),
             "source_manifest": str(source_root/"manifest.json"), "source_prepared": str(source_root/"prepared.json"),
             "p1_output": str(output/"p1-memory"), "output": str(output)}
    binding = {"version": 1, "condition": name, "dataset": condition["dataset"], "raw_root": str(raw_root),
               "cache_root": str(cache_root), "settings": settings, "learning_rate": learning_rate, "seeds": seeds,
               "encode_batch_size": args.encode_batch_size or previous.get("encode_batch_size", 64),
               "p2_schedules": args.p2_schedules or previous.get("p2_schedules", suite["p2_schedules"]),
               "p2_methods": args.p2_methods or previous.get("p2_methods", suite["p2_methods"]),
               "source_manifest_fingerprint": source_manifest["fingerprint"],
               "source_prepared_fingerprint": source_prepared["fingerprint"]}
    if previous and previous != binding:
        raise ValueError(f"Shifted condition settings/paths changed; use a new --output-root: {output}")
    return {"name": name, "condition": condition, "paths": paths, "settings": settings, "binding": binding,
            "source_p1": p1, "source_manifest": source_manifest, "source_prepared": source_prepared}


def exported(ctx, role, required=True):
    directory = Path(ctx["paths"][role])
    path = directory/("queries.json" if role == "queries" else "episodes.json")
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Complete shifted synthesis/preparation first: {path}")
        return None
    value = checked(path)
    if value["backend"] != ctx["condition"]["dataset"] or value["appearance"]["settings"] != ctx["settings"]:
        raise ValueError(f"Shifted dataset appearance differs from the selected condition: {path}")
    parent = checked(Path(ctx["paths"]["query_parent"] if role == "queries" else ctx["paths"]["source_manifest"]))
    if (value["appearance"]["parent_fingerprint"] != parent["fingerprint"]
            or bool(value.get("external_queries")) != (role == "queries")):
        raise ValueError(f"Shifted dataset derives from another parent/role: {path}")
    if role == "queries":
        for split in ("validation", "test"):
            expected = [parent["episode_metadata"]["ids"][i] for i in parent["episodes"][split]]
            actual = [value["episode_metadata"]["ids"][i] for i in value["episodes"][split]]
            if actual != expected or value["evaluation_windows"][split]["count"] != parent["evaluation_windows"][split]["count"]:
                raise ValueError("Full shifted evaluation requires the complete reserved fresh cohort, not a synthesis pilot")
    return value


def make_profiles(ctx, args, suite, dry_run=False):
    paths = ctx["paths"]
    output = Path(paths["output"])
    query = exported(ctx, "queries", required=not dry_run)
    memory = exported(ctx, "memory", required=not dry_run) if "shifted" in args.memory else None
    if memory is not None and "source" in args.memory:
        original = ctx["source_manifest"]
        ids = [original["episode_metadata"]["ids"][p] for p in original["episodes"]["memory"]]
        if memory["episode_metadata"]["ids"] != ids:
            raise ValueError("Paired source/shifted memory requires identical parent episodes; use --memory shifted for a subset-only study")
    write_once(output/"run-context.json", ctx["binding"], dry_run)
    write_once(Path(paths["p1_source"]), ctx["source_p1"], dry_run)
    if "shifted" in args.memory:
        p1 = deepcopy(ctx["source_p1"])
        p1["paths"].update(dataset=str(Path(paths["memory"])/"dataset.h5"), experiment_dir=paths["p1_output"], cache_dir=paths["memory_cache"])
        p1["data"].update(memory_only=True, memory_windows="all", validation_episodes=0, test_episodes=0,
                          historical_inspected_episodes=[], inspected_reports=[], inspected_report_globs=[], split_from=None)
        p1["preprocessing"] = {"raw_action_dim": ctx["source_manifest"]["source"]["schema"]["action"]["shape"][-1]}
        p1["memory_reference"] = {"prepared": paths["source_prepared"], "prepared_sha256": file_hash(Path(paths["source_prepared"])),
            "prepared_fingerprint": ctx["source_prepared"]["fingerprint"], "source_manifest": paths["source_manifest"],
            "source_fingerprint": ctx["source_manifest"]["fingerprint"],
            "memory_export": str(Path(paths["memory"])/"episodes.json"),
            "memory_export_fingerprint": memory["fingerprint"] if memory else None}
        write_once(Path(paths["p1_shifted"]), p1, dry_run)
        ctx["shifted_p1"] = p1
    jobs = []
    for memory_kind in args.memory:
        for name in args.jobs:
            stage, variant, backend, report_stage = JOBS[name]
            if name == "p2":
                variant = "fresh-temporal"
            config = resolve(profile(stage, variant)[1], dataset=ctx["condition"]["dataset"])
            config["p1_config"] = paths["p1_source" if memory_kind == "source" else "p1_shifted"]
            config["experiment_dir"] = str(output/memory_kind/name)
            config["query_dataset"] = {"manifest": str(Path(paths["queries"])/"queries.json"),
                                       "cache_dir": paths["query_cache"], "batch_size": ctx["binding"]["encode_batch_size"]}
            config["adaptation"]["learning_rate"] = ctx["binding"]["learning_rate"]
            config["evaluation"].update(query_limit=None, records="compact")
            config["appearance_condition"] = ctx["name"]
            config["memory_condition"] = memory_kind
            if "reuse" in config:
                config["reuse"] = {}
            if name == "p2":
                config["adaptation"]["methods"] = list(ctx["binding"]["p2_methods"])
                schedules = ctx["binding"]["p2_schedules"]
                if len(schedules) != len(set(schedules)) or set(schedules)-set(config["schedules"]):
                    raise ValueError("Choose distinct P2 schedules from the fresh-temporal catalog")
                config["schedule_names"] = list(schedules)
            if name == "temporal":
                config["evaluation"]["seeds"] = list(ctx["binding"]["seeds"])
                config["evaluation"].pop("seeds_by_mode", None)
            path = output/"configs"/f"{memory_kind}-{name}.json"
            write_once(path, config, dry_run)
            jobs.append({"name": name, "memory": memory_kind, "stage": stage, "backend": backend,
                         "report_stage": report_stage, "config": config, "config_path": path})
    return jobs, query
