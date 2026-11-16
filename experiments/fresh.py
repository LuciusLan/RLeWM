"""One workflow for fresh recorded-action and offline planning comparisons.

Orchestration and readiness checks are model-free. Numerical work always goes
through the existing stage CLI, including its validation locks and source replay.
"""

import argparse
from bisect import bisect_right
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.collection.manifest import checked
from lewm_tta.config import load_config as load_p1
from lewm_tta.rollout.config import effective_p1
from lewm_tta.rollout.record_io import read_saved_records
from lewm_tta.runtime import file_hash, object_hash, read_json
from .catalog import ROOT, profile
from .config import materialize, query_dataset_settings, read_profile, resolve
from .provenance import archive_entry, backend_identity
from .datasets import DATASETS, dataset_profile


JOBS = {"p2": ("p2", "fresh", "p2_followup", "P2-followup-iteration"),
        "p3": ("p3", "fresh", "p3c", "P3c-offline-factorial"),
        "temporal": ("p3", "fresh-temporal", "p3_temporal", "P3-temporal-cem")}


def contains(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and contains(value, actual[key])
                                                for key, value in expected.items())
    return expected == actual


def effective_config(config):
    value = deepcopy(config)
    value["experiment_dir"] = str((ROOT / value["experiment_dir"]).resolve())
    value["p1"] = load_p1(Path(value["p1_config"]))
    if "planning" in value:
        value["adaptation"] = effective_p1(value)["adaptation"]
    if value.get("comparison") == "temporal-planning" and "seeds" in value["evaluation"]:
        value["p1"]["evaluation"]["seeds"] = list(value["evaluation"]["seeds"])
    return value


def collection_status(settings, collection_path, regime, *, cache_only=False, memory_source=None):
    collection = read_profile(collection_path)
    directory = (ROOT / collection["output_dir"] / regime).resolve()
    query_path = Path(settings["manifest"])
    if not query_path.is_file():
        raise FileNotFoundError(f"Complete collection/export first: {query_path}")
    query = checked(query_path)
    reserved = checked(directory.parent / "manifest.json")
    if query["collection_manifest"] != reserved["fingerprint"] or query["regime"] != regime:
        raise ValueError("Collection reservation/export identities differ")
    configured = deepcopy(collection)
    for key in ("output_dir", "reference_dataset"):
        configured[key] = str((ROOT / configured[key]).resolve())
    if configured != reserved["config"]:
        raise ValueError("Collection profile changed after reservation; restore it or use another collection root")
    if query["episodes"]["memory"]:
        raise ValueError("Collection contains adaptation-memory episodes")
    if cache_only:
        if memory_source is None:
            raise ValueError("Cache-only collection status requires the prepared P1 source manifest")
        from lewm_tta.cache_loading import validate_query_receipts
        validate_query_receipts(settings, query, memory_source)
    else:
        missing = [row["path"] for row in query["episode_files"] if not Path(row["path"]).is_file()]
        if missing:
            raise ValueError("Collection is incomplete")
        source = Path(query["source"]["path"])
        if (file_hash(source) != query["dataset_sha256"] or source.stat().st_size != query["source"]["bytes"]
                or source.stat().st_mtime_ns != query["source"]["mtime_ns"]):
            raise ValueError("Fresh virtual dataset changed")
    counts = {split: len(query["episodes"][split]) for split in ("validation", "test")}
    if counts != collection["splits"]:
        raise ValueError("Exported split sizes disagree with the collection profile")
    audit_path = directory / "audit.json"
    audit_ok = cache_only
    if not cache_only and audit_path.is_file():
        audit = checked(audit_path)
        original = Path(configured["reference_dataset"])
        audit_ok = (audit.get("passed") is True and audit["query_manifest"] == query["fingerprint"]
                    and audit["collection_manifest"] == query["collection_manifest"]
                    and audit["source_dataset"] == str(original)
                    and audit["source_size"] == original.stat().st_size
                    and audit["source_mtime_ns"] == original.stat().st_mtime_ns)
    cache = []
    for progress in sorted(Path(settings["cache_dir"]).glob("observations-*/progress.json")):
        value = read_json(progress)
        if value.get("identity", {}).get("source") == query["source"]:
            cache.append({"path": str(progress.parent), "encoded_rows": value["encoded_rows"],
                          "complete": value.get("complete", False)})
    return query, {"episodes": counts,
                   "queries": {split: len(query["evaluation_windows"][split]["raw_start_rows"]) for split in counts},
                   "raw_frames": query["source"]["rows"], "audit_current": audit_ok,
                   "audit": str(audit_path), "observation_caches": cache}


def query_ids(query, starts):
    offsets = query["episode_metadata"]["offsets"]
    episodes = query["episode_metadata"]["ids"]
    prefix = f"{query['regime']}:{query['collection_manifest'][:16]}"
    return {f"{prefix}:{episodes[index]}:{row-offsets[index]}"
            for row in starts for index in [bisect_right(offsets, row)-1]}


def completed_report(path, job, query, split, limit=None):
    if not path.exists():
        if path.parent.exists():
            raise ValueError(f"Partial run preserved; choose a new --{split}-run (or --smoke-run): {path.parent}")
        return None
    if query is None:
        raise ValueError("Complete collection/export before verifying or reusing existing fresh runs")
    report = read_json(path)
    expected = effective_config(job["config"])
    if limit is not None:
        expected["evaluation"]["query_limit"] = limit
        expected["p1"]["evaluation"]["seeds"] = [42]
        if job["name"] == "temporal":
            expected["evaluation"]["seeds"] = [42]
            expected["evaluation"].pop("seeds_by_mode", None)
    if (not report.get("completed") or report.get("limited_run") != (limit is not None)
            or report.get("split") != split or report.get("stage") != job["report_stage"]
            or not contains(expected, report.get("config"))):
        raise ValueError(f"Existing run uses another protocol; choose a new run name: {path}")
    manifest = checked(Path(expected["experiment_dir"]) / "manifest.json")
    if (report["manifest_fingerprint"] != manifest["fingerprint"]
            or report["cohort"] != manifest["splits"][split]
            or manifest.get("query_source_fingerprint") != query["fingerprint"]):
        raise ValueError("Completed run belongs to another fresh cohort")
    code = report.get("code")
    if code != backend_identity(job["backend"]) and archive_entry(job["backend"], code) is None:
        raise ValueError("Completed run has no verified execution source")
    starts = manifest["splits"][split]["raw_start_rows"]
    if limit is not None:
        starts = starts[:limit]
    ids = query_ids(query, starts)
    seeds = expected["p1"]["evaluation"]["seeds"]
    rows = read_saved_records(report)
    if job["name"] == "p2":
        matrix = set(product(ids, seeds, expected["scopes"], expected["schedule_names"], expected["adaptation"]["methods"]))
        actual = {(row["query"]["transition_id"], row["seed"], row["scope"], row["schedule"], row["method"]) for row in rows}
    elif job["name"] == "p3":
        matrix = set(product(ids, seeds, expected["adaptation"]["methods"], expected["proposals"]["sources"]))
        actual = {(row["query"]["transition_id"], row["seed"], row["method"], row["proposal_source"]) for row in rows}
    else:
        matrix = {(item, seed, mode) for mode in expected["temporal"]["modes"]
                  for item in ids for seed in expected["evaluation"].get("seeds_by_mode", {}).get(mode, seeds)}
        actual = {(row["query"]["transition_id"], row["seed"], row["mode"]) for row in rows}
    if actual != matrix or len(rows) != len(matrix) or report["record_count"] != len(rows):
        raise ValueError("Completed run contains missing, duplicate or wrong-cohort records")
    return report


def verify_lock(path, report_path, report):
    if not path.exists():
        return False
    lock = read_json(path)
    if "fingerprint" in lock:
        lock = checked(path)
    elif lock.get("stage") not in ("P2-recorded-rollout", "P2.5-scope-comparison", "P2-followup-iteration"):
        raise ValueError(f"Validation lock has no fingerprint: {path}")
    # P2's established lock format has no self-fingerprint. Bind all its
    # scientific settings, including the validation-derived accuracy thresholds,
    # to the checksummed validation report rather than rewriting that lock.
    if (lock.get("stage") != report["stage"] or lock["config"] != report["config"]
            or lock["code"] != report["code"] or lock["manifest_fingerprint"] != report["manifest_fingerprint"]
            or Path(lock["validation_report"]).resolve() != report_path.resolve()
            or lock["report_sha256"] != file_hash(report_path)
            or lock.get("accurate_thresholds") != report.get("summary", {}).get("accurate_thresholds")):
        raise ValueError(f"Existing validation lock differs: {path}")
    return True


def main(action, arguments, dry_run=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", choices=tuple(JOBS), default=["p2", "p3"])
    parser.add_argument("--dataset", choices=DATASETS, help="Default: PushT, or the backend of an explicit collection profile")
    parser.add_argument("--collection-config", type=Path, help="Optional alternate fresh collection profile for the selected task")
    parser.add_argument("--p1-config", type=Path, help="Optional prepared source profile for the selected task")
    parser.add_argument("--regime", default="source")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-only", action="store_true", help="Require completed query caches and reuse the stored audit; never encode or re-audit raw data")
    parser.add_argument("--output-root", type=Path, help="Optional new suite root; one subdirectory per job")
    parser.add_argument("--p2-schedules", nargs="+", help="Optional names from the existing P2 schedule catalog")
    parser.add_argument("--p2-variant", choices=("fresh", "fresh-temporal"), default="fresh")
    parser.add_argument("--learning-rate", type=float, help="Optional common per-update LR; explicit P2 schedule LRs still take precedence")
    parser.add_argument("--validation-run", default="fresh-validation")
    parser.add_argument("--test-run", default="fresh-test")
    parser.add_argument("--smoke-run", default="fresh-smoke")
    parser.add_argument("--limit", type=int, default=2, help="Validation query limit for smoke only")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    if action in ("-h", "--help", "help"):
        parser.print_help()
        return
    args = parser.parse_args(arguments)
    if action not in ("status", "prepare", "check", "smoke", "validation", "test", "all"):
        raise ValueError("fresh supports status, prepare, check, smoke, validation, test and all")
    if args.limit < 1 or args.cpu_threads < 1 or len(args.jobs) != len(set(args.jobs)):
        raise ValueError("Use positive limits/threads and distinct jobs")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("learning-rate must be finite and positive")
    if action != "smoke" and any(value == "--limit" or value.startswith("--limit=") for value in arguments):
        raise ValueError("--limit belongs to smoke only; full validation/test cannot silently use a query limit")
    for name in (args.validation_run, args.test_run, args.smoke_run):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Run names must be simple directory names")
    if len({args.validation_run, args.test_run, args.smoke_run}) != 3:
        raise ValueError("Validation, test and smoke run names must differ")
    if args.collection_config:
        backend = read_profile(args.collection_config)["backend"]
        if args.dataset is not None and args.dataset != backend:
            raise ValueError("--dataset and --collection-config refer to different tasks")
        args.dataset = backend
    args.dataset = args.dataset or "pusht"
    if args.dataset not in DATASETS:
        raise ValueError("No P2/P3 evaluation profile for this collection backend")
    if args.collection_config is None:
        args.collection_config = (profile("collect", "default")[1] if args.dataset == "pusht"
                                  else ROOT / dataset_profile(args.dataset)["collection_config"])
    binding = {"collection_config": str(args.collection_config.resolve()), "regime": args.regime, "batch_size": 128}
    if args.cache_dir:
        binding["cache_dir"] = str(args.cache_dir.resolve())
    settings = query_dataset_settings(binding)
    memory_source = None
    if args.cache_only:
        from lewm_tta.rollout.resources import source_metadata
        source_profile = args.p1_config or (profile("p1", "default")[1] if args.dataset == "pusht"
                                            else ROOT/dataset_profile(args.dataset)["p1_config"])
        memory_source, _ = source_metadata({"p1": load_p1(Path(source_profile)), "cache_only": True})
    if dry_run and not Path(settings["manifest"]).is_file():
        collection = read_profile(args.collection_config)
        query = None
        readiness = {"episodes": collection["splits"],
            "queries": {split: count*collection["windows_per_episode"] for split, count in collection["splits"].items()},
            "raw_frames": sum(collection["splits"].values())*(collection["raw_steps"]+1),
            "audit_current": False, "observation_caches": []}
        print(f"Prerequisite: complete collection/export and its source audit: {settings['manifest']}", flush=True)
    else:
        query, readiness = collection_status(settings, args.collection_config, args.regime,
                                             cache_only=args.cache_only, memory_source=memory_source)
    print(f"Fresh {args.dataset} data: {readiness['episodes']} episodes; {readiness['queries']} reserved queries; {readiness['raw_frames']:,} frames.", flush=True)
    audit_label = "stored (not re-audited)" if args.cache_only else ("current" if readiness["audit_current"] else "required")
    print(f"Source audit: {audit_label}; shared query cache: {settings['cache_dir']}", flush=True)
    if args.cache_only:
        print("Cache-only: the prior sealed audit is reused; raw payloads are not reverified and missing caches will not be encoded.", flush=True)
    print(f"Observation cache metadata: {readiness['observation_caches'] or 'not encoded yet'}", flush=True)
    jobs = []
    for name in args.jobs:
        stage, variant, backend, report_stage = JOBS[name]
        if name == "p2":
            variant = args.p2_variant
        config = resolve(profile(stage, variant)[1], dataset=args.dataset if args.dataset != "pusht" else None,
                         cache_only=args.cache_only)
        if args.learning_rate is not None:
            config["adaptation"]["learning_rate"] = args.learning_rate
        if args.p1_config:
            config["p1_config"] = str(args.p1_config.resolve())
        config["query_dataset"] = deepcopy(settings)
        collection = read_profile(args.collection_config)
        p1 = load_p1(Path(config["p1_config"]))
        if (p1["data"].get("dataset_id", "pusht") != args.dataset
                or (ROOT/collection["reference_dataset"]).resolve() != Path(p1["paths"]["dataset"]).resolve()):
            raise ValueError("Fresh collection audit and P1 source profile must use the same task/reference dataset")
        from lewm_tta.collection.contracts import validate_evaluation_config
        validate_evaluation_config({**config, "p1": p1})
        if args.output_root:
            config["experiment_dir"] = str(args.output_root.resolve() / name)
        if name == "p2" and args.p2_schedules:
            if len(args.p2_schedules) != len(set(args.p2_schedules)) or set(args.p2_schedules)-set(config["schedules"]):
                raise ValueError("Choose distinct P2 schedules from its catalog")
            config["schedule_names"] = args.p2_schedules
        config["experiment_dir"] = str((ROOT / config["experiment_dir"]).resolve())
        if config.get("reuse"):
            raise ValueError("Fresh suite cannot import source-corpus experiment results")
        if config["evaluation"]["query_limit"] is not None or config["evaluation"].get("records") != "compact":
            raise ValueError("Fresh suite requires full-cohort profiles with compact persistent records")
        jobs.append({"name": name, "stage": stage, "backend": backend, "report_stage": report_stage, "config": config})
    if action == "status":
        for job in jobs:
            directory = Path(job["config"]["experiment_dir"])
            print(f"{job['name']}: {directory}; manifest={'present' if (directory/'manifest.json').is_file() else 'missing'}", flush=True)
            for split, name in (("validation", args.validation_run), ("test", args.test_run)):
                path = directory / "runs" / name / "report.json"
                completed = completed_report(path, job, query, split)
                print(f"  {split}: {'verified complete' if completed else 'not run'}", flush=True)
        return
    runtime = ["--device", args.device, "--cpu-threads", str(args.cpu_threads)]
    if args.verbose:
        runtime.append("--verbose")
    prefix = [sys.executable, "-X", "faulthandler", "-u", str(ROOT / "codes/experiment.py")]

    def run(stage, command, *options):
        invocation = [*prefix, stage, command, *map(str, options)]
        print(shlex.join(invocation), flush=True)
        if not dry_run:
            subprocess.run(invocation, cwd=ROOT, check=True)

    def config_path(job):
        value = job["config"]
        return ROOT / "artifacts/experiment-configs" / f"{object_hash(value)}.json" if dry_run else materialize(value)

    if action in ("prepare", "all"):
        if not readiness["audit_current"]:
            if args.cache_only:
                raise ValueError("Cache-only preparation requires an existing passed audit; it cannot run collect check")
            run("collect", "check", "--config", args.collection_config.resolve(), "--regimes", args.regime)
        else:
            print("Reuse prior sealed source-overlap audit." if args.cache_only else "Reuse current source-overlap audit.", flush=True)
        for job in jobs:
            run(job["stage"], "prepare", "--config", config_path(job))
        if action == "prepare":
            print(("Would prepare metadata." if dry_run else "Metadata prepared.")+
                  (" Run fresh check to verify the completed caches and numerical compatibility."
                   if args.cache_only else " Run fresh check to encode the new frames and check numerical compatibility."), flush=True)
            return
    elif not readiness["audit_current"] and not dry_run:
        raise ValueError("Run bash codes/run.sh fresh prepare first; the fresh source-overlap audit is missing/stale")
    if action in ("check", "all"):
        for job in jobs:
            run(job["stage"], "check", "--config", config_path(job), *runtime)
        if action == "check":
            return
    for job in jobs:
        directory = Path(job["config"]["experiment_dir"])
        if not dry_run and not (directory / "manifest.json").is_file():
            raise ValueError(f"Run fresh prepare for {job['name']} first")
        if action == "smoke":
            path = directory / "runs" / args.smoke_run / "report.json"
            if completed_report(path, job, query, "validation", args.limit) is None:
                run(job["stage"], "evaluate", "--config", config_path(job), "--limit", args.limit,
                    "--seeds", 42, "--run-name", args.smoke_run, *runtime)
                if not dry_run:
                    completed_report(path, job, query, "validation", args.limit)
            else:
                print(f"Skip verified smoke: {path}", flush=True)
            continue
        validation = directory / "runs" / args.validation_run / "report.json"
        lock = directory / "locks" / f"{args.validation_run}.json"
        report = completed_report(validation, job, query, "validation")
        if report is None:
            if action == "test" and not dry_run:
                raise ValueError(f"Complete fresh validation for {job['name']} before test")
            if action != "test":
                run(job["stage"], "evaluate", "--config", config_path(job), "--run-name", args.validation_run, *runtime)
                if not dry_run:
                    report = completed_report(validation, job, query, "validation")
        else:
            print(f"Skip verified validation: {validation}", flush=True)
        if action != "test":
            run(job["stage"], "analyze", "--report", validation)
        if report is None or not verify_lock(lock, validation, report):
            if action == "test":
                if not dry_run:
                    raise ValueError(f"Complete fresh validation/lock first: {lock}")
                print(f"Requires the completed validation report and its existing lock: {lock}", flush=True)
            else:
                run(job["stage"], "lock", "--report", validation, "--output", lock)
        if action in ("test", "all"):
            path = directory / "runs" / args.test_run / "report.json"
            if completed_report(path, job, query, "test") is None:
                run(job["stage"], "evaluate", "--split", "test", "--locked-config", lock,
                    "--run-name", args.test_run, *runtime)
                if not dry_run:
                    completed_report(path, job, query, "test")
            else:
                print(f"Skip verified test: {path}", flush=True)
            run(job["stage"], "analyze", "--report", path)
