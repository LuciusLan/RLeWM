"""Prepare/encode shifted pools and run existing P2/P3 engines with own locks."""

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.collection.manifest import checked
from lewm_tta.config import METHODS
from lewm_tta.rollout.resources import require_existing_cache, source_metadata
from lewm_tta.runtime import read_json
from .catalog import ROOT
from .fresh import completed_report, verify_lock
from .shifted_profiles import context, exported, make_profiles
from .shifted_storage import check_space


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", choices=("status", "prepare", "check", "smoke", "validation", "test", "all"))
    parser.add_argument("--suite-config", type=Path, default=ROOT/"codes/configs/appearance/evaluation.json")
    parser.add_argument("--conditions", nargs="+", default=["pusht-object"], help="Named conditions or all")
    parser.add_argument("--memory", nargs="+", choices=("source", "shifted"), default=["source", "shifted"])
    parser.add_argument("--jobs", nargs="+", choices=("p2", "p3", "temporal"), default=["p2", "p3"])
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--cache-root", type=Path, help="New arrays only; existing source caches are reused")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--p2-schedules", nargs="+")
    parser.add_argument("--p2-methods", nargs="+", choices=METHODS)
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Appearance synthesis processes; PushT may use4")
    parser.add_argument("--write-batch-size", type=int, default=16)
    parser.add_argument("--keep-free-gib", type=float, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--validation-run", default="shift-validation")
    parser.add_argument("--test-run", default="shift-test")
    parser.add_argument("--smoke-run", default="shift-smoke")
    parser.add_argument("--limit", type=int, default=2, help="smoke only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if (any(value < 1 for value in (args.workers, args.write_batch_size, args.cpu_threads, args.limit))
            or args.encode_batch_size is not None and args.encode_batch_size < 1
            or not math.isfinite(args.keep_free_gib) or args.keep_free_gib < 0):
        parser.error("Counts must be positive and keep-free-gib finite/nonnegative")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning-rate must be finite and positive")
    if args.seeds is not None and (len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds)):
        parser.error("Use distinct nonnegative seeds")
    for values in (args.jobs, args.memory, args.conditions):
        if len(values) != len(set(values)):
            parser.error("Select distinct jobs, memory kinds and conditions")
    for name in (args.validation_run, args.test_run, args.smoke_run):
        if Path(name).name != name or name in ("", ".", ".."):
            parser.error("Run names must be simple directory names")
    if len({args.validation_run, args.test_run, args.smoke_run}) != 3:
        parser.error("Validation/test/smoke run names must differ")
    supplied = sys.argv[1:] if argv is None else argv
    if args.action != "smoke" and any(value.split("=", 1)[0] == "--limit" for value in supplied):
        parser.error("--limit is for smoke only; validation/test always use the full reserved cohort")
    suite = read_json(args.suite_config)
    if args.conditions == ["all"]:
        args.conditions = list(suite["conditions"])
    if set(args.conditions)-set(suite["conditions"]):
        parser.error("Choose --conditions " + " ".join(suite["conditions"]) + " (or all)")
    contexts = [context(args, suite, name) for name in args.conditions]
    prefix = [sys.executable, "-X", "faulthandler", "-u", str(ROOT/"codes/experiment.py")]
    runtime = ["--device", args.device, "--cpu-threads", str(args.cpu_threads)] + (["--verbose"] if args.verbose else [])

    def run(*arguments):
        command = [*prefix, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    def raw(ctx, role):
        value = exported(ctx, role, required=False)
        directory = Path(ctx["paths"][role])
        if value is None:
            parent = ctx["paths"]["query_parent" if role == "queries" else "source_manifest"]
            options = []
            for key, value in ctx["condition"]["appearance"].items():
                options += ["--"+key.replace("_", "-"), *(value if isinstance(value, list) else [value])]
            existing_recipe = checked(directory/"manifest.json") if (directory/"manifest.json").exists() else None
            compression = existing_recipe.get("compression", "gzip") if existing_recipe is not None else "gzip"
            run("appearance", "synthesize", "--dataset", ctx["condition"]["dataset"], "--source-manifest", parent,
                "--output-dir", directory, "--workers", args.workers, "--write-batch-size", args.write_batch_size,
                "--min-free-gib", args.keep_free_gib, "--compression", compression, *options)
            value = exported(ctx, role, required=not args.dry_run)
        audit_path = directory/"audit.json"
        audit = checked(audit_path) if audit_path.exists() else None
        if audit is None or not audit.get("passed") or value is not None and audit.get("appearance_recipe") != value["appearance"]["fingerprint"]:
            run("appearance", "check", "--output-dir", directory)

    if args.action == "status":
        for ctx in contexts:
            print(f"{ctx['name']}: LR={ctx['binding']['learning_rate']}, seeds={ctx['binding']['seeds']}")
            for role in ("memory", "queries"):
                value = exported(ctx, role, required=False)
                print(f"  {role}: {'exported' if value is not None else 'synthesis required'} | {ctx['paths'][role]}")
            print(f"  shifted P1: {'prepared' if (Path(ctx['paths']['p1_output'])/'prepared.json').exists() else 'not prepared'}")
        check_space(contexts, args.memory, args.keep_free_gib)
        return

    if args.action in ("prepare", "all"):
        # Check arrays before expensive rendering; repeat with exact exported
        # row counts before the first model is loaded.
        fits = check_space(contexts, args.memory, args.keep_free_gib)
        if not fits and not args.dry_run:
            raise ValueError("Choose a larger --cache-root before preparation; current cache volume is insufficient")
        for ctx in contexts:
            if "shifted" in args.memory:
                raw(ctx, "memory")
            raw(ctx, "queries")
        check_space(contexts, args.memory, args.keep_free_gib, reserve=not args.dry_run)

    for ctx in contexts:
        jobs, query = make_profiles(ctx, args, suite, dry_run=args.dry_run)
        print(f"{ctx['name']}: LR={ctx['binding']['learning_rate']}; seeds={ctx['binding']['seeds']}; query cache {ctx['paths']['query_cache']}")
        if args.action in ("prepare", "all"):
            if not args.dry_run:
                manifest, prepared = source_metadata({"p1": ctx["source_p1"]})
                require_existing_cache(ctx["source_p1"], manifest, prepared)
            if "shifted" in args.memory:
                prepared_path = Path(ctx["paths"]["p1_output"])/"prepared.json"
                if prepared_path.exists():
                    manifest, prepared = source_metadata({"p1": ctx["shifted_p1"]})
                    require_existing_cache(ctx["shifted_p1"], manifest, prepared)
                    print(f"Reuse prepared shifted memory: {prepared_path}")
                else:
                    run("p1", "prepare", "--config", ctx["paths"]["p1_shifted"], *runtime,
                        "--encode-batch-size", ctx["binding"]["encode_batch_size"])
            for job in jobs:
                run(job["stage"], "prepare", "--config", job["config_path"])
            # The first check encodes all fresh shifted frames; subsequent checks
            # reuse the identical query cache across stages and memory conditions.
        if args.action in ("prepare", "check", "all"):
            for job in jobs:
                run(job["stage"], "check", "--config", job["config_path"], *runtime)
            if args.action in ("prepare", "check"):
                continue
        for job in jobs:
            directory = Path(job["config"]["experiment_dir"])
            if not args.dry_run and not (directory/"manifest.json").exists():
                raise ValueError("Run bash codes/prepare_shifted.sh for the selected conditions/jobs first")
            if args.action == "smoke":
                report = directory/"runs"/args.smoke_run/"report.json"
                if completed_report(report, job, query, "validation", args.limit) is None:
                    run(job["stage"], "evaluate", "--config", job["config_path"], "--limit", args.limit,
                        "--seeds", 42, "--run-name", args.smoke_run, *runtime)
                    if not args.dry_run:
                        completed_report(report, job, query, "validation", args.limit)
                continue
            validation = directory/"runs"/args.validation_run/"report.json"
            lock = directory/"locks"/(args.validation_run + ".json")
            report = completed_report(validation, job, query, "validation")
            if report is None and args.action == "test":
                if not args.dry_run:
                    raise ValueError(f"Complete shifted validation and lock first: {validation}")
                print(f"Requires completed validation and lock: {lock}")
            elif report is None:
                run(job["stage"], "evaluate", "--config", job["config_path"], "--run-name", args.validation_run, *runtime)
                if not args.dry_run:
                    report = completed_report(validation, job, query, "validation")
            else:
                print(f"Skip verified complete validation: {validation}")
            if args.action != "test":
                run(job["stage"], "analyze", "--report", validation)
            locked = report is not None and verify_lock(lock, validation, report)
            if not locked:
                if args.action == "test":
                    if not args.dry_run:
                        raise ValueError(f"Create the validation lock before testing: {lock}")
                else:
                    run(job["stage"], "lock", "--report", validation, "--output", lock)
            if args.action in ("test", "all"):
                test = directory/"runs"/args.test_run/"report.json"
                if completed_report(test, job, query, "test") is None:
                    run(job["stage"], "evaluate", "--split", "test", "--locked-config", lock,
                        "--run-name", args.test_run, *runtime)
                    if not args.dry_run:
                        completed_report(test, job, query, "test")
                else:
                    print(f"Skip verified complete test: {test}")
                run(job["stage"], "analyze", "--report", test)
