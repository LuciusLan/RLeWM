"""Single-pass interleave budget snapshots, with independent legacy planners."""

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.collection.manifest import checked
from lewm_tta.budget_receipts import verify_budget_capture
from lewm_tta.outcomes.inputs import condition
from lewm_tta.rollout.record_io import read_saved_records
from lewm_tta.runtime import object_hash, read_json
from .catalog import ROOT, infer_backend
from .config import materialize, resolve
from .fresh import effective_config, query_ids
from .provenance import backend_identity


def budget_configs(base, iterations, root):
    result = []
    for count in sorted(iterations):
        config = deepcopy(base)
        config["experiment_dir"] = str(Path(root).resolve()/"planning"/f"j-{count}")
        config["planning"]["iterations"] = count
        config["reuse"] = {}
        if "interleave" in config:
            config["interleave"]["checkpoints"] = [j for j in base["interleave"]["checkpoints"] if j < count]
        result.append(config)
    return result


def expected_config(config, seeds):
    value = effective_config(config)
    value["p1"]["evaluation"]["seeds"] = list(seeds)
    return value


def completed_report(path, expected, backend):
    if not path.is_file():
        if path.parent.exists():
            raise ValueError(f"Partial planning run preserved; choose another output root: {path.parent}")
        return False
    report = read_json(path)
    if (not report.get("completed") or report.get("split") != "validation" or report.get("config") != expected
            or report.get("code") != backend_identity(backend)):
        raise ValueError(f"Completed budget run differs in config/source; choose another output root: {path}")
    verify_budget_capture(report)
    manifest = checked(Path(expected["experiment_dir"])/"manifest.json")
    query = checked(Path(expected["query_dataset"]["manifest"]))
    if (manifest["fingerprint"] != report["manifest_fingerprint"]
            or manifest["query_source_fingerprint"] != query["fingerprint"]):
        raise ValueError("Completed planning run has a different prepared fresh cohort")
    starts = manifest["splits"]["validation"]["raw_start_rows"]
    limit = expected["evaluation"]["query_limit"]
    ids = query_ids(query, starts if limit is None else starts[:limit])
    seeds = expected["p1"]["evaluation"]["seeds"]
    if backend == "p3_interleave":
        cells = [(seed, label) for seed, label in product(seeds, expected["interleave"]["arms"])]
    elif backend == "p3_temporal":
        from lewm_tta.planning.temporal.config import seeds_for_mode
        cells = [(seed, mode) for mode in expected["temporal"]["modes"] for seed in seeds_for_mode(expected, mode)]
    else:
        methods = expected["adaptation"]["methods"]
        labels = methods if backend == "p3a" else [method+"/"+source for method, source in product(methods, expected["proposals"]["sources"])]
        cells = list(product(seeds, labels))
    expected_rows = {(query_id, seed, label) for query_id in ids for seed, label in cells}
    rows = read_saved_records(report)
    actual = {(row["query"]["transition_id"], row["seed"], condition(row)) for row in rows}
    if len(rows) != len(actual) or actual != expected_rows or not all(all(row["checks"].values()) for row in rows):
        raise ValueError("Completed planning run has an incomplete/failed query-seed-condition matrix")
    return True


def budget(selected, arguments, dry_run=False, *, dataset=None, cache_only=False):
    parser = argparse.ArgumentParser(description="P3 success/efficiency validation; single-pass interleave stopping plans")
    parser.add_argument("--iterations", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("auto", "single-pass", "independent"), default="auto",
                        help="auto uses one maximum-J search for interleave and independent runs for other backends")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--planning-only", action="store_true", help="Replay physical outcomes separately, e.g. in the original physics environment")
    parser.add_argument("--pusht-success", choices=("native-goal", "coverage"), default="native-goal")
    parser.add_argument("--coverage-threshold", type=float, default=.95)
    parser.add_argument("--success-targets", nargs="+", type=float, default=[.5, .8, .9])
    parser.add_argument("--deadlines", nargs="+", type=float, default=[.1, .5, 1., 2., 5., 10.])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(arguments)
    if (not args.iterations or min(args.iterations) < 1 or len(set(args.iterations)) != len(args.iterations)
            or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1)):
        raise ValueError("Choose distinct positive budgets, positive CPU threads and optional positive limit")
    base = resolve(selected, dataset=dataset, cache_only=cache_only)
    if not base.get("query_dataset"):
        raise ValueError("Physical planning budgets require fresh queries and complete reset receipts")
    seeds = args.seeds if args.seeds is not None else base["evaluation"].get("seeds", effective_config(base)["p1"]["evaluation"]["seeds"])
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct nonnegative planner seeds")
    base["evaluation"].update(query_limit=args.limit, records="compact")
    if base.get("comparison") in ("interleaved-planning", "temporal-planning"):
        base["evaluation"]["seeds"] = list(seeds)
    if args.seeds is not None:
        base["evaluation"].pop("seeds_by_mode", None)
    if args.learning_rate is not None:
        base["adaptation"]["learning_rate"] = args.learning_rate
    backend = infer_backend("p3", base)
    mode = ("single-pass" if backend == "p3_interleave" else "independent") if args.execution_mode == "auto" else args.execution_mode
    if mode == "single-pass" and backend != "p3_interleave":
        raise ValueError("Single-pass snapshots currently support --variant interleave; use independent for other planners")
    root = args.output_root.resolve()
    jobs = []
    for config in budget_configs(base, args.iterations, root):
        expected = expected_config(config, seeds)
        if backend == "p3_interleave":
            from lewm_tta.interleave.config import validate_config
        elif backend == "p3_temporal":
            from lewm_tta.planning.temporal.config import validate_config
        elif backend in ("p3b", "p3c"):
            from lewm_tta.proposals.config import validate_config
        else:
            from lewm_tta.planning.config import validate_config
        validate_config(expected)
        name = "validation"+(f"-smoke{args.limit}" if args.limit is not None else "")
        path = Path(config["experiment_dir"])/"runs"/name/"report.json"
        done = completed_report(path, expected, backend)
        if done:
            capture = read_json(path).get("budget_capture")
            recorded_mode = "single-pass" if capture else "independent"
            if recorded_mode != mode or (capture and capture["requested_iterations"] != sorted(args.iterations)):
                raise ValueError("Existing reports use another capture mode/budget list; preserve them and select a new output root")
        jobs.append((config, expected, name, path, done))
    if mode == "single-pass" and any(job[-1] for job in jobs) and not all(job[-1] for job in jobs):
        raise ValueError("Partial shared search preserved; single-pass capture needs a new output root")
    prefix = [sys.executable, "-X", "faulthandler", "-u", str(ROOT/"codes/experiment.py"), "p3"]
    def run(arguments):
        command = [*prefix, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    outcome_dir = root/("outcomes"+(f"-smoke{args.limit}" if args.limit is not None else ""))
    if not args.planning_only and outcome_dir.exists():
        raise ValueError(f"Physical output already exists; analyze it or choose a new --output-root: {outcome_dir}")
    # Validate physical settings before launching any planning work.
    from math import isfinite
    if (not 0 < args.coverage_threshold <= 1 or args.bootstrap_samples < 0
            or any(not isfinite(x) or not 0 < x <= 1 for x in args.success_targets)
            or any(not isfinite(x) or x <= 0 for x in args.deadlines)):
        raise ValueError("Invalid physical success targets, deadlines or bootstrap count")
    print(f"P3 validation capture={mode}: same population, horizon, query starts, seeds and LR; inactive checkpoints trimmed. Source-result reuse disabled.", flush=True)
    if mode == "single-pass":
        if not all(job[-1] for job in jobs):
            for config, expected, name, path, done in jobs:
                config_path = ROOT/"artifacts/experiment-configs"/f"{object_hash(config)}.json" if dry_run else materialize(config)
                run(["prepare", "--config", config_path])
            maximum, _, name, _, _ = jobs[-1]
            config_path = ROOT/"artifacts/experiment-configs"/f"{object_hash(maximum)}.json" if dry_run else materialize(maximum)
            run(["evaluate-snapshots", "--config", config_path, "--snapshot-iterations", *sorted(args.iterations),
                 "--snapshot-root", root, "--run-name", name, "--device", args.device, "--cpu-threads", args.cpu_threads,
                 *(["--verbose"] if args.verbose else [])])
            if not dry_run:
                for _, expected, _, path, _ in jobs:
                    if not completed_report(path, expected, backend):
                        raise ValueError("Shared search did not publish every requested stopping report")
        else:
            print("Skip verified completed shared budget capture.", flush=True)
    for config, expected, name, path, done in ([] if mode == "single-pass" else jobs):
        if done:
            print(f"Skip verified completed budget: {path}", flush=True)
            continue
        config_path = ROOT/"artifacts/experiment-configs"/f"{object_hash(config)}.json" if dry_run else materialize(config)
        run(["prepare", "--config", config_path])
        run(["evaluate", "--config", config_path, "--split", "validation", "--run-name", name,
             "--seeds", *seeds, "--device", args.device, "--cpu-threads", args.cpu_threads,
             *(["--verbose"] if args.verbose else [])])
        if not dry_run and not completed_report(path, expected, backend):
            raise ValueError("Planning budget did not produce a complete source report")
    physical = ["success", "evaluate", "--reports", *(job[3] for job in jobs), "--output-dir", outcome_dir,
        "--pusht-success", args.pusht_success, "--coverage-threshold", args.coverage_threshold,
        "--success-targets", *args.success_targets, "--deadlines", *args.deadlines, "--bootstrap-samples", args.bootstrap_samples]
    if args.planning_only:
        print("Run physical replay separately:", flush=True)
        print(shlex.join([*prefix, *map(str, physical)]), flush=True)
    else:
        run(physical)
