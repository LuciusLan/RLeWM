"""Validation-only temporal schedule/LR sweep through the existing P2 engine.

This module resolves configs and verifies saved JSON records. Model inference,
updates and numerical checks remain in the ordinary stage subprocesses.
"""

import argparse
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.collection.manifest import checked
from lewm_tta.config import METHODS
from lewm_tta.rollout.config import validate_config
from lewm_tta.rollout.record_io import open_records
from lewm_tta.runtime import file_hash, object_hash, read_json
from .catalog import ROOT
from .config import materialize, resolve
from .fresh import effective_config, query_ids
from .provenance import archive_entry, backend_identity


SCHEDULES = ("temporal-u1", "temporal-u2", "temporal-u1-reset", "temporal-u2-reset")


def completed_report(path, expected, seeds):
    """Stream compact records to verify the exact matrix before skipping a run."""
    if not path.exists():
        if path.parent.exists():
            raise ValueError(f"Partial run preserved; choose another --run-prefix: {path.parent}")
        return False
    report = read_json(path)
    if (not report.get("completed") or report.get("split") != "validation"
            or report.get("stage") != "P2-followup-iteration"
            or report.get("limited_run") != (expected["evaluation"]["query_limit"] is not None)
            or report.get("config") != expected):
        raise ValueError(f"Existing run uses another protocol; choose another --run-prefix: {path}")
    manifest = checked(Path(expected["experiment_dir"]) / "manifest.json")
    query = checked(Path(expected["query_dataset"]["manifest"]))
    if (report.get("manifest_fingerprint") != manifest["fingerprint"]
            or report.get("cohort") != manifest["splits"]["validation"]
            or manifest.get("query_source_fingerprint") != query["fingerprint"]
            or report.get("query_source_fingerprint") != query["fingerprint"]):
        raise ValueError(f"Existing run belongs to a different cohort: {path}")
    code = report.get("code")
    if not code or (code != backend_identity("p2_followup") and archive_entry("p2_followup", code) is None):
        raise ValueError(f"Existing run has no verified execution source: {path}")
    starts = manifest["splits"]["validation"]["raw_start_rows"]
    limit = expected["evaluation"]["query_limit"]
    ids = query_ids(query, starts if limit is None else starts[:limit])
    matrix = set(product(ids, seeds, expected["scopes"], SCHEDULES, expected["adaptation"]["methods"]))
    records = Path(report.get("records_file") or "")
    if (records.parent.resolve() != path.parent.resolve() or not records.is_file()
            or not records.stat().st_size or file_hash(records) != report.get("records_sha256")):
        raise ValueError(f"Missing or changed saved records: {path}")
    seen = set()
    with open_records(records) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["query"]["transition_id"], row["seed"], row["scope"], row["schedule"], row["method"])
            if key not in matrix or key in seen:
                raise ValueError(f"Unexpected or duplicate sweep record: {path}")
            seen.add(key)
    if seen != matrix or report.get("record_count") != len(matrix):
        raise ValueError(f"Incomplete saved sweep matrix: {path}")
    return True


def sweep(profile_path, arguments, dry_run=False, *, dataset=None, cache_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", "--lrs", nargs="+", type=float, default=[1e-5, 2.5e-5, 5e-5],
                        help="Per-update LRs, applied equally to all four schedules")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="Default: all seven methods; include frozen")
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: profile seeds (42/43/44 for fresh)")
    parser.add_argument("--run-prefix", default="temporal-sweep")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Integration only; limited runs cannot be locked")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(arguments)
    if (any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates)
            or len(set(args.learning_rates)) != len(args.learning_rates)):
        raise ValueError("Choose distinct finite positive learning rates")
    if (not args.run_prefix or Path(args.run_prefix).name != args.run_prefix
            or args.run_prefix in (".", "..") or args.cpu_threads < 1
            or (args.limit is not None and args.limit < 1)):
        raise ValueError("Use a simple run prefix and positive limits/threads")
    base = resolve(profile_path, dataset=dataset, cache_only=cache_only)
    if not base.get("query_dataset") or not base.get("scopes"):
        raise ValueError("The temporal sweep requires a fresh query profile; use --variant fresh-temporal")
    if set(SCHEDULES) - set(base.get("schedules", {})):
        raise ValueError("The profile must include all four temporal schedules in its catalog")
    if args.limit is None and base["evaluation"]["query_limit"] is not None:
        raise ValueError("Use an explicit --limit for an integration run")
    if args.experiment_dir:
        base["experiment_dir"] = str(args.experiment_dir.resolve())
    base["experiment_dir"] = str((ROOT / base["experiment_dir"]).resolve())
    base["schedule_names"] = list(SCHEDULES)
    base["schedules"] = {name: deepcopy(base["schedules"][name]) for name in SCHEDULES}
    for name, schedule in base["schedules"].items():
        updates = 2 if name.startswith("temporal-u2") else 1
        persistence = "position" if name.endswith("-reset") else "trajectory"
        # The sweep controls every LR, including inherited schedule overrides.
        schedule.pop("learning_rate", None)
        if schedule != {"placement": "each-step", "updates": updates, "persistence": persistence}:
            raise ValueError(f"Unexpected temporal schedule definition: {name}")
    if args.methods is not None:
        base["adaptation"]["methods"] = args.methods
    methods = base["adaptation"]["methods"]
    if "frozen" not in methods or len(set(methods)) != len(methods):
        raise ValueError("Choose distinct methods including frozen for paired comparisons")
    seeds = args.seeds if args.seeds is not None else effective_config(base)["p1"]["evaluation"]["seeds"]
    if min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct nonnegative seeds")
    base["evaluation"]["query_limit"] = args.limit
    base["evaluation"]["records"] = "compact"
    prefix = [sys.executable, "-X", "faulthandler", "-u", str(ROOT / "codes/experiment.py"), "p2"]
    runtime = ["--device", args.device, "--cpu-threads", str(args.cpu_threads)]
    if args.verbose:
        runtime.append("--verbose")

    def run(action, *options):
        command = [*prefix, action, *map(str, options)]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    print("P2 validation sweep: four temporal schedules x per-update LR.", flush=True)
    print(f"LRs: {args.learning_rates}; seeds: {seeds}; methods: {methods}", flush=True)
    print(f"Schedules: {list(SCHEDULES)}", flush=True)
    print("U1/U2 use 5/10 updates at H5; reset restores WM and optimizer at each imagined position.", flush=True)
    print("All schedules use the same per-update LR; U2 is not automatically scaled down.", flush=True)
    jobs = []
    for rate in args.learning_rates:
        config = deepcopy(base)
        config["adaptation"]["learning_rate"] = rate
        expected = effective_config(config)
        expected["p1"]["evaluation"]["seeds"] = list(seeds)
        validate_config(expected)
        name = f"{args.run_prefix}-lr{rate!r}" + (f"-smoke{args.limit}" if args.limit is not None else "")
        path = Path(config["experiment_dir"]) / "runs" / name / "report.json"
        # Preflight every job before starting any numerical subprocess.
        done = completed_report(path, expected, seeds)
        config_path = ROOT / "artifacts/experiment-configs" / f"{object_hash(config)}.json"
        jobs.append((config, config_path, name, path, done))
    pending = [job for job in jobs if not job[-1]]
    if pending:
        first, path, *_ = pending[0]
        if not dry_run:
            materialize(first)
        # Metadata only: reuses the exact prepared cohort and encoded cache.
        run("prepare", "--config", path)
    for config, config_path, name, report, done in jobs:
        if done:
            print(f"Skip verified completed validation: {report}", flush=True)
            continue
        if not dry_run:
            materialize(config)
        run("evaluate", "--config", config_path, "--split", "validation", "--run-name", name,
            "--seeds", *seeds, *runtime)
        if not dry_run:
            expected = effective_config(config)
            expected["p1"]["evaluation"]["seeds"] = list(seeds)
            if not completed_report(report, expected, seeds):
                raise ValueError(f"Evaluation did not write a completed report: {report}")
    print(f"{'Planned' if dry_run else 'Completed'} validation sweep: {base['experiment_dir']}/runs/{args.run_prefix}-*", flush=True)
    print("Review per-run summary.txt/report.json before selecting any locked test comparison.", flush=True)
