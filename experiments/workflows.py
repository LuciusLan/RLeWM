"""Resume the fixed P3 six-cell comparison through the existing stage commands."""

import argparse
from itertools import product
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.runtime import file_hash, object_hash, read_json
from lewm_tta.rollout.record_io import read_saved_records
from lewm_tta.rollout.config import effective_p1
from lewm_tta.config import load_config as load_p1
from .catalog import ROOT
from .config import materialize, resolve
from .provenance import archive_entry, backend_identity
from lewm_tta.proposals.reuse import normalize_reuse


def _subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _subset(v, actual[k]) for k, v in expected.items())
    return expected == actual


def completed_report(path, config, split):
    if not path.exists():
        return None
    report = read_json(path)
    if (not report.get("completed") or report.get("limited_run") or report.get("split") != split
            or report.get("stage") != "P3c-offline-factorial" or not _subset(config, report["config"])):
        raise ValueError(f"Existing run is incomplete or uses another protocol; choose a new run name: {path}")
    manifest = read_json(Path(config["experiment_dir"]) / "manifest.json")
    if report["manifest_fingerprint"] != manifest["fingerprint"] or report["cohort"] != manifest["splits"][split]:
        raise ValueError("Completed factorial run belongs to a different prepared cohort")
    identity = backend_identity("p3c")
    if report["code"] != identity and archive_entry("p3c", report["code"]) is None:
        raise ValueError("Completed factorial run has no verified execution source")
    rows = read_saved_records(report)
    ids = {row["query"]["transition_id"] for row in rows}
    seeds = report["config"]["p1"]["evaluation"]["seeds"]
    expected = set(product(ids, seeds, config["adaptation"]["methods"], config["proposals"]["sources"]))
    actual = {(r["query"]["transition_id"], r["seed"], r["method"], r["proposal_source"]) for r in rows}
    if (actual != expected or len(actual) != len(rows) or len(ids) != report["cohort"]["queries"]
            or len({row["query"]["source_episode_position"] for row in rows}) != report["cohort"]["episodes"]):
        raise ValueError("Completed factorial run has an incomplete or duplicate matrix")
    return report


def compare(profile_path, arguments, dry_run=False, *, dataset=None, cache_only=False):
    parser = argparse.ArgumentParser(description="Complete/resume the P3 random/retrieved TTT x CEM comparison")
    parser.add_argument("--phase", choices=("validation", "test", "all"), default="all")
    parser.add_argument("--validation-run", default="factorial-validation")
    parser.add_argument("--test-run", default="factorial-test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(arguments)
    resolved = resolve(profile_path, dataset=dataset, cache_only=cache_only)
    config = dict(resolved)
    config["experiment_dir"] = str((ROOT / config["experiment_dir"]).resolve())
    normalize_reuse(config)
    config["p1"] = load_p1(Path(config["p1_config"]))
    config["adaptation"] = effective_p1(config)["adaptation"]
    if (config.get("comparison") != "factorial"
            or config["adaptation"]["methods"] != ["frozen", "uniform", "fr-fixed"]
            or config["proposals"]["sources"] != ["gaussian", "context-fragments"]
            or config["evaluation"]["query_limit"] is not None
            or config["evaluation"].get("records", "full") == "none"):
        raise ValueError("p3 compare requires the full six-cell profile with persistent records")
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    for name in (args.validation_run, args.test_run):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Workflow run names must be simple directory names")
    output = Path(config["experiment_dir"])
    # Subprocesses must receive the bound configuration, not the original
    # shared PushT profile. Old calls without a selector retain their handoff.
    engine_profile = profile_path
    if dataset is not None:
        engine_profile = (ROOT/"artifacts/experiment-configs"/f"{object_hash(resolved)}.json"
                          if dry_run else materialize(resolved))
    validation = output / "runs" / args.validation_run / "report.json"
    test = output / "runs" / args.test_run / "report.json"
    lock_path = output / "locks" / f"{args.validation_run}.json"
    prefix = [sys.executable, "-X", "faulthandler", "-u", str(ROOT / "codes/experiment.py"), "p3"]
    runtime = ["--device", args.device, "--cpu-threads", str(args.cpu_threads)]
    if args.verbose:
        runtime.append("--verbose")

    def run(action, *options):
        command = [*prefix, action, *map(str, options)]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)

    print("P3: Frozen / random-support TTT / retrieved-support TTT x Gaussian / context-guided CEM.", flush=True)
    print("Pinned Gaussian source cells are reused; only missing searches are evaluated." if config.get("reuse")
          else "No historical source cells configured; all six cells use the selected query cohort.", flush=True)
    run("prepare", "--config", engine_profile)
    existing_validation = completed_report(validation, config, "validation")
    if existing_validation is None:
        if args.phase == "test":
            raise ValueError("Complete validation first (p3 compare --phase validation)")
        run("evaluate", "--config", engine_profile, "--run-name", args.validation_run, *runtime)
        if not dry_run:
            existing_validation = completed_report(validation, config, "validation")
    else:
        print(f"Skip completed validation: {validation}", flush=True)
    if lock_path.exists():
        lock = read_json(lock_path)
        if (lock.get("fingerprint") != object_hash({k: v for k, v in lock.items() if k != "fingerprint"})
                or existing_validation is None or lock["config"] != existing_validation["config"]
                or lock["code"] != existing_validation["code"]
                or Path(lock["validation_report"]).resolve() != validation.resolve()
                or lock["report_sha256"] != file_hash(validation)):
            raise ValueError("Existing factorial validation lock differs; choose new run names")
        print(f"Reuse validation lock: {lock_path}", flush=True)
    else:
        run("lock", "--report", validation, "--output", lock_path)
    if args.phase != "validation":
        if completed_report(test, config, "test") is None:
            run("evaluate", "--split", "test", "--locked-config", lock_path,
                "--run-name", args.test_run, *runtime)
            if not dry_run:
                completed_report(test, config, "test")
        else:
            print(f"Skip completed test: {test}", flush=True)
        run("analyze", "--report", test)
    run("analyze", "--report", validation)
