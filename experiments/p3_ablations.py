"""Frozen-only fresh-validation action-proposal controls and sensitivities.

Orchestration and saved-record verification are model-free. Every numerical
operation goes through the existing P3 proposal engine; test is never launched.
"""

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.collection.manifest import checked
from lewm_tta.rollout.record_io import open_records
from lewm_tta.runtime import file_hash, object_hash, read_json, write_json
from .catalog import ROOT
from .config import materialize, resolve
from .fresh import effective_config, query_ids
from .provenance import archive_entry, backend_identity


def plan_jobs(base, slots, iterations, groups):
    """One-factor changes; the controls run supplies the common default point."""
    baseline = deepcopy(base)
    baseline["comparison"] = "proposal-only"
    baseline["adaptation"]["methods"] = ["frozen"]
    baseline["proposals"]["sources"] = ["gaussian", "uniform-fragments", "context-fragments"]
    baseline["reuse"] = {}
    result = [("controls", baseline)]
    if "share" in groups:
        for count in slots:
            if count == base["proposals"]["injection_slots"]:
                continue
            config = deepcopy(baseline)
            config["proposals"].update(sources=["gaussian", "context-fragments"], injection_slots=count)
            result.append((f"share-{count}", config))
    if "budget" in groups:
        for count in iterations:
            if count == base["planning"]["iterations"]:
                continue
            config = deepcopy(baseline)
            config["proposals"]["sources"] = ["gaussian", "context-fragments"]
            config["planning"]["iterations"] = count
            result.append((f"iterations-{count}", config))
    if "goal" in groups:
        # One fixed operating point; never cross goal retrieval with the grids.
        config = deepcopy(baseline)
        config["planning"]["iterations"] = 6
        config["proposals"].update(
            sources=["gaussian", "context-fragments", "goal-context-fragments"],
            fragments=8, injection_slots=64, noise_std=.1, goal_weight=1.)
        result.append(("goal", config))
    return result


def completed_report(path, expected):
    if not path.exists():
        if path.parent.exists():
            raise ValueError(f"Partial run preserved; choose another --run-prefix: {path.parent}")
        return None
    report = read_json(path)
    if (not report.get("completed") or report.get("stage") != "P3b-offline-proposals"
            or report.get("split") != "validation" or report.get("config") != expected
            or report.get("limited_run") != (expected["evaluation"]["query_limit"] is not None)):
        raise ValueError(f"Existing ablation uses another protocol; choose another --run-prefix: {path}")
    manifest = checked(Path(expected["experiment_dir"]) / "manifest.json")
    query = checked(Path(expected["query_dataset"]["manifest"]))
    if (manifest["fingerprint"] != report["manifest_fingerprint"]
            or manifest["splits"]["validation"] != report["cohort"]
            or manifest.get("query_source_fingerprint") != query["fingerprint"]):
        raise ValueError("Existing ablation belongs to another prepared cohort")
    if (report["code"] != backend_identity("p3b")
            and archive_entry("p3b", report["code"]) is None):
        raise ValueError("Existing ablation has no verified execution source")
    starts = manifest["splits"]["validation"]["raw_start_rows"]
    limit = expected["evaluation"]["query_limit"]
    ids = query_ids(query, starts if limit is None else starts[:limit])
    matrix = set(product(ids, expected["p1"]["evaluation"]["seeds"], ["frozen"], expected["proposals"]["sources"]))
    records = Path(report.get("records_file") or "")
    if (not records.is_file() or records.parent.resolve() != path.parent.resolve()
            or not records.stat().st_size or file_hash(records) != report.get("records_sha256")):
        raise ValueError("Existing ablation records are missing or changed")
    seen = set()
    with open_records(records) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["query"]["transition_id"], row["seed"], row["method"], row["proposal_source"]
            if key not in matrix or key in seen or row["adaptation"]["updates"]:
                raise ValueError("Incomplete, duplicate or non-Frozen ablation records")
            seen.add(key)
    if seen != matrix or len(seen) != report["record_count"]:
        raise ValueError("Existing ablation has an incomplete query/seed/source matrix")
    from lewm_tta.proposals.reuse import verify_dependencies
    verify_dependencies(report)
    return report


def write_overview(jobs, directory, prefix):
    cells = []
    for job in jobs:
        report = read_json(job["report"])
        accumulated = {}
        with open_records(Path(report["records_file"])) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = row["proposal_source"]
                item = accumulated.setdefault(source, {"count": 0, "reused_calls": 0, "new_candidate_evaluations": 0,
                    "best_sampled_sse_sum": 0., "best_evaluated_sse_sum": 0., "convergence_sums": []})
                item["count"] += 1
                item["reused_calls"] += row["result_origin"] == "reused"
                item["new_candidate_evaluations"] += row["new_cem_candidate_evaluations"]
                curve = [step["best_scored_cost_so_far"] for step in row["cem_trace"]]
                item["best_sampled_sse_sum"] += curve[-1]
                item["best_evaluated_sse_sum"] += min(curve[-1], row["selected_model_goal_sse"])
                if not item["convergence_sums"]:
                    item["convergence_sums"] = [0.] * len(curve)
                for position, value in enumerate(curve):
                    item["convergence_sums"][position] += value
        for source, item in accumulated.items():
            search = report["summary"]["by_proposal"][source]["methods"]["frozen"]["search_diagnostics_not_physical_performance"]
            count = item["count"]
            cells.append({"job": job["label"], "source": source, "report": str(job["report"]),
                "report_sha256": file_hash(job["report"]), "iterations": report["config"]["planning"]["iterations"],
                "population": report["config"]["planning"]["population"],
                "injection_slots": 0 if source == "gaussian" else report["config"]["proposals"]["injection_slots"],
                "goal_weight": report["config"]["proposals"]["goal_weight"] if source == "goal-context-fragments" else None,
                "returned_mean_sse": search["selected_model_goal_sse"],
                "best_sampled_sse": item["best_sampled_sse_sum"] / count,
                "best_evaluated_sse": item["best_evaluated_sse_sum"] / count,
                "best_sampled_sse_by_iteration": [value / count for value in item["convergence_sums"]],
                "candidate_evaluations_per_call": search["cem_candidate_evaluations"],
                "primary_seconds": search["primary_seconds_with_checks"], "cem_seconds": search["cem_seconds"],
                "primary_seconds_p95": search["primary_seconds_p95"], "calls": count,
                "reused_calls": item["reused_calls"], "new_candidate_evaluations": item["new_candidate_evaluations"],
                "paired_effects": report["summary"]["paired_proposal_effects"]["frozen"]})
    overview = {"stage": "P3-Frozen-proposal-ablations", "split": "validation", "cells": cells,
        "interpretation": "Offline model-search diagnostics; no simulator execution or physical success measurement.",
        "timing": "Imported cells retain source-run primary/CEM latencies; mixed runs are not matched-wall-clock trials.",
        "return_rule": "Final CEM mean unchanged. Best sampled/evaluated costs are diagnostics, not alternative executed plans."}
    target = directory / f"{prefix}-overview.json"
    write_json(target, overview)
    lines = ["# Frozen P3 proposal ablations", "", overview["interpretation"], "", 
        "| Job | Source | Goal weight | Slots | Iterations | Returned mean SSE | Best sampled SSE | Seconds/call | Reused calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        weight = "—" if cell["goal_weight"] is None else f"{cell['goal_weight']:g}"
        lines.append(f"| {cell['job']} | {cell['source']} | {weight} | {cell['injection_slots']} | {cell['iterations']} | "
                     f"{cell['returned_mean_sse']:.6f} | {cell['best_sampled_sse']:.6f} | "
                     f"{cell['primary_seconds']:.3f} | {cell['reused_calls']}/{cell['calls']} |")
    lines += ["", "Within-job paired effects and episode-bootstrap intervals are in each report/summary.md.", "",
              overview["timing"], "", overview["return_rule"], ""]
    target.with_suffix(".md").write_text("\n".join(lines))
    print(f"Ablation overview: {target}", flush=True)


def ablate(profile_path, arguments, dry_run=False, *, dataset=None, cache_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=("controls", "share", "budget", "goal"),
                        default=["controls", "share", "budget", "goal"],
                        help="Controls are always included; goal adds one fixed J6/64-slot/lambda1 comparison")
    parser.add_argument("--injection-slots", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--iterations", nargs="+", type=int, default=[2, 6, 12])
    parser.add_argument("--reference-report", type=Path, help="Default: this dataset's fresh six-cell validation report")
    parser.add_argument("--no-reuse", action="store_true", help="Compute every cell instead of importing baseline searches")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--limit", type=int, help="Integration only; limited runs cannot be locked")
    parser.add_argument("--run-prefix", default="proposal-ablation")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--analyze-only", action="store_true", help="Verify completed runs and regenerate the overview without a model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(arguments)
    for values in (args.groups, args.injection_slots, args.iterations):
        if len(set(values)) != len(values):
            raise ValueError("Choose distinct ablation groups, injection counts and iteration counts")
    if (not args.run_prefix or Path(args.run_prefix).name != args.run_prefix or args.run_prefix in (".", "..")
            or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1)):
        raise ValueError("Use a simple run prefix and positive limits/threads")
    base = resolve(profile_path, dataset=dataset, cache_only=cache_only)
    if args.reference_report is None:
        task = effective_config(base)["p1"]["data"].get("dataset_id", "pusht")
        if task == "pusht":
            directory = ROOT/"artifacts/p3-fresh-pusht-v2"
        else:
            from .datasets import dataset_profile
            directory = ROOT/dataset_profile(task)["experiment_root"]/"p3"
        args.reference_report = directory/"runs/fresh-validation/report.json"
    if base.get("comparison") != "proposal-only" or not base.get("query_dataset"):
        raise ValueError("Use the Frozen fresh-proposals profile for this ablation")
    if args.experiment_dir:
        base["experiment_dir"] = str(args.experiment_dir.resolve())
    base["experiment_dir"] = str((ROOT / base["experiment_dir"]).resolve())
    base["evaluation"].update(query_limit=args.limit, records="compact")
    seeds = args.seeds if args.seeds is not None else effective_config(base)["p1"]["evaluation"]["seeds"]
    if min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct nonnegative seeds")
    reference = None
    if not args.no_reuse:
        if not args.reference_report.is_file():
            raise FileNotFoundError(f"Complete this task's six-cell validation first, supply --reference-report, or use --no-reuse: {args.reference_report}")
        reference = read_json(args.reference_report)
        reference_path = args.reference_report.resolve()
        pin = {"report": str(reference_path), "sha256": file_hash(reference_path)}
        query = checked(Path(base["query_dataset"]["manifest"]))
        previous_manifest = checked(Path(reference["config"]["experiment_dir"]) / "manifest.json")
        if previous_manifest["splits"]["validation"]["raw_start_rows"] != query["evaluation_windows"]["validation"]["raw_start_rows"]:
            raise ValueError("Reference report uses different fresh validation starts")
        prepared = checked(Path(effective_config(base)["p1"]["paths"]["experiment_dir"]) / "prepared.json")
        reference_manifest = {**previous_manifest, "p1_prepared_fingerprint": prepared["fingerprint"],
                              "query_source_fingerprint": query["fingerprint"]}
    # Keep numerical imports out of --help and orchestration-only module loading.
    from lewm_tta.proposals.config import validate_config
    from lewm_tta.proposals.reuse_frozen import load_frozen_cells
    jobs = []
    for label, config in plan_jobs(base, args.injection_slots, args.iterations, args.groups):
        reusable = set()
        if reference is not None and config["planning"] == reference["config"]["planning"]:
            config["reuse"] = {"validation": [pin]}
            expected = effective_config(config)
            expected["p1"]["evaluation"]["seeds"] = list(seeds)
            rows = load_frozen_cells(reference_path, reference, expected, "validation", reference_manifest)
            reusable = {key[3] for key in rows}
        expected = effective_config(config)
        expected["p1"]["evaluation"]["seeds"] = list(seeds)
        validate_config(expected)
        name = f"{args.run_prefix}-{label}" + (f"-smoke{args.limit}" if args.limit is not None else "")
        report = Path(config["experiment_dir"]) / "runs" / name / "report.json"
        done = completed_report(report, expected)
        jobs.append({"label": label, "config": config, "expected": expected, "name": name, "report": report,
                     "done": done is not None, "reusable": reusable})
    print("P3 Frozen-only validation ablations; no adaptation or LR sweep; controls are always included.", flush=True)
    print(f"Seeds: {seeds}; query limit: {args.limit}; noise std: {base['proposals']['noise_std']}; fragments: {base['proposals']['fragments']}", flush=True)
    for job in jobs:
        config = job["config"]
        new = set(config["proposals"]["sources"]) - job["reusable"]
        print(f"{job['name']}: J={config['planning']['iterations']}, slots={config['proposals']['injection_slots']}, "
              f"reuse={sorted(job['reusable'])}, new={sorted(new)}, completed={job['done']}", flush=True)
    if args.analyze_only and any(not job["done"] for job in jobs):
        raise ValueError("Complete the selected ablation jobs before --analyze-only")
    command_prefix = [sys.executable, "-X", "faulthandler", "-u", str(ROOT / "codes/experiment.py"), "p3"]
    runtime = ["--device", args.device, "--cpu-threads", str(args.cpu_threads)] + (["--verbose"] if args.verbose else [])

    def run(action, *options):
        command = [*command_prefix, action, *map(str, options)]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    pending = [job for job in jobs if not job["done"]]
    if pending:
        config = pending[0]["config"]
        path = ROOT / "artifacts/experiment-configs" / f"{object_hash(config)}.json" if dry_run else materialize(config)
        run("prepare", "--config", path)
    for job in jobs:
        if job["done"]:
            print(f"Skip verified completed ablation: {job['report']}", flush=True)
            continue
        config = job["config"]
        path = ROOT / "artifacts/experiment-configs" / f"{object_hash(config)}.json" if dry_run else materialize(config)
        run("evaluate", "--config", path, "--split", "validation", "--run-name", job["name"], "--seeds", *seeds, *runtime)
        if not dry_run and completed_report(job["report"], job["expected"]) is None:
            raise ValueError("Ablation evaluation did not write a completed report")
    if not dry_run:
        write_overview(jobs, Path(base["experiment_dir"]), args.run_prefix)
    print(f"{'Planned' if dry_run else 'Completed'} P3 ablations: {base['experiment_dir']}", flush=True)
