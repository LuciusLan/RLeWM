"""One resource load, configured core treatments and optional real MPC."""

from contextlib import ExitStack
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import torch

from experiments.progress import progress
from experiments.provenance import backend_identity
from ..collection.config import headless
from ..interleave.cohorts import load_manifest, prepare
from ..interleave.continuations import ContinuationProviders
from ..outcomes.metrics import PlanOutcome, metric_contract
from ..outcomes.replay import FreshReplay
from ..planning.cohorts import planning_task, goal_latent
from ..planning.pipeline import action_coordinates
from ..rollout.cohorts import evaluation_bank
from ..rollout.record_io import read_saved_records
from ..rollout.resources import load_resources
from ..rollout.support_updates import synchronize
from ..runtime import file_hash, object_hash, read_json, write_json
from .config import STAGE, cells, groups, mid_search_enabled, primary_endpoint, validate
from .execution import execute
from .planning import first_plan_row, search
from .reporting import format_summary, summarize, write_tables


def case_value(bank, index, task, query):
    position = query["episode_metadata"]["ids"].index(task.metadata["episode_id"])
    return {"case_id": task.metadata["transition_id"], "query": task.metadata,
            "episode_position": position, "episode_start_raw_row": query["episode_metadata"]["offsets"][position],
            "start_raw_row": int(bank.items.starts[index]), "current_raw_row": task.current_row,
            "goal_raw_row": task.goal_row}


def evaluate(config, device, cpu_threads, split="validation", run_name=None, lock=None, resume=False):
    validate(config)
    implementation = backend_identity("p3_core")
    if split not in ("validation", "test"):
        raise ValueError("Core split must be validation or test")
    if split == "test" and (lock is None or lock["config"] != config or lock["code"] != implementation):
        raise ValueError("Core test requires an unchanged verified validation lock")
    if split == "validation" and lock is not None:
        raise ValueError("Validation cannot use a test lock")
    name = run_name or split
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Use a simple core run name")
    output = Path(config["experiment_dir"])/"runs"/name
    previous = None
    if output.exists():
        path = output/"report.json"
        if path.is_file():
            previous = verified_report(path)
            if previous["config"] == config and previous["split"] == split:
                if load_manifest(config)["fingerprint"] != previous["manifest_fingerprint"]:
                    raise ValueError("Completed core cohort changed")
                if previous["completed"]:
                    print(f"Skip verified completed core run: {path}", flush=True)
                    return previous
                if not resume or not previous.get("budget_exhausted"):
                    raise FileExistsError("Clean budget-stopped run preserved; use --resume or a new root")
            else:
                raise FileExistsError("Core config/split changed; preserve output and select a new root")
        else:
            raise FileExistsError("Interrupted core output has no verified stop receipt; preserve it and choose a new root")
    headless()
    # Metadata preparation only; never encodes observations or rebuilds memory.
    manifest = prepare(config)
    if lock and lock["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Locked fresh cohort changed")
    output.mkdir(parents=True, exist_ok=previous is not None)
    write_json(output/"config.json", config)
    started, replay = time.perf_counter(), None
    records = read_saved_records(previous) if previous else []
    completed_queries = previous["completed_queries"] if previous else 0
    accounting = {"search_calls": 0, "actual_cem_sequence_evaluations": 0, "extra_terminal_mean_probes": 0,
        "measurement_overhead_seconds": 0., "instrumented_search_seconds": 0., "checkpoint_counters": {},
        "diagnostic_sequence_evaluations": 0, "diagnostic_seconds": 0., "initial_success_searches_skipped": 0}
    if previous:
        accounting.update(deepcopy(previous["accounting"]))
    paths = {name: output/(name+".jsonl.gz") for name in ("records", "plans", "calls")}
    try:
        write_json(output/"status.json", {"status": "loading"})
        resources = load_resources(config, device, cpu_threads)
        runtime = {"device": str(resources.adapter.device), "cpu_threads": cpu_threads,
                   "gpu": torch.cuda.get_device_name(resources.adapter.device) if resources.adapter.device.type == "cuda" else None}
        if resources.external_manifest is None:
            raise ValueError("Core requires fully receipted fresh queries")
        coordinates = action_coordinates(resources, config)
        group_configs = groups(config)
        selected_cells = cells(config)
        providers = {name: ContinuationProviders(resources, value, coordinates) for name, value in group_configs.items()}
        bank = evaluation_bank(resources, manifest, split, config["evaluation"]["query_limit"])
        query = resources.external_manifest
        replay = FreshReplay(query, config, config["core"])
        if previous:
            prefix_ids = [bank.items[i]["transition_id"] for i in range(completed_queries)]
            if (previous["requested_queries"] != len(bank) or previous["model"] != resources.adapter.identity
                    or previous["runtime"] != runtime
                    or [row["case"]["case_id"] for row in records if row["kind"] == "recorded-control"] != prefix_ids):
                raise ValueError("Resume model, runtime, cohort or completed-query prefix differs")
        # Existing numeric acceptance checks are run once per initialization
        # group, outside per-arm timings. Physics replay is checked per query.
        from ..interleave.checks import run_checks
        checks = {name: run_checks(resources, evaluation_bank(resources, manifest, "validation", 1), value,
                                   coordinates, providers[name]) for name, value in group_configs.items()}
        synchronize(resources.adapter)
        if resources.adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resources.adapter.device)
        loading_seconds = time.perf_counter()-started
        write_json(output/"acceptance-checks.json", checks)
        segments = previous.get("segments", 1)+1 if previous else 1
        if previous:
            write_json(output/"segments"/f"report-{segments-1}.json", previous)
        with ExitStack() as stack:
            streams = {name: stack.enter_context(gzip.open(path, "at" if previous else "xt", encoding="utf-8")) for name, path in paths.items()}
            counts = ({"records": previous["record_count"], "plans": previous["plan_count"], "calls": previous["call_count"]}
                      if previous else {name: 0 for name in paths})
            def append(name, row):
                streams[name].write(json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n")
                counts[name] += 1
                if name == "records":
                    records.append(row)
            for index in progress(range(completed_queries, len(bank)), config=config, desc=f"P3 core {split}"):
                task = planning_task(bank, index, config["planning"]["horizon"])
                goal = goal_latent(bank, task)
                case = case_value(bank, index, task, query)
                initial, _ = replay.reset(case)
                initial_outcome = PlanOutcome(replay.metrics(initial, replay.goal(case)))
                skip = config["core"]["control_budget"] == 50 and initial_outcome.ever_success
                control = None
                for seed in config["p1"]["evaluation"]["seeds"]:
                    for group, value in group_configs.items():
                        for arm in value["interleave"]["arms"]:
                            condition = selected_cells[group, arm]
                            if not skip:
                                recorder, injection, audit = search(resources, value, providers[group], coordinates,
                                    task.context, task.metadata, goal, arm, seed, config["core"]["iterations"], accounting)
                            else:
                                accounting["initial_success_searches_skipped"] += 1
                            for j in config["core"]["iterations"]:
                                if skip:
                                    result = {**initial_outcome.record(), "planning_calls": 0, "planning_seconds": 0.,
                                        "context_encoding_seconds": 0., "compute": {}, "executed_raw_actions": [],
                                        "skipped_initial_success": True}
                                else:
                                    snapshot = recorder.snapshots[j]
                                    row = first_plan_row(recorder, snapshot, task, bank, index, arm, seed,
                                                         injection, audit, accounting)
                                    append("plans", {"iterations": j, "condition": condition, **row})
                                    result = execute(replay, case, task, goal, resources, value, providers[group],
                                        coordinates, arm, seed, snapshot, row, accounting,
                                        lambda call: append("calls", call), condition)
                                if control is None:
                                    # Truth is a post-search reconstruction control,
                                    # never a CEM/adaptation/retrieval input.
                                    control = replay.execute(case)
                                    append("records", {"kind": "recorded-control", "case": case, **control})
                                if result["initial_metrics"] != control["initial_metrics"]:
                                    raise ValueError("Core paired treatments received different starting states/goals")
                                append("records", {"kind": "planned", "case": case, "iterations": j,
                                    "condition": condition, "seed": seed, **result})
                completed_queries += 1
                for stream in streams.values():
                    stream.flush()
                write_json(output/"status.json", {"status": "running", "completed_queries": completed_queries,
                    "requested_queries": len(bank), "accounting": accounting})
                # Always retain a complete paired query across cells, seeds and J.
                if time.perf_counter()-started >= config["core"]["runtime_budget_hours"]*3600:
                    break
        complete = completed_queries == len(bank)
        if backend_identity("p3_core") != implementation:
            raise ValueError("Core source changed during evaluation; partial records preserved without a completed scientific report")
        segment_seconds = time.perf_counter()-started
        elapsed = segment_seconds+(previous["evaluation_seconds"] if previous else 0.)
        loading_seconds += previous["resource_loading_and_checks_seconds"] if previous else 0.
        contract = metric_contract(replay.backend, config["core"]["pusht_success"], config["core"]["coverage_threshold"])
        contract.update(primary_endpoint=("final supplied-goal success after exactly 25 controls" if primary_endpoint(config) == "final"
                                         else "ever supplied-goal success within 50 controls; stop at first success, including initial success"))
        accounting["actual_cem_transition_predictions"] = accounting["actual_cem_sequence_evaluations"]*config["planning"]["horizon"]
        gpu_memory = {}
        if resources.adapter.device.type == "cuda":
            for name, value in (("torch_peak_allocated_bytes", torch.cuda.max_memory_allocated(resources.adapter.device)),
                                ("torch_peak_reserved_bytes", torch.cuda.max_memory_reserved(resources.adapter.device))):
                gpu_memory[name] = max(value, previous.get("gpu_memory", {}).get(name, 0) if previous else 0)
            gpu_memory["scope"] = "Maximum over recorded segments; Torch allocator only, some native cuVS allocations excluded"
        from ..cache_loading import report_fields
        report = {**report_fields(resources, replay=True), "stage": STAGE, "completed": complete, "budget_exhausted": not complete, "split": split,
            "config": config, "code": implementation, "manifest_fingerprint": manifest["fingerprint"],
            "p1_prepared_fingerprint": resources.prepared["fingerprint"], "query_source_fingerprint": query["fingerprint"],
            "query_manifest": config["query_dataset"]["manifest"],
            "query_manifest_sha256": file_hash(Path(config["query_dataset"]["manifest"])),
            "model": resources.adapter.identity, "scope_parameters": resources.adapter.scope_description(),
            "runtime": runtime,
            "gpu_memory": gpu_memory,
            "retrieval": {name: provider.description() for name, provider in providers.items()},
            "action_coordinates": coordinates.description(), "environment": replay.description, "checks": checks,
            "cells": list(selected_cells.values()), "control_budget": config["core"]["control_budget"],
            "completed_queries": completed_queries, "requested_queries": len(bank),
            "full_cohort_queries": manifest["splits"][split]["queries"],
            "limited_run": config["evaluation"]["query_limit"] is not None, "accounting": accounting,
            "resource_loading_and_checks_seconds": loading_seconds, "evaluation_seconds": elapsed,
            "segments": segments, "last_segment_seconds": segment_seconds,
            "rough_full_cohort_hours": (loading_seconds+(elapsed-loading_seconds)/completed_queries*manifest["splits"][split]["queries"])/3600,
            "cost_estimate_scope": "Extrapolation from completed queries on this runtime; exploratory, especially for small smokes or 50-control early termination. Includes loading, checks, planning and CPU physics while GPU resources remain allocated; no performance guarantee.",
            "records_file": str(paths["records"].resolve()), "records_sha256": file_hash(paths["records"]),
            "record_count": counts["records"], "plans_file": str(paths["plans"].resolve()),
            "plans_sha256": file_hash(paths["plans"]), "plan_count": counts["plans"],
            "calls_file": str(paths["calls"].resolve()), "calls_sha256": file_hash(paths["calls"]), "call_count": counts["calls"],
            "protocol": ("Six paired cells: Gaussian / retrieval initialization / initialization plus mid-search retrieval, "
                         if mid_search_enabled(config) else "Four paired cells: Gaussian / retrieval initialization, ")
                +"crossed with Frozen / retrieved checkpoint adaptation. "
                +("One 25-control open-loop execution; final success is primary. " if config["core"]["control_budget"] == 25 else
                  "Up to 50 actual controls; replan every 25 using real observed history; ever-success is primary and execution stops on success. ")
                +"First-call J snapshots share one maximum-J search; divergent real replans search independently at their own fixed J. "
                 "Pretrained predictor and fresh optimizer reset every real call; fixed observed goal throughout; original reset and complete past prefix verified. "
                 "Effective episode planning time includes live encoding; actual shared instrumentation is charged once. Fresh queries, history, bounds and normalization remain our protocol, not a full LeWM reproduction."}
        report["summary"] = summarize(records, config)
        write_json(output/"report.json", report)
        write_tables(output, report)
        write_json(output/"status.json", {"status": "completed" if complete else "budget-stopped",
                   "completed_queries": completed_queries, "requested_queries": len(bank), "accounting": accounting})
        print(format_summary(report), flush=True)
        print(f"Core report: {output/'report.json'}", flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "completed_queries": completed_queries, "accounting": accounting})
        raise
    finally:
        if replay is not None:
            replay.close()


def verified_report(path):
    report = read_json(path)
    if report.get("stage") != STAGE or report.get("code") != backend_identity("p3_core"):
        raise ValueError("Core report/source mismatch; use its verified archived implementation")
    for key in ("records", "plans", "calls", "query_manifest"):
        filename = report[key+"_file"] if key != "query_manifest" else report[key]
        if file_hash(Path(filename)) != report[key+"_sha256"]:
            raise ValueError(f"Core {key} changed")
    rows = read_saved_records(report)
    if summarize(rows, report["config"]) != report["summary"]:
        raise ValueError("Core records no longer reproduce the saved summary")
    return report


def analyze(path, output=None):
    report = verified_report(path)
    if output is not None:
        if Path(output).exists():
            raise FileExistsError("Preserve existing files; choose a new analysis output")
        Path(output).write_text(format_summary(report))
    print(format_summary(report), flush=True)


def lock_validation(path, output):
    if Path(output).exists():
        raise FileExistsError("Preserve the existing lock")
    report = verified_report(path)
    if report["split"] != "validation" or not report["completed"] or report["limited_run"]:
        raise ValueError("Only complete unlimited core validation may be locked")
    manifest = load_manifest(report["config"])
    if (report["completed_queries"] != manifest["splits"]["validation"]["queries"]
            or manifest["fingerprint"] != report["manifest_fingerprint"]):
        raise ValueError("Core validation does not cover the full reserved cohort")
    value = {"stage": STAGE, "config": report["config"], "code": report["code"],
             "manifest_fingerprint": manifest["fingerprint"], "validation_report": str(Path(path).resolve()),
             "report_sha256": file_hash(Path(path))}
    value["fingerprint"] = object_hash(value)
    write_json(output, value)
