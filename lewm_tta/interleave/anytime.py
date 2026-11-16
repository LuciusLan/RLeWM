"""One CEM search per query/seed/arm; ordinary saved-plan reports at each stop."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import time

import torch

from experiments.progress import progress
from ..budget_receipts import STAGE as CAPTURE_STAGE
from ..planning.actions import stream_seed
from ..planning.cohorts import planning_task, goal_latent
from ..planning.pipeline import action_coordinates
from ..planning.config import evaluation_limits
from ..planning.cem import score_sequences
from ..rollout.cohorts import evaluation_bank, rollout_inputs, observed_targets
from ..rollout.inference import fixed_predictor, latent_errors
from ..rollout.record_io import RecordWriter, record_mode
from ..rollout.resources import load_resources
from ..rollout.support_updates import synchronize
from ..runtime import file_hash, object_hash, read_json, write_json
from .cem import solve_interleaved_cem
from .cohorts import load_manifest
from .config import STAGE, code_identity, validate_config
from .continuations import ContinuationProviders
from .reporting import summarize, format_summary
from .session import CheckpointSession
from .snapshots import SnapshotRecorder


def stopped_config(config, iterations, experiment_dir):
    value = deepcopy(config)
    value["experiment_dir"] = str(Path(experiment_dir).resolve())
    value["planning"]["iterations"] = iterations
    value["interleave"]["checkpoints"] = [j for j in config["interleave"]["checkpoints"] if j < iterations]
    return value


def stopped_retrieval_identity(identity, experiment_dir):
    """Preserve calibration contents at each stop's ordinary lock-compatible path."""
    result = deepcopy(identity)
    def visit(value):
        if not isinstance(value, dict):
            return
        if value.get("calibration_file"):
            target = Path(experiment_dir)/"retrieval"/Path(value["calibration_file"]).name
            if target.exists() and read_json(target) != value["calibration"]:
                raise ValueError("Existing stopped-budget endpoint calibration differs")
            if not target.exists():
                write_json(target, value["calibration"])
            value["calibration_file"] = str(target.resolve())
        for child in value.values():
            if isinstance(child, dict):
                visit(child)
    visit(result)
    return result


def diagnostic_row(recorder, snapshot, task, bank, index, arm, seed, initial, initial_audit):
    session, plan = recorder.session, snapshot.plan
    adapter, config = session.adapter, session.config
    coordinates, settings = session.coordinates, config["planning"]
    started = time.perf_counter()
    recorder.restore(snapshot)  # Search is over; no optimizer step follows this.
    _, recorded_actions, target_rows = rollout_inputs(bank, index, settings["horizon"])
    observed = observed_targets(bank, target_rows)
    torch.testing.assert_close(observed[:, -1], session.goal, rtol=0, atol=0)
    with fixed_predictor(adapter):
        predicted = adapter.rollout(task.context, recorded_actions)
        selected = adapter.rollout(task.context, plan.actions[None])
    own_sse = float((selected[:, -1]-session.goal).square().sum())
    if abs(own_sse-plan.cost) > 1e-5*max(1., abs(plan.cost)):
        raise AssertionError("Restored snapshot predictor does not reproduce its terminal-mean score")
    raw = coordinates.to_raw(plan.actions)
    recorded_raw = coordinates.to_raw(recorded_actions[0])
    if ((raw < coordinates.raw_low-1e-6) | (raw > coordinates.raw_high+1e-6)).any():
        raise AssertionError("Snapshot raw controls violate configured bounds")
    adapter.reset(session.reset_seed)
    with fixed_predictor(adapter):
        frozen_score = float(score_sequences(adapter, task.context, session.goal, plan.actions[None],
                                             settings["scoring_batch_size"])[0])
    synchronize(adapter)
    return {"arm": arm, "seed": seed, "scope": config["scopes"][0], "query": task.metadata,
        "current_raw_row": task.current_row, "goal_raw_row": task.goal_row, "target_raw_rows": target_rows.tolist(),
        "cem_seed": stream_seed(seed, task.metadata["transition_id"], "cem"),
        "initial_proposals": initial_audit, "initial_injection_fingerprint": initial.fingerprint if initial is not None else None,
        "cem_first_population_fingerprint": plan.first_population_fingerprint,
        "cem_trace": plan.trace, "checkpoints": snapshot.checkpoints, "adaptation": snapshot.adaptation,
        "counters": snapshot.counters, "timing_seconds": snapshot.timing_seconds,
        "selected_actions_normalized": plan.actions.cpu().tolist(), "selected_raw_actions": raw.flatten(0, 1).cpu().tolist(),
        "selected_predicted_endpoint": selected[0, -1].cpu().tolist(), "selected_model_goal_sse": plan.cost,
        "selected_plan_frozen_model_goal_sse": frozen_score, "recorded_action_mse_by_block": latent_errors(predicted, observed),
        "selected_vs_recorded_raw_action_mse": float((raw-recorded_raw).square().mean()),
        "cem_candidate_evaluations": plan.candidate_evaluations,
        "cem_transition_predictions": plan.candidate_evaluations*settings["horizon"],
        "diagnostic_sequence_evaluations": 3, "diagnostic_transition_predictions": 3*settings["horizon"],
        "primary_seconds_with_checks": snapshot.primary_seconds,
        "diagnostic_seconds_with_checks": time.perf_counter()-started,
        "budget_snapshot": {"iterations": snapshot.iterations, **snapshot.capture},
        "checks": {"model_fixed_within_population": True, "query_local_reset": True, "global_support_budget": True,
            "observed_memory_targets_only": True, "recorded_future_excluded_from_search": True,
            "supplied_goal_not_an_adaptation_target": True, "source_episode_safe_suffixes": True,
            "full_candidates_rerolled_from_actual_context": True, "returned_mean_scored_under_final_model": True,
            "versioned_scores": True, "snapshot_before_equal_iteration_checkpoint": True,
            "diagnostics_only_after_maximum_search": True}}


def evaluate(config, device, cpu_threads, iterations, root, run_name):
    from .checks import run_checks
    iterations = list(iterations)
    if (not iterations or iterations != sorted(set(iterations)) or any(type(j) is not int or j < 1 for j in iterations)
            or iterations[-1] != config["planning"]["iterations"]):
        raise ValueError("Single-pass stopping budgets must be sorted, distinct and end at the configured maximum")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Use a simple run name")
    validate_config(config)
    if record_mode(config) == "none":
        raise ValueError("Single-pass stopping reports require compact or full saved records")
    root = Path(root).resolve()
    master_dir = root/"snapshots"/run_name
    jobs = {}
    for j in iterations:
        selected = stopped_config(config, j, root/"planning"/f"j-{j}")
        validate_config(selected)
        manifest = load_manifest(selected)
        output = Path(selected["experiment_dir"])/"runs"/run_name
        if output.exists():
            raise FileExistsError(f"Snapshot output exists; preserve it or choose another root: {output}")
        jobs[j] = {"config": selected, "manifest": manifest, "output": output, "records": []}
    if master_dir.exists():
        raise FileExistsError(f"Shared search output already exists: {master_dir}")
    master_dir.mkdir(parents=True)
    implementation = code_identity()
    started = time.perf_counter()
    totals = {"search_calls": 0, "actual_cem_sequence_evaluations": 0,
              "actual_cem_transition_predictions": 0, "diagnostic_sequence_evaluations": 0,
              "diagnostic_transition_predictions": 0, "checkpoint_counters": {},
              "extra_terminal_mean_probes": 0, "measurement_overhead_seconds": 0., "instrumented_search_seconds": 0.}
    try:
        loading = time.perf_counter()
        resources = load_resources(config, device, cpu_threads)
        adapter, settings = resources.adapter, config["planning"]
        coordinates = action_coordinates(resources, config)
        providers = ContinuationProviders(resources, config, coordinates)
        provider_identity = providers.description()
        bank = evaluation_bank(resources, jobs[iterations[-1]]["manifest"], "validation", config["evaluation"]["query_limit"])
        synchronize(adapter)
        loading_seconds = time.perf_counter()-loading
        check_started = time.perf_counter()
        checks = run_checks(resources, evaluation_bank(resources, jobs[iterations[-1]]["manifest"], "validation", 1),
                            config, coordinates, providers)
        synchronize(adapter)
        check_seconds = time.perf_counter()-check_started
        write_json(master_dir/"acceptance-checks.json", checks)
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        with ExitStack() as stack:
            for job in jobs.values():
                job["output"].mkdir(parents=True)
                write_json(job["output"]/"config.json", job["config"])
                job["writer"] = stack.enter_context(RecordWriter(job["output"], job["config"]))
            for index in progress(range(len(bank)), config=config, desc="P3 single-pass budgets"):
                task = planning_task(bank, index, settings["horizon"])
                goal = goal_latent(bank, task)
                for seed in resources.config["evaluation"]["seeds"]:
                    for arm in config["interleave"]["arms"]:
                        synchronize(adapter)
                        primary_started = time.perf_counter()
                        session = CheckpointSession(resources.engine, task.context, task.metadata, goal, coordinates,
                                                    providers, config, arm, seed)
                        initial_started = time.perf_counter()
                        initial, initial_audit = providers.initial_bundle(adapter, task.context, task.metadata, seed, goal)
                        synchronize(adapter)
                        session.timings["initial_proposals"] = time.perf_counter()-initial_started
                        session.counters["initial_action_retrievals"] = int(initial is not None)
                        recorder = SnapshotRecorder(session, primary_started)
                        solve_interleaved_cem(session.score, settings, coordinates,
                            stream_seed(seed, task.metadata["transition_id"], "cem"), config["interleave"]["checkpoints"],
                            session, initial, snapshot_iterations=iterations, on_snapshot=recorder)
                        synchronize(adapter)
                        totals["instrumented_search_seconds"] += time.perf_counter()-primary_started
                        totals["search_calls"] += 1
                        actual = settings["population"]*settings["iterations"]+1+recorder.extra_terminal_probes
                        totals["actual_cem_sequence_evaluations"] += actual
                        totals["actual_cem_transition_predictions"] += actual*settings["horizon"]
                        totals["diagnostic_sequence_evaluations"] += 3*len(iterations)
                        totals["diagnostic_transition_predictions"] += 3*len(iterations)*settings["horizon"]
                        for name, value in session.counters.items():
                            totals["checkpoint_counters"][name] = totals["checkpoint_counters"].get(name, 0)+value
                        totals["extra_terminal_mean_probes"] += recorder.extra_terminal_probes
                        totals["measurement_overhead_seconds"] += recorder.overhead_seconds
                        if set(recorder.snapshots) != set(iterations):
                            raise AssertionError("CEM omitted a requested stopping point")
                        # Future actions/observations become accessible only now.
                        for j in iterations:
                            row = diagnostic_row(recorder, recorder.snapshots[j], task, bank, index, arm, seed, initial, initial_audit)
                            job = jobs[j]
                            job["records"].append(job["writer"].append(row))
                for job in jobs.values():
                    job["writer"].flush()
                write_json(master_dir/"status.json", {"status": "running", "queries_completed": index+1,
                    "queries": len(bank), **totals})
        # The master pins records/configs, not child report hashes: children can
        # pin the completed master without introducing a checksum cycle.
        for job in jobs.values():
            job["summary"] = summarize(job["records"], job["config"])
            if job["summary"]["queries"] != len(bank):
                raise ValueError("Snapshot summary does not cover the selected cohort")
        entries = [{"iterations": j, "report": str(job["output"]/"report.json"),
                    "records_file": str(job["writer"].path.resolve()), "records_sha256": file_hash(job["writer"].path),
                    "record_count": len(job["records"]), "config_fingerprint": object_hash(job["config"]),
                    "manifest_fingerprint": job["manifest"]["fingerprint"]} for j, job in jobs.items()]
        master = {"completed": True, "stage": CAPTURE_STAGE, "split": "validation", "iterations": iterations,
            "code": implementation, "config": config, "reports": entries, "accounting": totals,
            "model": adapter.identity, "retrieval": provider_identity,
            "p1_prepared_fingerprint": resources.prepared["fingerprint"],
            "evaluation_seconds": time.perf_counter()-started, "resource_loading_seconds": loading_seconds,
            "acceptance_check_seconds": check_seconds,
            "gpu_memory": ({"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
                            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device),
                            "note": "Shared instrumented search and diagnostics; Torch allocator only, some native cuVS allocations excluded"}
                           if adapter.device.type == "cuda" else {}),
            "timing_scope": "Actual instrumented elapsed/work is charged once here; budget report prefix timings exclude earlier measurement overhead and include their own terminal-mean score"}
        master_path = master_dir/"report.json"
        write_json(master_path, master)
        for entry in entries:
            j = entry["iterations"]
            job = jobs[j]
            capture = {"mode": "single-pass", "requested_iterations": iterations, "max_iterations": iterations[-1],
                       "master_report": str(master_path), "master_report_sha256": file_hash(master_path)}
            report = {"completed": True, "stage": STAGE, "split": "validation", "config": job["config"],
                "code": implementation, "manifest_fingerprint": job["manifest"]["fingerprint"],
                "p1_prepared_fingerprint": resources.prepared["fingerprint"], "model": adapter.identity,
                "scope_parameters": adapter.scope_description(), "action_coordinates": coordinates.description(),
                "retrieval": stopped_retrieval_identity(provider_identity, job["config"]["experiment_dir"]),
                "checks": checks, "summary": job["summary"],
                "cohort": job["manifest"]["splits"]["validation"], "limits": evaluation_limits(job["config"]),
                "comparison_protocol": {"model_lifetime": "reset once per query/seed/arm; carry optimizer across checkpoints",
                    "scoring": "one shared predictor at every depth and for every candidate within a CEM phase",
                    "support": "balanced subsampling of elite/depth query pairs, then one global observed-support budget",
                    "proposals": "updated-context suffix retrieval, complete candidates scored from the actual context",
                    "pairing": "identical warm-up; independent CEM, reference, replay and perturbation streams",
                    "selected_actions_executed": False, "budget_capture": "one maximum-J search with prefix stop plans"},
                "records_file": entry["records_file"], "records_sha256": entry["records_sha256"],
                "record_count": entry["record_count"], "record_storage": job["writer"].metadata(),
                "limited_run": config["evaluation"]["query_limit"] is not None, "budget_capture": capture,
                "resource_loading_seconds": None, "acceptance_check_seconds": None, "evaluation_seconds": None,
                "timing_scope": "Measured single-pass prefix latency excluding earlier snapshots/probes; includes the stop's terminal-mean score. Model copies and offline diagnostics excluded; actual total work/time is in the shared receipt.",
                "query_source_fingerprint": job["manifest"]["query_source_fingerprint"]}
            write_json(job["output"]/"report.json", report)
            (job["output"]/"summary.md").write_text(format_summary(report))
            write_json(job["output"]/"status.json", {"status": "completed", "records": entry["record_count"], "budget_capture": capture})
        write_json(master_dir/"status.json", {"status": "completed", **totals})
        print(f"Single-pass receipt: {master_path}", flush=True)
        return master
    except BaseException as error:
        write_json(master_dir/"status.json", {"status": "failed", "error": str(error), **totals})
        raise


def main(argv):
    import argparse
    from ..runtime import configure_logging
    from .config import load_config
    parser = argparse.ArgumentParser(description="Capture exact stopping plans in one maximum-budget interleave search")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-iterations", nargs="+", type=int, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if args.cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    evaluate(load_config(args.config), args.device, args.cpu_threads, args.snapshot_iterations,
             args.snapshot_root, args.run_name)
