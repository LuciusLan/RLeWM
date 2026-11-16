"""Separate P3 evaluation, saved action plans, reanalysis, and validation locks."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from experiments.progress import progress
from ..planning.actions import stream_seed
from ..planning.cohorts import planning_task, goal_latent
from ..planning.config import evaluation_limits
from ..planning.pipeline import action_coordinates
from ..rollout.cohorts import evaluation_bank, rollout_inputs, observed_targets
from ..rollout.inference import fixed_predictor, latent_errors
from ..rollout.record_io import RecordWriter, read_saved_records
from ..rollout.resources import load_resources
from ..rollout.support_updates import synchronize
from ..runtime import file_hash, object_hash, read_json, write_json
from .cem import solve_interleaved_cem
from .cohorts import load_manifest
from .config import STAGE, code_identity, validate_config
from .continuations import ContinuationProviders
from .reporting import summarize, format_summary
from .session import CheckpointSession


def check(config, device, cpu_threads):
    from .checks import run_checks
    resources = load_resources(config, device, cpu_threads)
    bank = evaluation_bank(resources, load_manifest(config), "validation", 1)
    coordinates = action_coordinates(resources, config)
    providers = ContinuationProviders(resources, config, coordinates)
    result = run_checks(resources, bank, config, coordinates, providers)
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(config, device, cpu_threads, split="validation", run_name=None, lock=None):
    from .checks import run_checks
    validate_config(config)
    manifest, implementation = load_manifest(config), code_identity()
    if split == "test":
        if not lock or lock.get("stage") != STAGE or lock.get("config") != config:
            raise ValueError("Test requires an interleave validation lock, without overrides")
        if lock["code"] != implementation or lock["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Locked source or query cohort changed")
        if file_hash(Path(lock["validation_report"])) != lock["report_sha256"]:
            raise ValueError("Locked validation report changed")
    elif lock:
        raise ValueError("Validation does not accept a test lock")
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["experiment_dir"])/"runs"/name
    if output.exists():
        raise FileExistsError("Run already exists; choose a new name")
    loading_started = time.perf_counter()
    resources = load_resources(config, device, cpu_threads)
    adapter, settings = resources.adapter, config["planning"]
    coordinates = action_coordinates(resources, config)
    providers = ContinuationProviders(resources, config, coordinates)
    provider_identity = providers.description()
    if lock and lock["retrieval_fingerprint"] != object_hash(provider_identity):
        raise ValueError("Locked context/endpoint retrieval settings changed")
    bank = evaluation_bank(resources, manifest, split, config["evaluation"]["query_limit"])
    synchronize(adapter)
    loading_seconds = time.perf_counter()-loading_started
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", config)
    write_json(output/"status.json", {"status": "running", "split": split})
    records, started = [], time.perf_counter()
    writer = RecordWriter(output, config)
    try:
        checks_started = time.perf_counter()
        checks = run_checks(resources, evaluation_bank(resources, manifest, "validation", 1),
                            config, coordinates, providers)
        synchronize(adapter)
        check_seconds = time.perf_counter()-checks_started
        write_json(output/"acceptance-checks.json", checks)
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        with writer as stream, progress(range(len(bank)), config=config, desc=f"{STAGE} {split}") as query_progress:
            for index in query_progress:
                task = planning_task(bank, index, settings["horizon"])
                goal = goal_latent(bank, task)
                for seed in resources.config["evaluation"]["seeds"]:
                    for arm in config["interleave"]["arms"]:
                        synchronize(adapter)
                        primary_started = time.perf_counter()
                        session = CheckpointSession(resources.engine, task.context, task.metadata, goal,
                                                    coordinates, providers, config, arm, seed)
                        initial_started = time.perf_counter()
                        initial, initial_audit = providers.initial_bundle(adapter, task.context, task.metadata, seed, goal)
                        synchronize(adapter)
                        session.timings["initial_proposals"] = time.perf_counter()-initial_started
                        session.counters["initial_action_retrievals"] = int(initial is not None)
                        cem_seed = stream_seed(seed, task.metadata["transition_id"], "cem")
                        plan, checkpoints = solve_interleaved_cem(session.score, settings, coordinates, cem_seed,
                            config["interleave"]["checkpoints"], session, initial)
                        adapted = session.adaptation_record()
                        adapter.check_fixed_state()
                        synchronize(adapter)
                        primary_seconds = time.perf_counter()-primary_started
                        diagnostic_started = time.perf_counter()
                        # Recorded future controls/targets become accessible only after search.
                        _, recorded_actions, target_rows = rollout_inputs(bank, index, settings["horizon"])
                        observed = observed_targets(bank, target_rows)
                        torch.testing.assert_close(observed[:, -1], goal, rtol=0, atol=0)
                        with fixed_predictor(adapter):
                            predicted = adapter.rollout(task.context, recorded_actions)
                            selected = adapter.rollout(task.context, plan.actions[None])
                        own_sse = float((selected[:, -1]-goal).square().sum())
                        if abs(own_sse-plan.cost) > 1e-5*max(1., abs(plan.cost)):
                            raise AssertionError("Final mean score and rollout disagree")
                        raw, recorded_raw = coordinates.to_raw(plan.actions), coordinates.to_raw(recorded_actions[0])
                        if ((raw < coordinates.raw_low-1e-6) | (raw > coordinates.raw_high+1e-6)).any():
                            raise AssertionError("Returned raw actions violate bounds")
                        adapter.reset(session.reset_seed)
                        with fixed_predictor(adapter):
                            from ..planning.cem import score_sequences
                            frozen_score = float(score_sequences(adapter, task.context, goal, plan.actions[None],
                                                                 settings["scoring_batch_size"])[0])
                        synchronize(adapter)
                        row = {"arm": arm, "seed": seed, "scope": config["scopes"][0],
                            "query": task.metadata, "current_raw_row": task.current_row, "goal_raw_row": task.goal_row,
                            "target_raw_rows": target_rows.tolist(), "cem_seed": cem_seed,
                            "initial_proposals": initial_audit,
                            "initial_injection_fingerprint": initial.fingerprint if initial is not None else None,
                            "cem_first_population_fingerprint": plan.first_population_fingerprint,
                            "cem_trace": plan.trace, "checkpoints": checkpoints, "adaptation": adapted,
                            "counters": dict(session.counters), "timing_seconds": dict(session.timings),
                            "selected_actions_normalized": plan.actions.cpu().tolist(),
                            "selected_raw_actions": raw.flatten(0, 1).cpu().tolist(),
                            "selected_predicted_endpoint": selected[0, -1].cpu().tolist(),
                            "selected_model_goal_sse": plan.cost,
                            "selected_plan_frozen_model_goal_sse": frozen_score,
                            "recorded_action_mse_by_block": latent_errors(predicted, observed),
                            "selected_vs_recorded_raw_action_mse": float((raw-recorded_raw).square().mean()),
                            "cem_candidate_evaluations": plan.candidate_evaluations,
                            "cem_transition_predictions": plan.candidate_evaluations*settings["horizon"],
                            "diagnostic_sequence_evaluations": 3,
                            "diagnostic_transition_predictions": 3*settings["horizon"],
                            "primary_seconds_with_checks": primary_seconds,
                            "diagnostic_seconds_with_checks": time.perf_counter()-diagnostic_started,
                            "checks": {"model_fixed_within_population": True, "query_local_reset": True,
                                "global_support_budget": True, "observed_memory_targets_only": True,
                                "recorded_future_excluded_from_search": True,
                                "supplied_goal_not_an_adaptation_target": True, "source_episode_safe_suffixes": True,
                                "full_candidates_rerolled_from_actual_context": True,
                                "returned_mean_scored_under_final_model": True, "versioned_scores": True}}
                        records.append(stream.append(row))
                stream.flush()
                done, frequency = index+1, config["evaluation"]["log_every"]
                if done == len(bank) or (frequency and done % frequency == 0):
                    write_json(output/"status.json", {"status": "running", "split": split,
                               "completed_queries": done, "queries": len(bank), "completed_records": len(records)})
                query_progress.set_postfix(records=len(records), refresh=False)
        summary = summarize(records, config)
        if summary["queries"] != len(bank):
            raise ValueError("Summary does not cover the selected cohort")
        from ..cache_loading import report_fields
        report = {**report_fields(resources), "completed": True, "stage": STAGE, "split": split, "config": config,
            "code": implementation, "manifest_fingerprint": manifest["fingerprint"],
            "p1_prepared_fingerprint": resources.prepared["fingerprint"], "model": adapter.identity,
            "scope_parameters": adapter.scope_description(), "action_coordinates": coordinates.description(),
            "retrieval": provider_identity, "checks": checks, "summary": summary,
            "cohort": manifest["splits"][split], "limits": evaluation_limits(config),
            "comparison_protocol": {"model_lifetime": "reset once per query/seed/arm; carry optimizer across checkpoints",
                "scoring": "one shared predictor at every depth and for every candidate within a CEM phase",
                "support": "balanced subsampling of elite/depth query pairs, then one global observed-support budget",
                "proposals": "updated-context suffix retrieval, complete candidates scored from the actual context",
                "pairing": "identical warm-up; independent CEM, reference, replay and perturbation streams",
                "selected_actions_executed": False},
            "records_file": str(writer.path.resolve()) if writer.path else None,
            "records_sha256": file_hash(writer.path) if writer.path else None,
            "record_count": len(records), "record_storage": writer.metadata(),
            "limited_run": config["evaluation"]["query_limit"] is not None,
            "evaluation_seconds": time.perf_counter()-started, "resource_loading_seconds": loading_seconds,
            "acceptance_check_seconds": check_seconds,
            "timing_scope": "Standalone arm latency includes reset, reference rerolls, retrieval, updates, injection and CEM checks; offline diagnostics and resource loading are separate",
            "gpu_memory": ({"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
                            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device),
                            "note": "Torch allocator only; some native cuVS allocations are excluded"}
                           if adapter.device.type == "cuda" else {})}
        if config.get("query_dataset"):
            report["query_source_fingerprint"] = manifest["query_source_fingerprint"]
        write_json(output/"report.json", report)
        (output/"summary.md").write_text(format_summary(report))
        write_json(output/"status.json", {"status": "completed", "records": len(records)})
        print(format_summary(report), flush=True)
        print(f"Report: {output/'report.json'}", flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "records": len(records),
                                       "error": f"{type(error).__name__}: {error}"})
        raise


def verified_report(path):
    report = read_json(path)
    if not report.get("completed") or report.get("stage") != STAGE:
        raise ValueError("Require a completed interleave report")
    validate_config(report["config"])
    if report["code"] != code_identity():
        raise ValueError("Interleave code changed; restore its recorded source before reanalysis/locking")
    from ..budget_receipts import verify_budget_capture
    verify_budget_capture(report)
    records = read_saved_records(report)
    if summarize(records, report["config"]) != report["summary"]:
        raise ValueError("Saved interleave records and summary differ")
    return report


def analyze(path, output=None):
    report = verified_report(path)
    destination = Path(output) if output else Path(path).with_name("analysis.md")
    if destination.resolve() in (Path(path).resolve(), Path(report["records_file"]).resolve()):
        raise ValueError("Preserve the source report and records")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_summary(report))
    print(format_summary(report), flush=True)


def lock_validation(path, output):
    if Path(output).exists():
        raise FileExistsError("Lock already exists; choose a new name")
    report = verified_report(path)
    if report["split"] != "validation" or report["limited_run"]:
        raise ValueError("Only complete unlimited validation may be locked")
    manifest = load_manifest(report["config"])
    if (manifest["fingerprint"] != report["manifest_fingerprint"]
            or report["summary"]["queries"] != manifest["splits"]["validation"]["queries"]):
        raise ValueError("Validation cohort changed or is incomplete")
    lock = {"stage": STAGE, "config": report["config"], "code": report["code"],
            "manifest_fingerprint": report["manifest_fingerprint"], "validation_report": str(Path(path).resolve()),
            "report_sha256": file_hash(Path(path)), "retrieval_fingerprint": object_hash(report["retrieval"]),
            "limits": evaluation_limits(report["config"])}
    lock["fingerprint"] = object_hash(lock)
    write_json(output, lock)
    print(f"Validation configuration locked: {output}", flush=True)
