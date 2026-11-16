"""Cached-episode CEM evaluation with a strict planning/evaluator boundary."""

from datetime import datetime, timezone
import json
from pathlib import Path

import time

import torch

from experiments.progress import progress

from ..runtime import file_hash, object_hash, read_json, write_json
from ..rollout.record_io import RecordWriter, read_saved_records
from ..rollout.cohorts import evaluation_bank, rollout_inputs, observed_targets
from ..rollout.inference import fixed_predictor, latent_errors
from ..rollout.resources import load_resources
from ..rollout.support_updates import synchronize
from .actions import ActionCoordinates, provisional_actions, stream_seed
from .cem import solve_cem, score_sequences
from .cohorts import load_manifest, planning_task, goal_latent
from .config import STAGE, evaluation_limits, code_identity, validate_config
from .reporting import summarize, format_summary
from .shared_ttt import adapt_shared


def action_coordinates(resources, config):
    result = ActionCoordinates(resources.prepared["normalization"], config["planning"],
                               resources.manifest["data_config"]["frameskip"], resources.adapter.device)
    if result.width != resources.adapter.action_dim:
        raise ValueError("Raw action bounds/frameskip do not match the model's ordered block width")
    return result


def check(config, device, cpu_threads):
    from .checks import run_checks
    manifest = load_manifest(config)
    resources = load_resources(config, device, cpu_threads)
    queries = evaluation_bank(resources, manifest, "validation", 1)
    result = run_checks(resources, queries, config, action_coordinates(resources, config))
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(config, device, cpu_threads, split="validation", run_name=None, lock=None):
    from .checks import run_checks
    validate_config(config)
    manifest, implementation = load_manifest(config), code_identity()
    if split == "test":
        if not lock or lock.get("stage") != STAGE or lock.get("config") != config:
            raise ValueError("P3-a test requires a completed P3-a validation lock")
        if lock["code"] != implementation or lock["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Locked P3-a code/cohort changed")
        if file_hash(Path(lock["validation_report"])) != lock["report_sha256"]:
            raise ValueError("Locked validation report changed")
    elif lock:
        raise ValueError("Use a lock only for test")
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["experiment_dir"])/"runs"/name
    if output.exists():
        raise FileExistsError("Run exists; use a new name to preserve completed/partial results")
    resource_started = time.perf_counter()
    resources = load_resources(config, device, cpu_threads)
    adapter, settings = resources.adapter, config["planning"]
    coordinates = action_coordinates(resources, config)
    queries = evaluation_bank(resources, manifest, split, config["evaluation"]["query_limit"])
    synchronize(adapter)
    resource_seconds = time.perf_counter()-resource_started
    seeds = resources.config["evaluation"]["seeds"]
    methods = ["frozen"] + [m for m in config["adaptation"]["methods"] if m != "frozen"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", config)
    write_json(output/"status.json", {"status": "running", "split": split})
    records, started = [], time.perf_counter()
    writer = RecordWriter(output, config)
    path = writer.path
    try:
        check_started = time.perf_counter()
        checks = run_checks(resources, evaluation_bank(resources, manifest, "validation", 1), config, coordinates)
        synchronize(adapter)
        check_seconds = time.perf_counter()-check_started
        write_json(output/"acceptance-checks.json", checks)
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        with writer as stream, progress(
                range(len(queries)), config=config, desc=f"P3 shared {split}") as query_progress:
            for index in query_progress:
                task = planning_task(queries, index, settings["horizon"])
                goal = goal_latent(queries, task)
                for seed in seeds:
                    provisional_seed = stream_seed(seed, task.metadata["transition_id"], "provisional")
                    cem_seed = stream_seed(seed, task.metadata["transition_id"], "cem")
                    synchronize(adapter)
                    provisional_started = time.perf_counter()
                    provisional = provisional_actions(coordinates, settings["provisional_actions"], settings["initial_std"], provisional_seed)
                    synchronize(adapter)
                    provisional_seconds = time.perf_counter()-provisional_started
                    frozen_plan, first_population = None, None
                    for method in methods:
                        synchronize(adapter)
                        row_started = time.perf_counter()
                        adapted = adapt_shared(resources.engine, task.context, task.metadata, provisional,
                                               method, seed, settings["support_scan_limit"])
                        synchronize(adapter)
                        cem_started = time.perf_counter()
                        with fixed_predictor(adapter):
                            plan = solve_cem(lambda candidates: score_sequences(
                                adapter, task.context, goal, candidates, settings["scoring_batch_size"]),
                                settings, coordinates, cem_seed)
                        synchronize(adapter)
                        cem_seconds = time.perf_counter()-cem_started
                        # Charge matched provisional generation to every standalone call.
                        primary_seconds = time.perf_counter()-row_started+provisional_seconds
                        if frozen_plan is None:
                            frozen_plan = plan.actions.detach().clone()
                            first_population = plan.first_population_fingerprint
                        if first_population != plan.first_population_fingerprint:
                            raise AssertionError("Methods did not receive the same initial CEM population")

                        # Evaluator-only data access begins after the complete search.
                        diagnostic_started = time.perf_counter()
                        _, recorded_actions, target_rows = rollout_inputs(queries, index, settings["horizon"])
                        with fixed_predictor(adapter):
                            recorded_prediction = adapter.rollout(task.context, recorded_actions)
                            selected_prediction = adapter.rollout(task.context, plan.actions[None])
                            frozen_plan_this_model = float(score_sequences(adapter, task.context, goal,
                                frozen_plan[None], settings["scoring_batch_size"])[0])
                        observed = observed_targets(queries, target_rows)
                        torch.testing.assert_close(observed[:, -1], goal, rtol=0, atol=0)
                        # Separate reference-model rescoring; never used for selection.
                        adapter.reset(adapted["reset_seed"])
                        with fixed_predictor(adapter):
                            frozen_rescore = float(score_sequences(adapter, task.context, goal,
                                plan.actions[None], settings["scoring_batch_size"])[0])
                        raw = coordinates.to_raw(plan.actions)
                        recorded_raw = coordinates.to_raw(recorded_actions[0])
                        action_error = (raw-recorded_raw).square()
                        recorded_oob = ((recorded_raw < coordinates.raw_low-1e-6)
                                        | (recorded_raw > coordinates.raw_high+1e-6)).float().mean()
                        if ((raw < coordinates.raw_low-1e-6) | (raw > coordinates.raw_high+1e-6)).any():
                            raise AssertionError("Selected plan exceeds the declared raw action bounds")
                        # Different scorer batch sizes may round differently; check tolerance.
                        own_sse = float((selected_prediction[:, -1]-goal).square().sum())
                        if abs(own_sse-plan.cost) > 1e-5*max(1.0, abs(plan.cost)):
                            raise AssertionError("Returned mean cost differs from its actual prediction")
                        synchronize(adapter)
                        row = {"method": method, "scope": config["scopes"][0], "seed": seed,
                               "query": task.metadata, "current_raw_row": task.current_row,
                               "goal_raw_row": task.goal_row, "target_raw_rows": target_rows.tolist(),
                               "provisional_seed": provisional_seed, "cem_seed": cem_seed,
                               "provisional_actions_normalized": provisional.cpu().tolist(),
                               "provisional_fingerprint": object_hash(provisional.cpu().tolist()),
                               "cem_first_population_fingerprint": plan.first_population_fingerprint,
                               "adaptation": adapted, "cem_trace": plan.trace,
                               "selected_actions_normalized": plan.actions.cpu().tolist(),
                               "selected_raw_actions": raw.flatten(0, 1).cpu().tolist(),
                               "selected_predicted_endpoint": selected_prediction[0, -1].cpu().tolist(),
                               "selected_model_goal_sse": plan.cost,
                               "selected_plan_frozen_model_goal_sse": frozen_rescore,
                               "frozen_plan_this_model_goal_sse": frozen_plan_this_model,
                               "recorded_action_mse_by_block": latent_errors(recorded_prediction, observed),
                               "selected_vs_recorded_raw_action_mse": float(action_error.mean()),
                               "first_block_vs_recorded_raw_action_mse": float(action_error[0].mean()),
                               "recorded_raw_action_out_of_bounds_fraction": float(recorded_oob),
                               "cem_candidate_evaluations": plan.candidate_evaluations,
                               "cem_transition_predictions": plan.candidate_evaluations*settings["horizon"],
                               "diagnostic_sequence_evaluations": 4,
                               "diagnostic_transition_predictions": 4*settings["horizon"],
                               "cem_seconds": cem_seconds, "primary_seconds_with_checks": primary_seconds,
                               "provisional_generation_seconds": provisional_seconds,
                               "diagnostic_seconds_with_checks": time.perf_counter()-diagnostic_started,
                               "total_seconds": time.perf_counter()-row_started,
                               "checks": {"parameters_and_buffers_fixed_through_cem": True,
                                          "paired_initial_population": True, "no_recorded_future_actions_in_planning": True,
                                          "goal_excluded_from_ttt": True, "returned_mean_scored": True,
                                          "selected_raw_actions_within_bounds": True}}
                        records.append(stream.append(row))
                stream.flush()
                done, frequency = index+1, config["evaluation"]["log_every"]
                if done == len(queries) or (frequency and done % frequency == 0):
                    write_json(output/"status.json", {"status": "running", "split": split,
                        "completed_queries": done, "queries": len(queries), "completed_records": len(records)})
                query_progress.set_postfix(records=len(records), refresh=False)
        summary = summarize(records, config)
        if summary["queries"] != len(queries):
            raise ValueError("P3 report does not cover the selected cohort")
        from ..cache_loading import report_fields
        report = {**report_fields(resources), "completed": True, "stage": STAGE, "split": split, "config": config,
                  "code": implementation, "manifest_fingerprint": manifest["fingerprint"],
                  "p1_prepared_fingerprint": resources.prepared["fingerprint"], "model": adapter.identity,
                  "scope_parameters": adapter.scope_description(), "action_coordinates": coordinates.description(),
                  "checks": checks, "summary": summary, "limits": evaluation_limits(config),
                  "protocol": "independent offline goal-conditioned CEM; one shared TTT phase then fixed model",
                  "cohort": manifest["splits"][split], "records_file": str(path.resolve()) if path else None,
                  "records_sha256": file_hash(path) if path else None, "record_count": len(records),
                  "record_storage": writer.metadata(),
                  "limited_run": config["evaluation"]["query_limit"] is not None,
                  "evaluation_seconds": time.perf_counter()-started,
                  "resource_loading_seconds": resource_seconds, "acceptance_check_seconds": check_seconds,
                  "timing_scope": "evaluation_seconds includes acceptance and offline diagnostics; resource loading is separate",
                  "retrieval": {"backend": resources.config["retrieval"]["backend"],
                                "memory_windows": len(resources.memory),
                                "key_payload_bytes": (resources.prepared.get("retrieval_assets") or {}).get("numeric_payload_bytes"),
                                "note": "Key payload excludes runtime index overhead and the observed frame/action store"},
                  "gpu_memory": ({"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
                                  "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device),
                                  "note": "run-wide Torch allocator; excludes some native/cuVS allocations"}
                                 if adapter.device.type == "cuda" else {})}
        write_json(output/"report.json", report)
        (output/"summary.md").write_text(format_summary(report))
        write_json(output/"status.json", {"status": "completed", "records": len(records)})
        print(format_summary(report), flush=True)
        print(f"Report: {output/'report.json'}", flush=True)
        print(f"Records: {writer.mode}, {writer.metadata()['bytes']/1024**2:.2f} MiB", flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "records": len(records),
                                         "error": f"{type(error).__name__}: {error}"})
        raise


def verified_report(path):
    report = read_json(path)
    if report.get("stage") != STAGE or not report.get("completed"):
        raise ValueError("Require a completed P3-a offline report")
    if report["code"] != code_identity():
        raise ValueError("P3 source changed since evaluation; restore the recorded source before analysis/locking")
    validate_config(report["config"])
    records = read_saved_records(report)
    summary = summarize(records, report["config"])
    if len(records) != report["record_count"] or summary != report["summary"]:
        raise ValueError("Saved P3 report/record summary mismatch")
    return report


def analyze(path, output=None):
    report = verified_report(path)
    destination = Path(output) if output else Path(path).with_name("analysis.md")
    if destination.resolve() in (Path(path).resolve(), Path(report["records_file"]).resolve()):
        raise ValueError("Analysis output must preserve the source report and records")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_summary(report))
    print(format_summary(report), flush=True)


def lock_validation(path, output):
    if Path(output).exists():
        raise FileExistsError("Lock exists; choose a new filename")
    report = verified_report(path)
    if report["split"] != "validation" or report["limited_run"]:
        raise ValueError("Only a complete, unlimited validation run can be locked")
    manifest = load_manifest(report["config"])
    if manifest["fingerprint"] != report["manifest_fingerprint"]:
        raise ValueError("Validation cohort changed")
    if report["summary"]["queries"] != manifest["splits"]["validation"]["queries"]:
        raise ValueError("Validation report does not cover the full cohort")
    lock = {"stage": STAGE, "config": report["config"], "code": report["code"],
            "manifest_fingerprint": report["manifest_fingerprint"],
            "validation_report": str(Path(path).resolve()), "report_sha256": file_hash(Path(path)),
            "limits": evaluation_limits(report["config"])}
    lock["fingerprint"] = object_hash(lock)
    write_json(output, lock)
    print(f"P3-a validation configuration locked: {output}", flush=True)
