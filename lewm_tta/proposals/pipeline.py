"""Offline proposal and factorial comparisons with independent shared-TTT factors."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from experiments.progress import progress

from ..runtime import file_hash, object_hash, read_json, write_json
from ..rollout.resources import load_resources
from ..rollout.cohorts import evaluation_bank, rollout_inputs, observed_targets
from ..rollout.inference import fixed_predictor, latent_errors
from ..rollout.support_updates import synchronize
from ..rollout.record_io import RecordWriter, read_saved_records
from ..planning.actions import provisional_actions, stream_seed
from ..planning.pipeline import action_coordinates
from ..planning.cem import score_sequences
from ..planning.cohorts import planning_task, goal_latent
from ..planning.shared_ttt import adapt_shared
from ..planning.config import evaluation_limits
from .cem import solve_guided_cem
from .cohorts import load_manifest
from .config import code_identity, stage_name, validate_config
from .providers import make_provider, proposal_bundle
from .reporting import summarize, format_summary
from .reuse import ReusedPlans, verify_dependencies


def check(config, device, cpu_threads):
    from .checks import run_checks
    resources = load_resources(config, device, cpu_threads)
    bank = evaluation_bank(resources, load_manifest(config), "validation", 1)
    coordinates = action_coordinates(resources, config)
    provider = make_provider(resources, config, coordinates)
    result = run_checks(resources, bank, config, coordinates, provider)
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(config, device, cpu_threads, split="validation", run_name=None, lock=None):
    from .checks import run_checks
    validate_config(config)
    manifest, implementation = load_manifest(config), code_identity()
    if split == "test":
        if not lock or lock.get("stage") != stage_name(config) or lock.get("config") != config:
            raise ValueError("Test requires a completed validation lock for this proposal comparison")
        if lock["code"] != implementation or lock["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Locked proposal code/cohort changed")
        if file_hash(Path(lock["validation_report"])) != lock["report_sha256"]:
            raise ValueError("Locked validation report changed")
    elif lock:
        raise ValueError("Validation does not use a test lock")
    reuse = ReusedPlans(config, split, manifest)
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["experiment_dir"])/"runs"/name
    if output.exists():
        raise FileExistsError("Run exists; choose a new name")
    loading_started = time.perf_counter()
    resources = load_resources(config, device, cpu_threads)
    adapter, settings = resources.adapter, config["planning"]
    coordinates = action_coordinates(resources, config)
    provider = make_provider(resources, config, coordinates)
    goal_description = provider.goal_search.description() if provider.goal_search else {"enabled": False}
    if split == "test" and goal_description["enabled"]:
        if lock.get("goal_retrieval_fingerprint") != goal_description["fingerprint"]:
            raise ValueError("Locked goal-retrieval calibration or metric changed")
    bank = evaluation_bank(resources, manifest, split, config["evaluation"]["query_limit"])
    reuse.check_resources(resources, coordinates)
    synchronize(adapter)
    loading_seconds = time.perf_counter()-loading_started
    seeds = resources.config["evaluation"]["seeds"]
    methods = ["frozen"]+[m for m in config["adaptation"]["methods"] if m != "frozen"]
    sources = ["gaussian"]+[s for s in config["proposals"]["sources"] if s != "gaussian"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", config)
    write_json(output/"status.json", {"status": "running", "split": split})
    records, started = [], time.perf_counter()
    reuse_verification_seconds = 0.
    writer = RecordWriter(output, config)
    try:
        checks_started = time.perf_counter()
        checks = run_checks(resources, evaluation_bank(resources, manifest, "validation", 1), config, coordinates, provider)
        synchronize(adapter)
        check_seconds = time.perf_counter()-checks_started
        write_json(output/"acceptance-checks.json", checks)
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        with writer as stream, progress(
                range(len(bank)), config=config, desc=f"{stage_name(config)} {split}") as query_progress:
            for index in query_progress:
                task = planning_task(bank, index, settings["horizon"])
                goal = goal_latent(bank, task)
                for seed in seeds:
                    # Select action proposals before any TTT; reuse verbatim across models.
                    bundles = {source: proposal_bundle(provider, adapter, task.context, task.metadata, source, seed, goal=goal)
                               for source in sources}
                    provisional_seed = stream_seed(seed, task.metadata["transition_id"], "provisional")
                    cem_seed = stream_seed(seed, task.metadata["transition_id"], "cem")
                    synchronize(adapter)
                    provisional_started = time.perf_counter()
                    provisional = provisional_actions(coordinates, settings["provisional_actions"], settings["initial_std"], provisional_seed)
                    synchronize(adapter)
                    provisional_seconds = time.perf_counter()-provisional_started
                    frozen_plans, base_signature = {}, None
                    for method in methods:
                        adapted = adapt_shared(resources.engine, task.context, task.metadata, provisional,
                                               method, seed, settings["support_scan_limit"])
                        parity_started = time.perf_counter()
                        reuse.verify_adaptation(task, seed, method, provisional, adapted, adapter, coordinates)
                        synchronize(adapter)
                        reuse_verification_seconds += time.perf_counter()-parity_started
                        reference = reuse.reference(task, seed, method)
                        pending = []
                        for source in sources:
                            injection, proposal = bundles[source]
                            plan_reference = reuse.reference(task, seed, method, source)
                            reuse.verify_proposal(task, seed, method, source, proposal)
                            guided = reuse.guided_plan(task, seed, method, adapter, source)
                            imported = guided is not None
                            if imported:
                                if plan_reference["cem_seed"] != cem_seed:
                                    raise ValueError("Reused CEM RNG seed differs")
                                cem_seconds = plan_reference["cem_seconds"]
                            else:
                                synchronize(adapter)
                                cem_started = time.perf_counter()
                                with fixed_predictor(adapter):
                                    guided = solve_guided_cem(lambda actions: score_sequences(
                                        adapter, task.context, goal, actions, settings["scoring_batch_size"]),
                                        settings, coordinates, cem_seed, injection)
                                synchronize(adapter)
                                cem_seconds = time.perf_counter()-cem_started
                            if base_signature is None:
                                base_signature = guided.base_first_population_fingerprint
                            if base_signature != guided.base_first_population_fingerprint:
                                raise AssertionError("TTT/proposal arms changed the base CEM random population")
                            if method == "frozen":
                                frozen_plans[source] = guided.plan.actions.detach().clone()
                            pending.append((source, proposal, guided, cem_seconds, imported, plan_reference))

                        # All searches under this model finish before recorded futures are read.
                        _, recorded_actions, target_rows = rollout_inputs(bank, index, settings["horizon"])
                        if reference is not None and reference["target_raw_rows"] != target_rows.tolist():
                            raise ValueError("Reused evaluator target alignment differs")
                        observed = observed_targets(bank, target_rows)
                        torch.testing.assert_close(observed[:, -1], goal, rtol=0, atol=0)
                        current_rows = []
                        for source, proposal, guided, cem_seconds, imported, plan_reference in pending:
                            synchronize(adapter)
                            diagnostic_started = time.perf_counter()
                            plan = guided.plan
                            with fixed_predictor(adapter):
                                predicted = adapter.rollout(task.context, recorded_actions)
                                selected_prediction = adapter.rollout(task.context, plan.actions[None])
                                frozen_plan_this_model = float(score_sequences(adapter, task.context, goal,
                                    frozen_plans[source][None], settings["scoring_batch_size"])[0])
                            raw, recorded_raw = coordinates.to_raw(plan.actions), coordinates.to_raw(recorded_actions[0])
                            if ((raw < coordinates.raw_low-1e-6) | (raw > coordinates.raw_high+1e-6)).any():
                                raise AssertionError("Selected plan violates raw action bounds")
                            error = (raw-recorded_raw).square()
                            own_sse = float((selected_prediction[:, -1]-goal).square().sum())
                            if abs(own_sse-plan.cost) > 1e-5*max(1.0, abs(plan.cost)):
                                raise AssertionError("Returned mean's cost does not match its prediction")
                            synchronize(adapter)
                            primary = provisional_seconds+adapted["timing_seconds"]["total_with_checks"]+proposal["total_seconds"]+cem_seconds
                            if imported:
                                primary = plan_reference["primary_seconds_with_checks"]
                            row = {"method": method, "scope": config["scopes"][0], "seed": seed,
                                "proposal_source": source, "proposal": proposal, "query": task.metadata,
                                "current_raw_row": task.current_row, "goal_raw_row": task.goal_row,
                                "target_raw_rows": target_rows.tolist(), "cem_seed": cem_seed,
                                "provisional_seed": provisional_seed, "provisional_actions_normalized": provisional.cpu().tolist(),
                                "provisional_fingerprint": object_hash(provisional.cpu().tolist()), "adaptation": adapted,
                                "base_first_population_fingerprint": guided.base_first_population_fingerprint,
                                "cem_first_population_fingerprint": plan.first_population_fingerprint,
                                "cem_trace": plan.trace, "direct_elite_source_fraction": guided.direct_elite_source_fraction,
                                "selected_actions_normalized": plan.actions.cpu().tolist(),
                                "selected_raw_actions": raw.flatten(0, 1).cpu().tolist(),
                                "selected_predicted_endpoint": selected_prediction[0, -1].cpu().tolist(),
                                "selected_model_goal_sse": plan.cost, "frozen_plan_this_model_goal_sse": frozen_plan_this_model,
                                "recorded_action_mse_by_block": reuse.forecast(latent_errors(predicted, observed), reference),
                                "selected_vs_recorded_raw_action_mse": float(error.mean()),
                                "first_block_vs_recorded_raw_action_mse": float(error[0].mean()),
                                "recorded_raw_action_out_of_bounds_fraction": float(((recorded_raw < coordinates.raw_low-1e-6)
                                    | (recorded_raw > coordinates.raw_high+1e-6)).float().mean()),
                                "cem_candidate_evaluations": plan.candidate_evaluations,
                                "cem_transition_predictions": plan.candidate_evaluations*settings["horizon"],
                                "diagnostic_sequence_evaluations": 4, "diagnostic_transition_predictions": 4*settings["horizon"],
                                "result_origin": "reused" if imported else "computed",
                                "reuse": reuse.origin(task, seed, method, source) if imported else None,
                                "forecast_source": reuse.origin(task, seed, method),
                                "new_cem_candidate_evaluations": 0 if imported else plan.candidate_evaluations,
                                "primary_timing_origin": "source run" if imported else "current factorial run",
                                "cem_seconds": cem_seconds, "primary_seconds_with_checks": primary,
                                "provisional_generation_seconds": provisional_seconds,
                                "diagnostic_seconds_with_checks": time.perf_counter()-diagnostic_started,
                                "checks": {"fixed_model_through_cem": True, "common_base_gaussian_population": True,
                                    "goal_excluded_from_ttt": True,
                                    "recorded_future_excluded_from_ttt_and_action_retrieval": True,
                                    "supplied_goal_used_only_by_explicit_goal_source_and_cem": True,
                                    "memory_only_episode_safe_fragments": True, "returned_mean_scored": True,
                                    "proposal_independent_of_ttt": True}}
                            current_rows.append((row, plan.actions, plan_reference))
                        # Reset once; cross-scoring each plan is evaluation-only.
                        synchronize(adapter)
                        reset_started = time.perf_counter()
                        adapter.reset(adapted["reset_seed"])
                        synchronize(adapter)
                        reset_seconds = time.perf_counter()-reset_started
                        for row, actions, plan_reference in current_rows:
                            diagnostic_started = time.perf_counter()
                            with fixed_predictor(adapter):
                                row["selected_plan_frozen_model_goal_sse"] = float(score_sequences(
                                    adapter, task.context, goal, actions[None], settings["scoring_batch_size"])[0])
                            synchronize(adapter)
                            row["diagnostic_seconds_with_checks"] += time.perf_counter()-diagnostic_started+reset_seconds
                            row["total_seconds"] = row["primary_seconds_with_checks"]+row["diagnostic_seconds_with_checks"]
                            if row["result_origin"] == "reused":
                                reuse.retain_scores(row, plan_reference)
                                row["checks"]["reused_support_parameters_rng_and_metric_parity"] = True
                            records.append(stream.append(row))
                stream.flush()
                done, frequency = index+1, config["evaluation"]["log_every"]
                if done == len(bank) or (frequency and done % frequency == 0):
                    write_json(output/"status.json", {"status": "running", "completed_queries": done,
                                                       "queries": len(bank), "completed_records": len(records), "split": split})
                query_progress.set_postfix(records=len(records), reused=reuse.reused_count, refresh=False)
        summary = summarize(records, config)
        if summary["queries"] != len(bank):
            raise ValueError("Summary does not cover the selected query cohort")
        from ..cache_loading import report_fields
        report = {**report_fields(resources), "completed": True, "stage": stage_name(config), "split": split, "config": config,
            "comparison_protocol": {
                "comparison": config["comparison"], "model_treatments": methods, "proposal_treatments": sources,
                "cells_per_query_seed": len(methods)*len(sources), "total_planning_calls": len(records),
                "model_reset": "Pretrained weights and fresh optimizer for each query/seed/TTT treatment",
                "shared_ttt": "One adaptation phase using independent provisional actions, then fixed weights across every CEM call",
                "action_proposals": "Selected before TTT and reused verbatim across model treatments",
                "retrieval_goal": "Only goal-context-fragments compares the supplied goal with achieved memory H-step endpoints; no goal enters TTT",
                "pairing": "Same cached context, goal, provisional actions and base Gaussian noise; independent factor random streams",
                "cross_model_effect_metric": "Selected plans rescored by the common Frozen reference; model proxy only",
                "observed_outcomes": "Only recorded-action prediction has observed future targets",
                "selected_actions_executed": False},
            "code": implementation, "manifest_fingerprint": manifest["fingerprint"],
            "p1_prepared_fingerprint": resources.prepared["fingerprint"], "model": adapter.identity,
            "scope_parameters": adapter.scope_description(), "action_coordinates": coordinates.description(),
            "context_retrieval": provider.search.description(),
            "goal_retrieval": goal_description,
            "memory_fragment_eligibility": {"total_memory_windows": len(resources.memory),
                                            "horizon_safe_windows": int(provider.store.eligible.sum())},
            "checks": checks, "summary": summary, "limits": evaluation_limits(config), "cohort": manifest["splits"][split],
            "reuse": reuse.description(records), "reuse_verification_seconds": reuse_verification_seconds,
            "records_file": str(writer.path.resolve()) if writer.path else None,
            "records_sha256": file_hash(writer.path) if writer.path else None, "record_count": len(records),
            "record_storage": writer.metadata(), "limited_run": config["evaluation"]["query_limit"] is not None,
            "evaluation_seconds": time.perf_counter()-started, "resource_loading_seconds": loading_seconds,
            "acceptance_check_seconds": check_seconds,
            "timing_scope": "Per-cell latency charges standalone TTT/proposal/reset costs even when identical setup is reused across cells; evaluation_seconds is actual run time excluding resources",
            "gpu_memory": ({"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
                            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device),
                            "note": "Torch allocator only; excludes some cuVS/native allocations"} if adapter.device.type == "cuda" else {})}
        if config.get("query_dataset"):
            report["query_source_fingerprint"] = manifest["query_source_fingerprint"]
        write_json(output/"report.json", report)
        (output/"summary.md").write_text(format_summary(report))
        write_json(output/"status.json", {"status": "completed", "records": len(records)})
        print(format_summary(report), flush=True)
        print(f"Report: {output/'report.json'}", flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "records": len(records), "error": f"{type(error).__name__}: {error}"})
        raise


def verified_report(path):
    report = read_json(path)
    if not report.get("completed") or report.get("stage") != stage_name(report["config"]):
        raise ValueError("Require a completed proposal-comparison report")
    validate_config(report["config"])
    if report["code"] != code_identity():
        raise ValueError("Proposal source changed; restore the recorded implementation before reanalysis/locking")
    records = read_saved_records(report)
    verify_dependencies(report)
    reused_count = sum(row.get("result_origin") == "reused" for row in records)
    if report.get("reuse", {}).get("reused_cem_calls", 0) != reused_count:
        raise ValueError("Reused record count differs from report accounting")
    if summarize(records, report["config"]) != report["summary"]:
        raise ValueError("Saved summary/records mismatch")
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
        raise FileExistsError("Lock exists; choose a new filename")
    report = verified_report(path)
    if report["split"] != "validation" or report["limited_run"]:
        raise ValueError("Only unlimited completed validation runs may be locked")
    manifest = load_manifest(report["config"])
    if (manifest["fingerprint"] != report["manifest_fingerprint"]
            or report["summary"]["queries"] != manifest["splits"]["validation"]["queries"]):
        raise ValueError("Validation cohort changed or is incomplete")
    lock = {"stage": report["stage"], "config": report["config"], "code": report["code"],
            "manifest_fingerprint": report["manifest_fingerprint"], "validation_report": str(Path(path).resolve()),
            "report_sha256": file_hash(Path(path)), "limits": evaluation_limits(report["config"])}
    if report.get("goal_retrieval", {}).get("enabled"):
        lock["goal_retrieval_fingerprint"] = report["goal_retrieval"]["fingerprint"]
    lock["fingerprint"] = object_hash(lock)
    write_json(output, lock)
    print(f"Validation configuration locked: {output}", flush=True)
