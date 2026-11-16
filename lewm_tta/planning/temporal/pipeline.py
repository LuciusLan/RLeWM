"""Optional temporal scorers around the existing bounded Gaussian CEM solver."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from experiments.progress import progress

from ...runtime import file_hash, object_hash, read_json, write_json
from ...rollout.cohorts import evaluation_bank, observed_targets, rollout_inputs
from ...rollout.inference import fixed_predictor, latent_errors
from ...rollout.record_io import RecordWriter, read_saved_records
from ...rollout.resources import load_resources
from ...rollout.support_updates import synchronize
from ..actions import stream_seed
from ..cem import score_sequences, solve_cem
from ..cohorts import goal_latent, planning_task
from ..config import evaluation_limits
from ..pipeline import action_coordinates
from .cohorts import load_manifest
from .config import STAGE, code_identity, nominal_budget, seed_schedule, seeds_for_mode, validate_config
from .reporting import format_summary, summarize
from .rollouts import reference_actions
from .scoring import TemporalScorer
from .state import fresh_counters


def check(config, device, cpu_threads):
    from .checks import run_checks
    resources = load_resources(config, device, cpu_threads)
    bank = evaluation_bank(resources, load_manifest(config), "validation", 1)
    result = run_checks(resources, bank, config, action_coordinates(resources, config))
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(config, device, cpu_threads, split="validation", run_name=None, lock=None):
    from .checks import run_checks
    validate_config(config)
    manifest, implementation = load_manifest(config), code_identity()
    if split == "test":
        if not lock or lock.get("stage") != STAGE or lock.get("config") != config:
            raise ValueError("Test requires a completed temporal validation lock")
        if lock["code"] != implementation or lock["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Locked temporal code/cohort changed")
        if file_hash(Path(lock["validation_report"])) != lock["report_sha256"]:
            raise ValueError("Locked temporal validation report changed")
    elif lock:
        raise ValueError("Validation does not use a test lock")
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["experiment_dir"])/"runs"/name
    if output.exists():
        raise FileExistsError("Run exists; choose a new name")
    resources = load_resources(config, device, cpu_threads)
    adapter, planning = resources.adapter, config["planning"]
    coordinates = action_coordinates(resources, config)
    bank = evaluation_bank(resources, manifest, split, config["evaluation"]["query_limit"])
    modes = ["frozen"]+[mode for mode in config["temporal"]["modes"] if mode != "frozen"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", config)
    write_json(output/"status.json", {"status": "running", "split": split})
    print("Nominal primary budgets per query/seed:", json.dumps({
        mode: {"seeds": seeds_for_mode(config, mode), **nominal_budget(config, mode)} for mode in modes}), flush=True)
    records, started = [], time.perf_counter()
    writer = RecordWriter(output, config)
    try:
        checks = run_checks(resources, evaluation_bank(resources, manifest, "validation", 1), config, coordinates)
        write_json(output/"acceptance-checks.json", checks)
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        with writer as stream, progress(
                range(len(bank)), config=config, desc=f"P3 temporal {split}") as query_progress:
            for index in query_progress:
                task = planning_task(bank, index, planning["horizon"])
                goal = goal_latent(bank, task)
                for seed, active_modes in seed_schedule(config):
                    reference_started = time.perf_counter()
                    references = reference_actions(coordinates, config, task.metadata, seed)
                    synchronize(adapter)
                    reference_seconds = time.perf_counter()-reference_started
                    reference_fp = object_hash(references.cpu().tolist())
                    cem_seed = stream_seed(seed, task.metadata["transition_id"], "cem")
                    pending, signature = [], None
                    for mode in active_modes:
                        query_progress.set_postfix(seed=seed, mode=mode)
                        scorer = TemporalScorer(resources.engine, task.context, task.metadata, references, mode, seed, config)
                        synchronize(adapter)
                        search_started = time.perf_counter()
                        plan = solve_cem(lambda actions: scorer.costs(actions, goal), planning, coordinates, cem_seed)
                        synchronize(adapter)
                        cem_seconds = time.perf_counter()-search_started
                        if signature is None:
                            signature = plan.first_population_fingerprint
                        if signature != plan.first_population_fingerprint:
                            raise AssertionError("Temporal treatment changed the base CEM population")
                        accounting = scorer.accounting()
                        budget = nominal_budget(config, mode)
                        if accounting["requested_updates"] != budget["requested_gradient_updates"]:
                            raise AssertionError("Actual requested temporal budget differs from config")
                        pending.append((mode, scorer, plan, accounting, cem_seconds))
                    # Recorded future actions and targets are read only after every search.
                    _, recorded_actions, target_rows = rollout_inputs(bank, index, planning["horizon"])
                    observed = observed_targets(bank, target_rows)
                    frozen_plan = pending[0][2]
                    for mode, scorer, plan, primary, cem_seconds in pending:
                        diagnostic_started = time.perf_counter()
                        before = scorer.accounting()
                        prediction = scorer.rollout(recorded_actions)
                        selected = scorer.rollout(plan.actions[None])
                        selected_audit = scorer.last_audit
                        frozen_plan_this_model = float(scorer.costs(frozen_plan.actions[None], goal)[0])
                        # All diagnostic adaptation is separately charged, never re-used
                        # to select a plan or alter the already-completed CEM search.
                        after = scorer.accounting()
                        diagnostic = {key: after[key]-before[key] for key in fresh_counters()}
                        own_cost = float((selected[:, -1]-goal).square().sum())
                        if abs(own_cost-plan.cost) > 1e-5*max(1., abs(plan.cost)):
                            raise AssertionError("Selected mean's adaptive score is not reproducible")
                        adapter.reset(stream_seed(seed, task.metadata["transition_id"], "model-reset"))
                        with fixed_predictor(adapter):
                            common_cost = float(score_sequences(adapter, task.context, goal, plan.actions[None], planning["scoring_batch_size"])[0])
                        synchronize(adapter)
                        row = {"mode": mode, "method": "frozen" if mode == "frozen" else config["temporal"]["selector"],
                            "scope": config["scopes"][0], "query": task.metadata, "seed": seed,
                            "goal_raw_row": task.goal_row, "target_raw_rows": target_rows.tolist(),
                            "reference_fingerprint": reference_fp, "cem_seed": cem_seed,
                            "cem_first_population_fingerprint": plan.first_population_fingerprint, "cem_trace": plan.trace,
                            "selected_actions_normalized": plan.actions.cpu().tolist(),
                            "selected_raw_actions": coordinates.to_raw(plan.actions).cpu().tolist(),
                            "selected_model_goal_sse": plan.cost, "selected_plan_frozen_model_goal_sse": common_cost,
                            "frozen_plan_this_model_goal_sse": frozen_plan_this_model,
                            "recorded_action_mse_by_block": latent_errors(prediction, observed),
                            "adaptation": {**primary, "rounds": []}, "setup_adaptation": scorer.setup,
                            "selected_plan_adaptation_trace": selected_audit,
                            "diagnostic_adaptation": diagnostic,
                            "diagnostic_common_frozen_sequence_evaluations": 1,
                            "diagnostic_common_frozen_transition_predictions": planning["horizon"],
                            "cem_seconds": cem_seconds, "setup_seconds": scorer.setup_seconds,
                            "primary_seconds_with_checks": reference_seconds+scorer.setup_seconds+cem_seconds,
                            "diagnostic_seconds_with_checks": time.perf_counter()-diagnostic_started,
                            "checks": {"same_base_cem_population": True, "goal_excluded_from_ttt": True,
                                "recorded_futures_read_after_search": True, "observed_memory_targets_only": True,
                                "candidate_state_isolated_or_depth_models_fixed": True,
                                "returned_mean_adaptive_score_reproduced": True,
                                "no_actions_executed": True}}
                        row = stream.append(row)
                        records.append(row)
                    stream.flush()
                query_progress.set_postfix(records=len(records), refresh=False)
        summary = summarize(records, config)
        from ...cache_loading import report_fields
        report = {**report_fields(resources), "completed": True, "stage": STAGE, "split": split, "config": config, "code": implementation,
            "manifest_fingerprint": manifest["fingerprint"], "p1_prepared_fingerprint": resources.prepared["fingerprint"],
            "model": adapter.identity, "scope_parameters": adapter.scope_description(), "action_coordinates": coordinates.description(),
            "checks": checks, "summary": summary, "cohort": manifest["splits"][split],
            "records_file": str(writer.path.resolve()) if writer.path else None,
            "records_sha256": file_hash(writer.path) if writer.path else None,
            "record_count": len(records), "record_storage": writer.metadata(),
            "limited_run": config["evaluation"]["query_limit"] is not None,
            "evaluation_seconds": time.perf_counter()-started,
            "protocol": {"shared_depth": "Build reference-conditioned model snapshots once, shared across all CEM candidates/iterations at each depth",
                "seeds_by_mode": {mode: seeds_for_mode(config, mode) for mode in modes},
                "seed_pairing": "Each mode uses its configured seeds; cross-mode comparisons use only common query/seed pairs",
                "candidate_temporal": "Fresh original parameters, optimizer and query RNG for every scored sequence, including the returned mean",
                "support": "Observed historical transitions only; predicted contexts are retrieval queries, never labels",
                "reference_actions": "Goal-independent provisional Gaussian action sequences; first block matches existing P3",
                "proposals": "Gaussian CEM only; action-fragment injection is not combined with this ablation",
                "counterfactual_outcomes": "No simulator; only recorded-action predictions have observed targets",
                "diagnostics": "Three adaptive sequence evaluations and one common-Frozen sequence evaluation, charged separately",
                "budget": "Counts include the final mean; candidate training losses and support reevaluation add work beyond CEM transition predictions"},
            "limits": evaluation_limits(config)+["Expensive exploratory adaptation treatment, not a guaranteed performance upper bound."],
            "gpu_memory": ({"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
                            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device)}
                           if adapter.device.type == "cuda" else {})}
        write_json(output/"report.json", report)
        (output/"summary.txt").write_text(format_summary(report))
        write_json(output/"status.json", {"status": "completed", "records": len(records)})
        print(format_summary(report), flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "records": len(records), "error": f"{type(error).__name__}: {error}"})
        raise


def verified_report(path):
    report = read_json(path)
    if not report.get("completed") or report.get("stage") != STAGE:
        raise ValueError("Require a completed temporal planning report")
    validate_config(report["config"])
    if report["code"] != code_identity():
        raise ValueError("Temporal source differs from the report; use verified source replay")
    if summarize(read_saved_records(report), report["config"]) != report["summary"]:
        raise ValueError("Temporal records and summary differ")
    return report


def analyze(path, output=None):
    report = verified_report(path)
    destination = Path(output) if output else Path(path).with_name("analysis.txt")
    if destination.resolve() in (Path(path).resolve(), Path(report["records_file"]).resolve()):
        raise ValueError("Preserve the source report and records")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_summary(report))
    print(format_summary(report), flush=True)


def lock_validation(path, output):
    if Path(output).exists():
        raise FileExistsError("Lock exists; choose a new filename")
    report = verified_report(path)
    manifest = load_manifest(report["config"])
    if report["split"] != "validation" or report["limited_run"] or report["summary"]["queries"] != manifest["splits"]["validation"]["queries"]:
        raise ValueError("Only unlimited complete validation cohorts may be locked")
    lock = {"stage": STAGE, "config": report["config"], "code": report["code"],
            "manifest_fingerprint": report["manifest_fingerprint"], "validation_report": str(Path(path).resolve()),
            "report_sha256": file_hash(Path(path))}
    lock["fingerprint"] = object_hash(lock)
    write_json(output, lock)
    print(f"Validation configuration locked: {output}", flush=True)
