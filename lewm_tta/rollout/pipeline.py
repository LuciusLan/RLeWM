"""One evaluator for recorded-rollout scope and iteration comparisons."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from experiments.progress import progress

from .inference import compact_adaptation, compact_schedule_adaptation, fixed_predictor, latent_errors
from .cohorts import evaluation_bank, load_manifest, observed_targets, rollout_inputs
from ..runtime import file_hash, object_hash, read_json, write_json
from .checks import run_checks
from .config import (
    P2_STAGE, SCOPE_STAGE, SCHEDULE_STAGE, entrypoint, is_scope_comparison, scope_names,
    is_schedule_comparison, schedule_names,
    stage_name, validate_config,
)
from .resources import load_resources
from .provenance import code_identity
from .reporting import format_summary, summarize
from .record_io import RecordWriter, read_saved_records
from .temporal import run_scheduled_rollout


def _sync(adapter):
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)


def check(config: dict, device: str, cpu_threads: int) -> dict:
    manifest = load_manifest(config)
    resources = load_resources(config, device, cpu_threads)
    queries = evaluation_bank(resources, manifest, "validation", 1)
    result = run_checks(resources, queries, config)
    print(json.dumps(result, indent=2), flush=True)
    return result


def scope_descriptions(adapter, scopes: list[str]) -> dict:
    result = {}
    for scope in scopes:
        adapter.set_scope(scope)
        result[scope] = adapter.scope_description()
    return result


def _gpu_memory(adapter) -> dict:
    if adapter.device.type != "cuda":
        return {}
    return {"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device),
            "scope": "run-wide Torch allocator; does not include all cuVS/native allocations"}


def evaluate(config: dict, device: str, cpu_threads: int, split: str = "validation",
             run_name: str | None = None, lock: dict | None = None) -> dict:
    validate_config(config)
    manifest, implementation = load_manifest(config), code_identity(entrypoint(config))
    if split == "test":
        if not lock or lock.get("stage") != stage_name(config):
            raise ValueError("Rollout test requires its own completed validation lock")
        if lock["config"] != config or lock["code"] != implementation or lock["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Rollout locked config/code/cohort changed")
        if file_hash(Path(lock["validation_report"])) != lock["report_sha256"]:
            raise ValueError("The locked validation report changed")
    elif lock is not None:
        raise ValueError("Validation does not use a test lock")
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["experiment_dir"])/"runs"/name
    if output.exists():
        raise FileExistsError("Run already exists; choose a new name to preserve it")
    resources = load_resources(config, device, cpu_threads)
    queries = evaluation_bank(resources, manifest, split, config["evaluation"]["query_limit"])
    check_queries = evaluation_bank(resources, manifest, "validation", 1)
    methods, seeds = resources.config["adaptation"]["methods"], resources.config["evaluation"]["seeds"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", config)
    write_json(output/"status.json", {"status": "running", "split": split})
    writer = RecordWriter(output, config)
    records_path, records, started = writer.path, [], time.perf_counter()
    try:
        descriptions = scope_descriptions(resources.adapter, scope_names(config))
        checks = run_checks(resources, check_queries, config)
        write_json(output/"acceptance-checks.json", checks)
        if resources.adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resources.adapter.device)
        with writer as stream, progress(
                range(len(queries)), config=config, desc=f"{stage_name(config)} {split}") as query_progress:
            for index in query_progress:
                query, actions, targets = rollout_inputs(queries, index, max(config["horizons"]))
                initial_reference, first_support = None, {}
                for seed in seeds:
                    for scope in scope_names(config):
                        resources.adapter.set_scope(scope)
                        for schedule_name in schedule_names(config):
                            first_fr = None
                            for method in methods:
                                _sync(resources.adapter)
                                row_started = time.perf_counter()
                                if schedule_name is None:
                                    adapted = resources.engine.run(query, method, seed)
                                else:
                                    predicted, adapted = run_scheduled_rollout(resources.engine, query, actions,
                                        method, seed, config["schedules"][schedule_name])
                                initial = torch.tensor(adapted["prediction_initial"])
                                if initial_reference is None:
                                    initial_reference = initial
                                torch.testing.assert_close(initial, initial_reference, rtol=0, atol=0)
                                if adapted["rounds"]:
                                    signature = object_hash([{key: item[key] for key in ("transition_id", "weight", "channels")}
                                        for item in adapted["rounds"][0]["support"]])
                                    pair = (seed, method)
                                    if pair in first_support and first_support[pair] != signature:
                                        raise AssertionError("First-update support differs across scopes")
                                    first_support[pair] = signature
                                if method in ("fr-fixed", "fr-refresh") and adapted["rounds"]:
                                    first = torch.tensor(adapted["rounds"][0]["prediction_after"])
                                    if first_fr is None:
                                        first_fr = first
                                    else:
                                        torch.testing.assert_close(first, first_fr, rtol=1e-6, atol=1e-7)
                                if schedule_name is None:
                                    _sync(resources.adapter)
                                    rollout_started = time.perf_counter()
                                    with fixed_predictor(resources.adapter):
                                        predicted = resources.adapter.rollout(query.context, actions)
                                    _sync(resources.adapter)
                                    row_timing = {"rollout_seconds_with_checks": time.perf_counter()-rollout_started}
                                else:
                                    row_timing = {"scheduled_rollout_seconds_with_checks": adapted["timing_seconds"]["total_with_checks"]}
                                torch.testing.assert_close(predicted[:, 0].cpu(),
                                    torch.tensor(adapted["prediction_final"]).reshape(1, -1), rtol=1e-6, atol=1e-7)
                                target = observed_targets(queries, targets)
                                compact = (compact_adaptation(adapted, target[:, 0]) if schedule_name is None
                                           else compact_schedule_adaptation(adapted, target))
                                compact["module_updates"] = adapted["module_updates"]
                                row = {"scope": scope, "method": method, "seed": seed, "query": query.metadata,
                                    "target_raw_rows": targets.tolist(), "mse_by_block": latent_errors(predicted, target),
                                    "adaptation": compact, **row_timing,
                                    "total_seconds": time.perf_counter()-row_started,
                                    "checks": {"initial_prediction_matches_all_scopes": True,
                                        "first_update_support_matches_scope_reference": True,
                                        "one_step_matches_adapted_prediction": True,
                                        "all_parameters_and_buffers_fixed_through_rollout": True,
                                        "no_intermediate_observations_in_prediction": True}}
                                if schedule_name is not None:
                                    row["schedule"] = schedule_name
                                    row["checks"].pop("all_parameters_and_buffers_fixed_through_rollout")
                                    row["checks"].update(adapted["checks"])
                                    row["checks"]["first_update_support_matches_all_schedules"] = True
                                records.append(stream.append(row))
                stream.flush()
                done, frequency = index+1, config["evaluation"]["log_every"]
                if done == len(queries) or (frequency and done % frequency == 0):
                    write_json(output/"status.json", {"status": "running", "split": split,
                        "completed_queries": done, "queries": len(queries), "completed_records": len(records)})
                query_progress.set_postfix(records=len(records), refresh=False)
        if records_path is not None and not records_path.stat().st_size:
            raise ValueError("Cannot complete without persistent records")
        expected_rows = len(queries)*len(scope_names(config))*len(schedule_names(config))*len(methods)*len(seeds)
        if len(records) != expected_rows:
            raise ValueError("Incomplete scope/query/method/seed evaluation")
        if is_schedule_comparison(config):
            print("Summarizing paired schedules, per-horizon budgets and episode uncertainty...", flush=True)
        summary = summarize(records, config, lock["accurate_thresholds"] if lock else None)
        if summary["queries"] != len(queries):
            raise ValueError("Summary does not cover the selected query cohort")
        from ..cache_loading import report_fields
        report = {**report_fields(resources), "completed": True, "stage": stage_name(config), "split": split, "config": config,
            "code": implementation, "manifest_fingerprint": manifest["fingerprint"],
            "p1_prepared_fingerprint": resources.prepared["fingerprint"],
            "model": resources.adapter.identity, "scope_parameters": descriptions,
            "checks": checks, "summary": summary, "cohort": manifest["splits"][split],
            "records_file": str(records_path.resolve()) if records_path else None,
            "records_sha256": file_hash(records_path) if records_path else None,
            "record_storage": writer.metadata(),
            "record_count": len(records), "evaluation_seconds": time.perf_counter()-started,
            "limited_run": config["evaluation"]["query_limit"] is not None,
            "query_episode_positions": sorted({row["query"]["source_episode_position"] for row in records}),
            "gpu_memory": _gpu_memory(resources.adapter),
            "limits": ["Independent-query adaptation on recorded controls; not persistent JEPA-TTT",
                ("Fresh q4 sample from maximum-horizon-safe windows; not the archived P1 q4 or P2 q16 cohort"
                 if is_scope_comparison(config) else "Common cohort filters short futures at the maximum requested horizon"),
                "P1 test episodes already inspected; no fresh episode holdout",
                "No counterfactual action ranking, simulator task success, or dynamics-shift result",
                "Timing includes rank diagnostics, per-module metrics and frozen-state checks"]}
        if is_schedule_comparison(config):
            report["schedule_definitions"] = {name: config["schedules"][name] for name in schedule_names(config)}
            report["protocol"] = "detached recorded-action rollouts; placement and within-position refresh controlled separately"
            report["limits"][1] = "q4 sampled at cohort_horizon; scored horizons are recorded explicitly in config"
            report["limits"][-1] = "Timing includes support reevaluation and state checks; exhaustive rank diagnostics omitted"
        if config.get("query_dataset"):
            report["query_source_fingerprint"] = manifest["query_source_fingerprint"]
            report["limits"][1] = "Uses the fresh collection's prespecified horizon-safe query windows without resampling"
            report["limits"][2] = "Fresh simulator queries; exact released-source state audit passed; unknown additional pretraining data cannot be audited"
        path = output/"report.json"
        write_json(path, report)
        text = format_summary(summary, path)
        if records_path is None:
            text = text.replace("Compact per-query records are persisted; no simulator counterfactual claims.",
                                "Summary-only run; per-query reanalysis and record-based locking are unavailable.")
        (output/"summary.txt").write_text(text)
        write_json(output/"status.json", {"status": "complete", "split": split})
        print(text, end="", flush=True)
        print(f"Records: {writer.mode}, {writer.metadata()['bytes']/1024**2:.2f} MiB", flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "split": split,
            "error": f"{type(error).__name__}: {error}", "completed_records": len(records)})
        raise


def read_records(report: dict) -> list[dict]:
    return read_saved_records(report)


def analyze(report_path: Path, output: Path | None = None) -> dict:
    report = read_json(report_path)
    if not report.get("completed") or report.get("stage") not in (P2_STAGE, SCOPE_STAGE, SCHEDULE_STAGE):
        raise ValueError("Expected a completed Rollout report")
    validate_config(report["config"])
    if report["stage"] != stage_name(report["config"]):
        raise ValueError("Report stage differs from its configuration")
    summary = summarize(read_records(report), report["config"], report["summary"]["accurate_thresholds"])
    if summary != report["summary"]:
        raise ValueError("Saved summary differs from recomputed records")
    target = output or report_path.with_name("analysis.json")
    if target.exists() or target.resolve() == report_path.resolve():
        raise FileExistsError("Choose a new analysis filename")
    value = {"source_report": str(report_path.resolve()), "report_sha256": file_hash(report_path),
        "source_report_sha256": file_hash(report_path),
        "records_sha256": report["records_sha256"], "summary": summary}
    write_json(target, value)
    print(format_summary(summary, target), end="", flush=True)
    return value


def lock_validation(report_path: Path, output: Path) -> dict:
    report = read_json(report_path)
    if (not report.get("completed") or report.get("stage") not in (P2_STAGE, SCOPE_STAGE, SCHEDULE_STAGE) or report["split"] != "validation"
            or report.get("limited_run")):
        raise ValueError("Only a completed full Rollout validation run can be locked")
    validate_config(report["config"])
    if report["stage"] != stage_name(report["config"]):
        raise ValueError("Report stage differs from its configuration")
    if report["code"] != code_identity(entrypoint(report["config"])):
        raise ValueError("Rollout implementation changed since validation")
    summary = summarize(read_records(report), report["config"], report["summary"]["accurate_thresholds"])
    if summary != report["summary"]:
        raise ValueError("Persistent records disagree with the saved summary")
    manifest = load_manifest(report["config"])
    if manifest["fingerprint"] != report["manifest_fingerprint"]:
        raise ValueError("Validation cohort identity changed")
    if output.exists():
        raise FileExistsError("Choose a new lock filename")
    value = {"stage": report["stage"], "config": report["config"], "code": report["code"],
        "manifest_fingerprint": manifest["fingerprint"], "accurate_thresholds": summary["accurate_thresholds"],
        "validation_report": str(report_path.resolve()), "report_sha256": file_hash(report_path),
        "note": "Rollout-specific scope/config lock; P1 test episodes are already inspected"}
    write_json(output, value)
    print(f"Locked Rollout validation: {output}", flush=True)
    return value
