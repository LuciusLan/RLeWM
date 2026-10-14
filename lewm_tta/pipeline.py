"""P1 orchestration; dataset preparation and evaluation are independent steps."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import torch

from .adaptation import AdaptationEngine
from .cache import build_or_load_cache
from .checks import check_frame_bank_layout, run_checks
from .config import ROOT, validate_config
from .data import action_statistics, get_manifest
from .model import LeWMAdapter
from .reporting import format_summary, score_record, summarize
from .retrieval import calibrate_scales, make_retriever, prepare_retrieval
from .runtime import file_hash, object_hash, read_json, write_json


def code_identity() -> dict:
    sources = sorted((ROOT / "codes" / "lewm_tta").glob("*.py")) + [ROOT / "codes" / "p1.py"]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sources}


def initialize(config: dict, device: str, cpu_threads: int):
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["data"]["split_seed"])
    adapter = LeWMAdapter(Path(config["paths"]["repo"]), Path(config["paths"]["checkpoint"]), device)
    if config["data"]["history"] != adapter.history:
        raise ValueError("Configured history differs from the checkpoint")
    manifest = get_manifest(config)
    return adapter, manifest


def prepare(config: dict, device: str, cpu_threads: int, batch_size: int) -> dict:
    started = time.perf_counter()
    validate_config(config)
    backend_checks = {}
    if config["retrieval"].get("backend", "torch") == "cuvs":
        from .cuvs_retrieval import check_cuvs_backend
        backend_checks = check_cuvs_backend(device)
        print(f"cuVS {backend_checks['version']}: synthetic backend checks passed", flush=True)
    adapter, manifest = initialize(config, device, cpu_threads)
    layout_checks = check_frame_bank_layout() if manifest["version"] == 2 else {}
    normalization_path = Path(config["paths"]["experiment_dir"]) / "action-normalization.json"
    if normalization_path.exists():
        normalization = read_json(normalization_path)
    else:
        normalization = action_statistics(manifest)
        write_json(normalization_path, normalization)
    memory = build_or_load_cache(adapter, manifest, normalization, "memory",
                                Path(config["paths"]["cache_dir"]), batch_size,
                                config.get("cache", {}).get("checkpoint_rows", 8192))
    validation = build_or_load_cache(adapter, manifest, normalization, "validation",
                                    Path(config["paths"]["cache_dir"]), batch_size)
    test = build_or_load_cache(adapter, manifest, normalization, "test",
                              Path(config["paths"]["cache_dir"]), batch_size) if manifest["version"] == 2 else None
    calibration_settings = {key: config["retrieval"][key] for key in ("calibration_pairs", "distance_floor")}
    calibration = calibrate_scales(memory, calibration_settings["calibration_pairs"],
                                   config["data"]["split_seed"], calibration_settings["distance_floor"])
    retrieval_assets = prepare_retrieval(memory, calibration, config["retrieval"],
                                        Path(config["paths"]["cache_dir"]), device)
    prepared = {"manifest": manifest["fingerprint"], "model": adapter.identity,
                "model_fingerprint": adapter.fingerprint, "normalization": normalization,
                "calibration": calibration, "calibration_settings": calibration_settings,
                "memory_cache": memory.identity, "validation_cache": validation.identity,
                "parameter_names": adapter.parameter_names, "parameter_count": adapter.parameter_count,
                "adaptable_module": adapter.adaptable_module,
                "test_cache": test.identity if test is not None else None,
                "cache_counts": {"memory": len(memory), "validation": len(validation),
                                 "test": len(test) if test is not None else None},
                "layout_checks": layout_checks, "retrieval_assets": retrieval_assets,
                "retrieval_backend_checks": backend_checks}
    prepared["fingerprint"] = object_hash(prepared)
    output = Path(config["paths"]["experiment_dir"]) / "prepared.json"
    write_json(output, prepared)
    if manifest["version"] == 2:
        rows = manifest["source"]["rows"]
        payload = rows * (adapter.latent_dim + memory.store.actions.shape[1]) * 4
        payload += sum(prepared["cache_counts"].values()) * 8
        cache_summary = {"raw_observations": rows, "total_windows": sum(prepared["cache_counts"].values()),
            "window_counts": prepared["cache_counts"],
            "episode_counts": {split: len(group) for split, group in manifest["episodes"].items()},
            "evaluation_queries": {split: selection["count"] for split, selection in manifest["evaluation_windows"].items()},
            "numeric_payload_bytes": payload, "numeric_payload_gib": payload / 2**30,
            "retrieval_key_payload_bytes": retrieval_assets.get("numeric_payload_bytes", 0),
            "retrieval_key_payload_gib": retrieval_assets.get("numeric_payload_bytes", 0) / 2**30,
            "layout": "one projected latent per raw observation, normalized raw actions, int64 window starts",
            "observations": memory.store.identity}
        write_json(Path(config["paths"]["experiment_dir"]) / "cache-summary.json", cache_summary)
        print(f"P1 full cache prepared | {rows:,} observations | memory {len(memory):,} | "
              f"validation {len(validation):,} | test {len(test):,}\n"
              f"Shared cache: {payload/2**30:.2f} GiB | "
              f"retrieval keys: {retrieval_assets.get('numeric_payload_bytes', 0)/2**30:.2f} GiB | "
              f"{time.perf_counter()-started:.1f}s\n"
              f"All splits cached; test remains unscored.\nPrepared record: {output}", flush=True)
    else:
        print(f"P1 prepared | memory {len(memory)} | validation {len(validation)} | "
              f"{time.perf_counter()-started:.1f}s\nTest episodes reserved; test cache not built.\nPrepared record: {output}", flush=True)
    return prepared


def evaluate(config: dict, device: str, cpu_threads: int, batch_size: int,
             split: str = "validation", run_name: str | None = None, lock: dict | None = None) -> dict:
    validate_config(config)
    if split == "test" and lock is None:
        raise ValueError("Test evaluation requires a locked validation configuration")
    adapter, manifest = initialize(config, device, cpu_threads)
    prepared = read_json(Path(config["paths"]["experiment_dir"]) / "prepared.json")
    if prepared["manifest"] != manifest["fingerprint"] or prepared["model_fingerprint"] != adapter.fingerprint:
        raise ValueError("Prepared data/model identity changed; rerun prepare in a compatible experiment directory")
    expected_calibration = {key: config["retrieval"][key] for key in ("calibration_pairs", "distance_floor")}
    if prepared["calibration_settings"] != expected_calibration:
        raise ValueError("Memory calibration settings changed; run prepare again")
    implementation = code_identity()
    if lock is not None and (lock["prepared_fingerprint"] != prepared["fingerprint"] or
                             lock["code"] != implementation or lock["config"] != config):
        raise ValueError("Locked configuration, code, or prepared identity changed")
    memory = build_or_load_cache(adapter, manifest, prepared["normalization"], "memory",
                                Path(config["paths"]["cache_dir"]), batch_size).to(adapter.device)
    queries = build_or_load_cache(adapter, manifest, prepared["normalization"], split,
                                 Path(config["paths"]["cache_dir"]), batch_size).to(adapter.device)
    if memory.identity != prepared["memory_cache"] or (split == "validation" and
                                                       queries.identity != prepared["validation_cache"]):
        raise ValueError("Cache implementation/identity changed since preparation; run prepare again")
    if split == "test" and prepared.get("test_cache") is not None and queries.identity != prepared["test_cache"]:
        raise ValueError("Test cache identity changed since preparation")
    queries = queries.for_evaluation()
    if not len(queries):
        raise ValueError("No eligible query windows")
    if config["retrieval"].get("backend", "torch") == "cuvs" and "retrieval_assets" not in prepared:
        raise ValueError("cuVS assets are not recorded in prepared.json; run prepare again")
    retriever = make_retriever(memory, prepared["calibration"], config["retrieval"],
                               Path(config["paths"]["cache_dir"]))
    if config["retrieval"].get("backend", "torch") == "cuvs":
        assets = retriever.assets if len(memory) else prepared["retrieval_assets"]
        if assets["fingerprint"] != prepared["retrieval_assets"]["fingerprint"]:
            raise ValueError("cuVS index keys changed since preparation; run prepare again")
    engine = AdaptationEngine(adapter, memory, retriever, config["adaptation"])
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["paths"]["experiment_dir"]) / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"status": "running", "split": split})
    write_json(output / "query-episodes.json", {"split": split, "manifest_fingerprint": manifest["fingerprint"],
        "query_episode_positions": sorted({item["source_episode_position"] for item in queries.items})})
    checks_started = time.perf_counter()
    # Test checks run only on validation queries, never on test labels or windows.
    check_queries = queries if split == "validation" else build_or_load_cache(
        adapter, manifest, prepared["normalization"], "validation", Path(config["paths"]["cache_dir"]), batch_size).to(adapter.device)
    if check_queries.identity != prepared["validation_cache"]:
        raise ValueError("Validation cache identity changed since preparation")
    check_queries = check_queries.for_evaluation()
    checks = run_checks(engine, check_queries, config["evaluation"]["seeds"][0])
    checks["elapsed_seconds"] = time.perf_counter() - checks_started
    write_json(output / "acceptance-checks.json", checks)
    records, started = [], time.perf_counter()
    import json
    records_path = output / "records.jsonl"
    with records_path.open("w") as stream:
        for index in range(len(queries)):
            query = queries.query(index)
            initial_reference = None
            for seed in config["evaluation"]["seeds"]:
                paired = {}
                for method in config["adaptation"]["methods"]:
                    record = engine.run(query, method, seed)
                    initial = torch.tensor(record["prediction_initial"])
                    if initial_reference is None:
                        initial_reference = initial
                    torch.testing.assert_close(initial, initial_reference, rtol=1e-6, atol=1e-7)
                    # True query successors are used here, after the adapter returns.
                    score_record(record, queries.successors[index].cpu().tolist())
                    if config["adaptation"]["updates"] == 0:
                        assert record["mse_final"] == record["mse_initial"]
                    records.append(record)
                    paired[method] = record
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if all(method in paired and paired[method]["rounds"] for method in ("fr-fixed", "fr-refresh")):
                    torch.testing.assert_close(torch.tensor(paired["fr-fixed"]["rounds"][0]["prediction_after"]),
                        torch.tensor(paired["fr-refresh"]["rounds"][0]["prediction_after"]), rtol=1e-6, atol=1e-7)
            done, frequency = index + 1, config["evaluation"]["log_every"]
            if done == len(queries) or (frequency and done % frequency == 0):
                stream.flush()
                print(f"P1 {split}: {done}/{len(queries)} queries | {time.perf_counter()-started:.1f}s", flush=True)
    summary = summarize(records, config, lock["accurate_threshold"] if lock else None)
    report = {"completed": True, "stage": "P1", "split": split, "config": config,
              "prepared_fingerprint": prepared["fingerprint"], "manifest_fingerprint": manifest["fingerprint"],
              "model": adapter.identity, "code": implementation, "checks": checks,
              "retrieval_backend": config["retrieval"].get("backend", "torch"),
              "retrieval_runtime": getattr(retriever, "backend_info", {"backend": "torch"}),
              "retrieval_assets": prepared.get("retrieval_assets"),
              "retrieval_backend_checks": prepared.get("retrieval_backend_checks"),
              "parameter_names": adapter.parameter_names, "parameter_count": adapter.parameter_count,
              "records_file": str(records_path), "records_sha256": file_hash(records_path),
              "summary": summary, "evaluation_seconds": time.perf_counter()-started,
              "query_episode_positions": sorted({item["source_episode_position"] for item in queries.items}),
              "checkpoint_overlap": manifest["checkpoint_overlap"],
              "limits": ["Physical contact/motion subsets deferred until state coordinates are audited",
                         "Prototype latency includes full-corpus distance/rank diagnostics and frozen-state checks",
                         "Rank consensus is fixed at the initial prediction; this run does not establish refresh utility"]}
    report_path = output / "report.json"
    write_json(report_path, report)
    text = format_summary(summary, report_path)
    (output / "summary.txt").write_text(text)
    write_json(output / "status.json", {"status": "complete", "split": split})
    print(text, end="", flush=True)
    return report


def lock_validation(report_path: Path, output: Path) -> dict:
    report = read_json(report_path)
    if not report.get("completed") or report["split"] != "validation":
        raise ValueError("Only a completed validation report can be locked")
    if report["code"] != code_identity() or file_hash(Path(report["records_file"])) != report["records_sha256"]:
        raise ValueError("Validation code or records changed since evaluation")
    prepared = read_json(Path(report["config"]["paths"]["experiment_dir"]) / "prepared.json")
    if prepared["fingerprint"] != report["prepared_fingerprint"]:
        raise ValueError("Prepared identity changed since validation")
    if Path(output).exists():
        raise FileExistsError("Lock already exists; use a new filename for another selected configuration")
    locked = {"config": report["config"], "prepared_fingerprint": report["prepared_fingerprint"],
              "code": report["code"], "accurate_threshold": report["summary"]["accurate_threshold"],
              "validation_report": str(Path(report_path).resolve()), "report_sha256": file_hash(report_path),
              "note": "Selected on validation; test labels cannot alter configuration or thresholds"}
    write_json(output, locked)
    print(f"Locked validation configuration: {output}")
    return locked
