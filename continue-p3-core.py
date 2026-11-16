#!/usr/bin/env python3
"""Finish the missing P3 validation queries from an offline recovered report.

Run in the original dataset's model/simulator environment. The original core
pipeline evaluates only a suffix view of its main query bank; acceptance checks
still use the original validation bank. Existing evaluation sources and saved
artifacts remain unchanged. No synthetic clean-stop receipt is constructed.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PARTIAL_STAGE = "P3-core-recovered-partial"
SEGMENT_STAGE = "P3-core-continuation-segment"
COMBINED_STAGE = "P3-core-recovered-continuation"
SPEC = importlib.util.spec_from_file_location("p3_recovery", Path(__file__).with_name("recover-p3-core.py"))
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def extension_identity():
    return {str(path.relative_to(ROOT)): recovery.digest(path) for path in
            (Path(__file__).resolve(), Path(recovery.__file__).resolve(), ROOT/"codes/run-p3-continue.sh")}


def matrix_keys(config):
    from lewm_tta.core.config import cells
    return {(j, seed, label) for j in config["core"]["iterations"]
            for seed in config["p1"]["evaluation"]["seeds"] for label in cells(config).values()}


def verify_artifacts(report, expected, ordered_ids):
    """Check hashes, complete matrices, physical receipts and a contiguous view."""
    from lewm_tta.core.reporting import summarize
    paths = {name: Path(report[name+"_file"]).resolve() for name in recovery.STREAMS}
    recovery.assert_idle(paths.values())
    for name, path in paths.items():
        if recovery.digest(path) != report[name+"_sha256"]:
            raise ValueError(f"Changed {name} artifact: {path}")
    keys = matrix_keys(report["config"])
    records, plans, calls, receipts = recovery.scan(paths, report["config"], expected, keys)
    selected = set(ordered_ids)
    if set(records) != selected or set(plans)-selected or set(calls)-selected:
        raise ValueError("Artifact queries differ from the expected contiguous query view")
    for name, receipt in receipts.items():
        count_key = {"records": "record_count", "plans": "plan_count", "calls": "call_count"}[name]
        if (receipt["truncated_gzip"] or receipt["discarded_tail_bytes"]
                or receipt["rows"] != report[count_key]):
            raise ValueError(f"Incomplete or mismatched repaired {name} stream")
    rows = []
    for identity in ordered_ids:
        reason = recovery.complete_case(identity, records[identity], plans[identity], calls[identity], keys, report["config"])
        if reason:
            raise ValueError(f"Incomplete paired query {identity}: {reason}")
        rows.extend(records[identity])
    controls = []
    for _, row in recovery.json_lines(paths["records"], {}):
        if row["kind"] == "recorded-control":
            controls.append(row["case"]["case_id"])
    if controls != ordered_ids:
        raise ValueError("Saved query order differs from the cohort")
    if summarize(rows, report["config"]) != report["summary"]:
        raise ValueError("Saved outcomes no longer reproduce the report summary")
    for name, path in paths.items():
        if recovery.digest(path) != report[name+"_sha256"]:
            raise ValueError(f"Artifact changed during validation: {path}")
    return rows


def load_partial(path):
    path = Path(path).resolve()
    checksum = recovery.digest(path)
    report = recovery.read_json(path)
    if report.get("stage") != PARTIAL_STAGE or report.get("split") != "validation":
        raise ValueError("Provide a P3-core-recovered-partial validation report")
    source = report["recovery"]["source_files"]
    for item in source.values():
        if recovery.digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Original recovery source changed: {item['path']}")
    for name, sha in report["recovery"]["reporting_code"].items():
        if recovery.digest(ROOT/name) != sha:
            raise ValueError(f"Recovered reporting/config implementation changed: {name}")
    config = report["config"]
    if recovery.read_json(Path(report["recovery"]["source_run"])/"config.json") != config:
        raise ValueError("Recovered and original saved configurations differ")
    manifest = recovery.checked_manifest(Path(config["experiment_dir"])/"manifest.json")
    query = recovery.checked_manifest(Path(report["query_manifest"]))
    prepared = recovery.checked_manifest(Path(config["p1"]["paths"]["experiment_dir"])/"prepared.json")
    if (manifest["fingerprint"] != report["manifest_fingerprint"]
            or query["fingerprint"] != report["query_source_fingerprint"]
            or prepared["fingerprint"] != report["p1_prepared_fingerprint"]):
        raise ValueError("Prepared model/assets, query source or cohort changed")
    if recovery.digest(Path(report["query_manifest"])) != report["query_manifest_sha256"]:
        raise ValueError("Recovered query manifest changed")
    if recovery.read_json(Path(source["checks"]["path"])) != report["checks"]:
        raise ValueError("Saved acceptance checks changed")
    expected = recovery.expected_cases(config, manifest, query, "validation")
    completed = report["completed_queries"]
    if (type(completed) is not int or not 0 < completed < len(expected)
            or report["requested_queries"] != len(expected)):
        raise ValueError("Recovered report must describe a proper prefix with queries still missing")
    ids = list(expected)[:completed]
    if ids != report["recovery"]["audit"]["retained_case_ids"]:
        raise ValueError("Recovery audit is not the saved cohort's prefix")
    rows = verify_artifacts(report, expected, ids)
    if recovery.digest(path) != checksum:
        raise ValueError("Recovered report changed during verification")
    return {"path": path, "sha256": checksum, "report": report,
            "manifest": manifest, "expected": expected, "rows": rows}


def checked_segment(path, source, code, extension):
    checksum = recovery.digest(path)
    report = recovery.read_json(path)
    receipt = report.get("continuation", {})
    prefix = source["report"]["completed_queries"]
    remaining = list(source["expected"])[prefix:]
    if (report.get("stage") != SEGMENT_STAGE or report.get("config") != source["report"]["config"]
            or report.get("code") != code or receipt.get("extension_code") != extension
            or receipt.get("backend_code") != code
            or receipt.get("source_report_sha256") != source["sha256"]
            or receipt.get("source_report") != str(source["path"])
            or receipt.get("selected_case_ids") != remaining
            or report.get("requested_queries") != len(remaining)
            or report.get("manifest_fingerprint") != source["report"]["manifest_fingerprint"]
            or report.get("p1_prepared_fingerprint") != source["report"]["p1_prepared_fingerprint"]
            or report.get("query_source_fingerprint") != source["report"]["query_source_fingerprint"]):
        raise ValueError("Continuation segment differs from the recovered config, cohort or implementation")
    if not report["completed"] or report["completed_queries"] != len(remaining):
        raise ValueError("Continuation segment stopped early; preserve it and use a new --run-name to retry the suffix")
    rows = verify_artifacts(report, source["expected"], remaining)
    if recovery.digest(path) != checksum:
        raise ValueError("Continuation report changed during verification")
    return report, rows


def suffix_bank_factory(original, source):
    """Restrict only evaluate's first/main bank call; keep acceptance banks full."""
    selected = False
    report = source["report"]
    prefix = report["completed_queries"]
    wanted = list(source["expected"])[prefix:]

    def evaluation_bank(resources, manifest, split, limit=None):
        nonlocal selected
        if selected:
            return original(resources, manifest, split, limit)
        if split != report["split"] or limit != report["config"]["evaluation"]["query_limit"]:
            raise ValueError("Unexpected main query-bank request; core evaluation interface changed")
        selected = True
        sliced = deepcopy(manifest)
        starts = sliced["splits"][split]["raw_start_rows"]
        if limit is not None:
            starts = starts[:limit]
        sliced["splits"][split]["raw_start_rows"] = starts[prefix:]
        sliced["splits"][split]["queries"] = len(wanted)
        bank = original(resources, sliced, split, None)
        if [bank.items[index]["transition_id"] for index in range(len(bank))] != wanted:
            raise ValueError("Live query bank does not match the missing cohort suffix")
        return bank
    return evaluation_bank


def run_segment(source, run_name, device, cpu_threads, code, extension):
    from lewm_tta.core import pipeline
    original_write = pipeline.write_json
    original_load = pipeline.load_resources
    config = source["report"]["config"]
    directory = Path(config["experiment_dir"])/"runs"/run_name
    receipt = {"source_report": str(source["path"]), "source_report_sha256": source["sha256"],
               "reused_queries": source["report"]["completed_queries"],
               "selected_case_ids": list(source["expected"])[source["report"]["completed_queries"]:],
               "extension_code": extension, "backend_code": code,
               "created_utc": datetime.now(timezone.utc).isoformat()}

    def write(path, value):
        path = Path(path)
        if path == directory/"report.json":
            if extension_identity() != extension:
                raise ValueError("Continuation helper source changed during evaluation")
            value.update(stage=SEGMENT_STAGE, limited_run=True, continuation=deepcopy(receipt))
            value["protocol"] = "Continuation suffix only. "+value["protocol"]
        original_write(path, value)
        if path == directory/"config.json":
            original_write(directory/"continuation-request.json", receipt)

    def load(*args, **kwargs):
        resources = original_load(*args, **kwargs)
        if (resources.prepared["fingerprint"] != source["report"]["p1_prepared_fingerprint"]
                or resources.external_manifest["fingerprint"] != source["report"]["query_source_fingerprint"]):
            raise ValueError("Live prepared model/assets or query source changed")
        receipt["model"] = resources.adapter.identity
        receipt["runtime"] = {"device": str(resources.adapter.device), "cpu_threads": cpu_threads,
                              "gpu": pipeline.torch.cuda.get_device_name(resources.adapter.device)
                              if resources.adapter.device.type == "cuda" else None}
        original_write(directory/"continuation-request.json", receipt)
        return resources

    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, "evaluation_bank", suffix_bank_factory(pipeline.evaluation_bank, source)))
        stack.enter_context(patch.object(pipeline, "write_json", write))
        stack.enter_context(patch.object(pipeline, "load_resources", load))
        pipeline.evaluate(config, device, cpu_threads, split="validation", run_name=run_name)
    return directory/"report.json"


def add_counters(left, right):
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict):
            result[key] = add_counters(result.get(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[key] = result.get(key, 0)+value
        else:
            raise ValueError(f"Invalid accounting counter: {key}")
    return result


def merge(source, segment_path, segment, rows, output):
    from lewm_tta.core.reporting import summarize, write_tables
    if recovery.read_json(segment_path) != segment:
        raise ValueError("Continuation report changed before merging")
    report = deepcopy(source["report"])
    report.update(stage=COMBINED_STAGE, completed=True, budget_exhausted=False,
                  completed_queries=report["requested_queries"], limited_run=report["config"]["evaluation"]["query_limit"] is not None,
                  summary=summarize(source["rows"]+rows, report["config"]), validation_lock_eligible=False)
    report["protocol"] = segment["protocol"].removeprefix("Continuation suffix only. ")+(
        " Recovered prefix and freshly evaluated suffix cover the configured cohort. "
        "Legacy prefix startup source/model/runtime provenance remains unavailable; this report cannot create a validation lock.")
    report["continuation"] = {"source_report": str(source["path"]), "source_report_sha256": source["sha256"],
        "segment_report": str(segment_path), "segment_report_sha256": recovery.digest(segment_path),
        "reused_queries": source["report"]["completed_queries"], "new_queries": segment["completed_queries"],
        "backend_code": segment["code"], "extension_code": segment["continuation"]["extension_code"],
        "new_segment_evaluation_seconds": segment["evaluation_seconds"],
        "new_segment_loading_and_checks_seconds": segment["resource_loading_and_checks_seconds"],
        "new_segment_gpu_memory": segment["gpu_memory"], "new_segment_model": segment["model"],
        "new_segment_runtime": segment["runtime"]}
    if source["report"]["accounting_scope"] == "Saved committed-prefix counters; excludes unfinished query/tail work":
        report["accounting"] = add_counters(report["accounting"], segment["accounting"])
        report["accounting"]["actual_cem_transition_predictions"] = (
            report["accounting"]["actual_cem_sequence_evaluations"]*report["config"]["planning"]["horizon"])
        report["accounting_scope"] = "Committed recovered-prefix plus continuation counters; discarded original unfinished work excluded"
    else:
        report["accounting"] = None
        report["accounting_scope"] = "Original prefix work cannot be isolated; retain per-source counters separately"
    report["recovery"]["selection_limit"] = "All configured query outcomes are now present; bootstrap intervals use the combined cohort."
    report["recovery"]["continuation_note"] = "The original recovery audit describes the reused prefix only."
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p3-continuation-", dir=output.parent))
    try:
        for name in recovery.STREAMS:
            target = temporary/(name+".jsonl.gz")
            with target.open("xb") as destination:
                for part in (source["report"], segment):
                    part_path = Path(part[name+"_file"])
                    if recovery.digest(part_path) != part[name+"_sha256"]:
                        raise ValueError(f"Input {name} changed before merging")
                    with part_path.open("rb") as stream:
                        shutil.copyfileobj(stream, destination)
                    if recovery.digest(part_path) != part[name+"_sha256"]:
                        raise ValueError(f"Input {name} changed while merging")
            count_key = {"records": "record_count", "plans": "plan_count", "calls": "call_count"}[name]
            report.update({name+"_file": str(output/target.name), name+"_sha256": recovery.digest(target),
                           count_key: source["report"][count_key]+segment[count_key]})
        recovery.write_json(temporary/"report.json", report)
        recovery.write_json(temporary/"config.json", report["config"])
        write_tables(temporary, report)
        with (temporary/"summary.md").open("a") as stream:
            stream.write("\nReused "+str(source["report"]["completed_queries"])+" saved queries; evaluated "
                         +str(segment["completed_queries"])+" missing queries.\n\n"
                         +report["recovery"]["provenance_limit"]+"\n\n"
                         "All configured outcomes are present. Original total wall time and peak GPU memory remain unknown; "
                         "new-segment measurements are recorded separately. This recovered report cannot create a validation lock.\n")
        # Check the final gzip concatenation, counts and combined matrices before publication.
        check = deepcopy(report)
        for name in recovery.STREAMS:
            check[name+"_file"] = str(temporary/(name+".jsonl.gz"))
        verify_artifacts(check, source["expected"], list(source["expected"]))
        if recovery.digest(source["path"]) != source["sha256"] or recovery.digest(segment_path) != report["continuation"]["segment_report_sha256"]:
            raise ValueError("Input report changed while merging")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Preserve existing merged output: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Combined {report['completed_queries']}/{report['requested_queries']} queries: {output/'summary.md'}", flush=True)
    return report


def continue_run(report_path, run_name=None, output=None, device="cuda", cpu_threads=8, inspect_only=False):
    if cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    source = load_partial(report_path)
    partial = source["report"]
    run_name = run_name or f"validation-continuation-q{partial['completed_queries']}"
    if Path(run_name).name != run_name or run_name in (".", "..", "validation", "test"):
        raise ValueError("Use a distinct, simple continuation run name")
    directory = Path(partial["config"]["experiment_dir"])/"runs"/run_name
    if directory.resolve() == Path(partial["recovery"]["source_run"]).resolve():
        raise ValueError("Continuation must not write into the interrupted original run")
    output = Path(output).resolve() if output else Path(partial["recovery"]["source_run"])/"continued"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Merged output already exists: {output}")
    if output == directory.resolve() or output in directory.resolve().parents:
        raise ValueError("Merged output must be separate from the continuation segment")
    print(f"Reuse {partial['completed_queries']} queries; evaluate {partial['requested_queries']-partial['completed_queries']} "
          f"missing queries with unchanged cells/seeds/J/control budget. Segment: {directory}", flush=True)
    if inspect_only:
        print("Missing query IDs:\n"+"\n".join(list(source["expected"])[partial["completed_queries"]:]))
        return source
    from experiments.provenance import backend_identity
    from lewm_tta.runtime import configure_logging
    configure_logging()
    code, extension = backend_identity("p3_core"), extension_identity()
    segment_path = directory/"report.json"
    if directory.exists():
        if not segment_path.is_file():
            raise FileExistsError("Interrupted continuation preserved. Select a new --run-name to retry only the missing suffix.")
    else:
        segment_path = run_segment(source, run_name, device, cpu_threads, code, extension)
    segment, rows = checked_segment(segment_path, source, code, extension)
    if backend_identity("p3_core") != code or extension_identity() != extension:
        raise ValueError("Continuation source changed; segment preserved without merging")
    # Recheck all saved sources after GPU work, before producing the combined report.
    refreshed = load_partial(report_path)
    if refreshed["sha256"] != source["sha256"]:
        raise ValueError("Recovered source report changed during continuation")
    source = refreshed
    return merge(source, segment_path, segment, rows, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="recovered-partial/report.json")
    parser.add_argument("--run-name", help="New suffix segment name (default: validation-continuation-qN)")
    parser.add_argument("--output", type=Path, help="New combined directory (default: original RUN/continued)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--inspect", action="store_true", help="Validate prefix and print missing IDs without GPU work")
    args = parser.parse_args()
    try:
        continue_run(args.report, args.run_name, args.output, args.device, args.cpu_threads, args.inspect)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Continuation refused: {error}\n")


if __name__ == "__main__":
    main()
