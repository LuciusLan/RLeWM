#!/usr/bin/env python3
"""Recover executed P3 core outcomes offline, without changing the original run.

This entry point deliberately lives outside the core backend's source identity:
adding it must not invalidate an evaluation already running in another process.
Only reporting code (NumPy) is imported; no model, CUDA or simulator is loaded.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
STREAMS = ("records", "plans", "calls")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_manifest(path):
    value = read_json(path)
    if value.get("fingerprint") != object_digest({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Manifest fingerprint mismatch: {path}")
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def finite_json(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def json_lines(path, receipt):
    """Read complete lines from all gzip members, tolerating only a torn EOF.

    Missing gzip trailers and a final non-newline-terminated JSON row are common
    after SIGKILL. CRC/deflate errors and malformed *complete* rows remain fatal.
    Stream the large plan/call diagnostics rather than retaining them in RAM.
    """
    decoder, pending = None, b""
    receipt.update(rows=0, finished_gzip_members=0, truncated_gzip=False, discarded_tail_bytes=0)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(64*1024), b""):
            while block:
                if decoder is None:
                    decoder = zlib.decompressobj(16+zlib.MAX_WBITS)
                try:
                    data = decoder.decompress(block)
                except zlib.error as error:
                    raise ValueError(f"Corrupt gzip stream {path}: {error}") from error
                pending += data
                parts = pending.split(b"\n")
                pending = parts.pop()
                for line in parts:
                    receipt["rows"] += 1
                    try:
                        row = json.loads(line, parse_constant=finite_json)
                    except (ValueError, UnicodeError) as error:
                        raise ValueError(f"Malformed complete JSON row {receipt['rows']} in {path}") from error
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected JSON object in {path}:{receipt['rows']}")
                    yield line, row
                block = decoder.unused_data
                if decoder.eof:
                    receipt["finished_gzip_members"] += 1
                    decoder = None
                elif block:
                    raise ValueError(f"Unexpected compressed tail in {path}")
    receipt["truncated_gzip"] = decoder is not None
    receipt["discarded_tail_bytes"] = len(pending)


def assert_idle(paths):
    """Refuse a live writer when /proc exposes its file descriptors."""
    targets = {str(path.resolve()) for path in paths}
    for process in Path("/proc").glob("[0-9]*"):
        try:
            descriptors = list((process/"fd").iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
                if target not in targets:
                    continue
                fields = (process/"fdinfo"/descriptor.name).read_text().splitlines()
                flags = next(int(line.split()[1], 8) for line in fields if line.startswith("flags:"))
                if (flags & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR):
                    raise ValueError(f"Run is still being written by PID {process.name}: {target}")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue


def source_receipts(paths):
    result = {}
    for name, path in paths.items():
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while hashing: {path}")
        result[name] = {"path": str(path), "bytes": after.st_size,
                        "mtime_ns": after.st_mtime_ns, "sha256": sha}
    return result


def expected_cases(config, manifest, query, split):
    if manifest.get("query_source_fingerprint") != query["fingerprint"]:
        raise ValueError("Cohort and query source fingerprints differ")
    if manifest["history"] != config["p1"]["data"]["history"] or manifest["frameskip"] != config["p1"]["data"]["frameskip"]:
        raise ValueError("Saved cohort and config temporal conventions differ")
    if manifest["horizon"] != config["planning"]["horizon"]:
        raise ValueError("Saved cohort and config horizons differ")
    starts = manifest["splits"][split]["raw_start_rows"]
    limit = config["evaluation"]["query_limit"]
    if limit is not None:
        starts = starts[:limit]
    offsets, ids = query["episode_metadata"]["offsets"], query["episode_metadata"]["ids"]
    result = {}
    for start in starts:
        position = bisect_right(offsets, start)-1
        if position < 0 or position >= len(ids):
            raise ValueError("Cohort start is outside the query source")
        local = start-offsets[position]
        current = start+(manifest["history"]-1)*manifest["frameskip"]
        goal = current+manifest["horizon"]*manifest["frameskip"]
        if goal >= offsets[position]+query["episode_metadata"]["lengths"][position]:
            raise ValueError("Cohort goal crosses an episode boundary")
        case_id = f"{query['regime']}:{query['collection_manifest'][:16]}:{ids[position]}:{local}"
        if case_id in result:
            raise ValueError("Duplicate query in the saved cohort")
        result[case_id] = {"start_raw_row": start, "current_raw_row": current, "goal_raw_row": goal,
                           "episode_position": position, "episode_start_raw_row": offsets[position],
                           "episode_id": ids[position], "start_in_episode": local}
    return result


def row_key(row):
    return row["iterations"], row["seed"], row["condition"]


def case_id(row, stream):
    if stream == "records":
        return row["case"]["case_id"]
    return row["query"]["transition_id"] if stream == "plans" else row["case_id"]


def validate_case(row, expected):
    case, query = row["case"], row["case"]["query"]
    for key in ("start_raw_row", "current_raw_row", "goal_raw_row", "episode_position", "episode_start_raw_row"):
        if case[key] != expected[key]:
            raise ValueError(f"Saved case differs from the cohort: {case['case_id']} / {key}")
    if (query["transition_id"] != case["case_id"] or query["episode_id"] != expected["episode_id"]
            or query["start_in_episode"] != expected["start_in_episode"]):
        raise ValueError(f"Saved query differs from the cohort: {case['case_id']}")


def reconstruction(receipt, settings):
    for key, threshold in (("state_max_abs_error", settings["state_atol"]),
                           ("rgb_mae_0_255", settings["rgb_mae_atol"])):
        number = receipt[key]
        if not math.isfinite(number) or not 0 <= number <= threshold:
            raise ValueError(f"Saved physical reconstruction failed: {key}={number}")


def scan(paths, config, expected, expected_keys):
    records, plans, calls = defaultdict(list), defaultdict(Counter), defaultdict(lambda: defaultdict(list))
    receipts = {}
    for name in STREAMS:
        receipt = receipts[name] = {}
        for _, row in json_lines(paths[name], receipt):
            identity = case_id(row, name)
            if identity not in expected:
                raise ValueError(f"{name} contains a query outside the saved cohort: {identity}")
            if name == "records":
                validate_case(row, expected[identity])
                if row["kind"] not in ("recorded-control", "planned"):
                    raise ValueError("Unknown physical outcome kind")
                if row["kind"] == "planned" and row_key(row) not in expected_keys:
                    raise ValueError("Outcome outside the configured budget/seed/cell matrix")
                records[identity].append(row)
            else:
                key = row_key(row)
                if key not in expected_keys:
                    raise ValueError(f"{name} contains a budget/seed/cell outside the saved config")
                if name == "plans":
                    plans[identity][key] += 1
                else:
                    # Keep only the small execution receipt, not large diagnostics.
                    calls[identity][key].append({k: row[k] for k in (
                        "replan", "raw_steps_before", "first_call_snapshot", "executed_raw_actions", "query")})
    return records, plans, calls, receipts


def complete_case(identity, rows, plans, calls, expected_keys, config):
    controls = [row for row in rows if row["kind"] == "recorded-control"]
    outcomes = [row for row in rows if row["kind"] == "planned"]
    keys = Counter(row_key(row) for row in outcomes)
    if len(controls) > 1 or any(n > 1 for n in keys.values()) or any(n > 1 for n in plans.values()):
        raise ValueError(f"Duplicate control, outcome or plan for {identity}")
    if not controls or set(keys) != expected_keys:
        return "incomplete paired outcome matrix"
    control = controls[0]
    settings = config["core"]
    reconstruction(control["prefix_reconstruction"], settings)
    reconstruction(control["recorded_future_reconstruction"], settings)
    required_plans = set()
    for row in outcomes:
        key = row_key(row)
        if row["case"] != control["case"] or row["initial_metrics"] != control["initial_metrics"]:
            raise ValueError(f"Treatments are not paired to the same initial state/goal: {identity}")
        number = row["planning_calls"]
        if type(number) is not int or number < 0:
            raise ValueError("Invalid planning-call count")
        saved = calls.get(key, [])
        if len(saved) != number:
            return "incomplete executed-call receipts"
        if number:
            required_plans.add(key)
            if key not in plans:
                return "missing first-call plan"
            reconstruction(row["prefix_reconstruction"], settings)
        elif not (settings["control_budget"] == 50 and row["initial_success"] and row["ever_success"]
                  and row.get("skipped_initial_success")):
            raise ValueError("Zero-call outcome without an initially successful 50-control case")
        actions = []
        for replan, call in enumerate(saved):
            query_id = identity if replan == 0 else f"{identity}/real-replan-{replan}"
            if (call["replan"] != replan or call["raw_steps_before"] != len(actions)
                    or call["first_call_snapshot"] != (replan == 0)
                    or call["query"]["transition_id"] != query_id):
                raise ValueError(f"Executed-call sequence differs from the outcome: {identity} / {key}")
            actions.extend(call["executed_raw_actions"])
        if actions != row["executed_raw_actions"] or len(actions) != row["raw_steps"]:
            raise ValueError(f"Executed actions differ from the outcome: {identity} / {key}")
        if not 0 <= row["raw_steps"] <= settings["control_budget"]:
            raise ValueError("Execution exceeds its configured control budget")
    if set(plans) != required_plans or set(calls)-required_plans:
        raise ValueError(f"Unexpected plan/call receipts for {identity}")
    return None


def paired_prefix(records, plans, calls, expected, expected_keys, config, status):
    ids = list(expected)
    committed = status.get("completed_queries", len(ids))
    if type(committed) is not int or not 0 <= committed <= len(ids):
        raise ValueError("Invalid completed-query count in status.json")
    if status.get("requested_queries", len(ids)) != len(ids):
        raise ValueError("Status and saved cohort query counts differ")
    retained, stop_reason = [], None
    for identity in ids[:committed]:
        reason = complete_case(identity, records[identity], plans[identity], calls[identity], expected_keys, config)
        if reason is not None:
            stop_reason = f"{identity}: {reason}"
            break
        retained.append(identity)
    if not retained:
        raise ValueError("No complete committed paired query is recoverable")
    return retained, stop_reason or "end of committed query prefix"


def recover(run_dir, output=None, split=None, inspect_only=False):
    # Import only config/reporting helpers; this path never loads resources.
    from lewm_tta.core.config import cells
    from lewm_tta.core.reporting import summarize, write_tables

    run_dir = Path(run_dir).resolve()
    config_path = run_dir/"config.json"
    config = read_json(config_path)
    split = split or run_dir.name
    if split not in ("validation", "test"):
        raise ValueError("For a custom run name, provide --split validation or test")
    if (run_dir/"report.json").exists():
        raise FileExistsError("A scientific report already exists; use normal core analysis/resume")
    manifest_path = Path(config["experiment_dir"])/"manifest.json"
    query_path = (ROOT/config["query_dataset"]["manifest"]).resolve()
    paths = {name: run_dir/(name+".jsonl.gz") for name in STREAMS}
    inputs = {**paths, "config": config_path, "status": run_dir/"status.json",
              "manifest": manifest_path, "query_manifest": query_path, "checks": run_dir/"acceptance-checks.json"}
    assert_idle(inputs.values())
    before = source_receipts(inputs)
    manifest, query = checked_manifest(manifest_path), checked_manifest(query_path)
    status, checks = read_json(inputs["status"]), read_json(inputs["checks"])
    if not checks or any(not group or any(not value for value in group.values()) for group in checks.values()):
        raise ValueError("Saved acceptance checks are missing or failed")
    expected = expected_cases(config, manifest, query, split)
    labels = list(cells(config).values())
    expected_keys = {(j, seed, label) for j in config["core"]["iterations"]
                     for seed in config["p1"]["evaluation"]["seeds"] for label in labels}
    records, plans, calls, receipts = scan(paths, config, expected, expected_keys)
    retained, reason = paired_prefix(records, plans, calls, expected, expected_keys, config, status)
    selected = set(retained)
    rows = [row for identity in retained for row in records[identity]]
    summary = summarize(rows, config)
    if source_receipts(inputs) != before:
        raise ValueError("Source artifacts changed during recovery; leave the writer stopped and retry")
    audit = {"completed_queries": len(retained), "requested_queries": len(expected),
             "remaining_queries": len(expected)-len(retained), "retained_case_ids": retained,
             "selection": "Longest complete paired prefix, capped at status.completed_queries when present",
             "stop_reason": reason, "streams": receipts,
             "discarded_complete_rows": {name: receipts[name]["rows"]-(len(rows) if name == "records" else
                 sum(sum(plans[i].values()) if name == "plans" else sum(len(v) for v in calls[i].values()) for i in retained))
                 for name in STREAMS}}
    if inspect_only:
        print(json.dumps(audit, indent=2))
        return audit
    output = Path(output).resolve() if output else run_dir/"recovered-partial"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Recovery output already exists; preserve it or choose --output: {output}")
    # Publish all outputs together. On failure, original data remain untouched.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p3-recovery-", dir=output.parent))
    try:
        report = {"stage": "P3-core-recovered-partial", "completed": False, "budget_exhausted": False,
                  "split": split, "config": config, "code": None, "model": None, "runtime": None,
                  "cells": labels, "control_budget": config["core"]["control_budget"],
                  "completed_queries": len(retained), "requested_queries": len(expected),
                  "full_cohort_queries": manifest["splits"][split]["queries"], "limited_run": True,
                  "manifest_fingerprint": manifest["fingerprint"], "query_source_fingerprint": query["fingerprint"],
                  "p1_prepared_fingerprint": manifest["p1_prepared_fingerprint"], "checks": checks,
                  "query_manifest": str(query_path), "query_manifest_sha256": before["query_manifest"]["sha256"],
                  "evaluation_seconds": None, "resource_loading_and_checks_seconds": None,
                  "rough_full_cohort_hours": None,
                  "accounting": status.get("accounting"),
                  "accounting_scope": ("Saved committed-prefix counters; excludes unfinished query/tail work"
                      if status.get("status") == "running" and status.get("completed_queries") == len(retained)
                      else "Unverified status counters; may include work outside the recovered prefix"),
                  "protocol": "Recovered executed P3 core outcomes using the saved config and complete paired queries. "
                      "First-call iteration snapshots share search; later real replans have their own fixed-J search. "
                      "This selected prefix is exploratory and cannot create a validation lock or resume receipt.",
                  "summary": summary,
                  "recovery": {"created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(run_dir),
                      "source_files": before, "audit": audit,
                      "provenance_limit": "The interrupted legacy run did not save startup source/model/runtime identities. "
                          "They cannot be reconstructed from outcome rows. Current code is recorded only as reporting provenance.",
                      "selection_limit": "Queries follow the saved cohort order; missing tail episodes/windows may change rates. "
                          "Bootstrap intervals describe only the retained queries, not the full reserved cohort.",
                      "reporting_code": {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__).resolve(),
                          ROOT/"codes/lewm_tta/core/reporting.py", ROOT/"codes/lewm_tta/core/config.py",
                          ROOT/"codes/lewm_tta/outcomes/reporting.py", ROOT/"codes/lewm_tta/interleave/config.py"]}}}
        for name in STREAMS:
            count = 0
            receipt = {}
            target = temporary/(name+".jsonl.gz")
            with gzip.open(target, "wb") as stream:
                for line, row in json_lines(paths[name], receipt):
                    if case_id(row, name) in selected:
                        stream.write(line+b"\n")
                        count += 1
            if receipt != receipts[name]:
                raise ValueError(f"{name} changed during recovery")
            report.update({name+"_file": str(output/target.name), name+"_sha256": digest(target),
                           {"records": "record_count", "plans": "plan_count", "calls": "call_count"}[name]: count})
        write_json(temporary/"config.json", config)
        write_json(temporary/"manifest.json", manifest)
        write_json(temporary/"acceptance-checks.json", checks)
        write_json(temporary/"recovery-audit.json", audit)
        write_json(temporary/"report.json", report)
        write_tables(temporary, report)
        with (temporary/"summary.md").open("a") as stream:
            stream.write("\nRecovery notes: "+report["recovery"]["provenance_limit"]+"\n\n"
                         +report["recovery"]["selection_limit"]+"\n\n"
                         "Original wall time, loading time and GPU peak memory are unavailable. "
                         "Per-outcome saved planning latency remains available.\n")
        if source_receipts(inputs) != before:
            raise ValueError("Source artifacts changed before publication; no recovered output published")
        assert_idle(inputs.values())
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Recovery output appeared during processing: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Recovered {len(retained)}/{len(expected)} paired queries: {output/'summary.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Interrupted runs/validation or runs/test directory")
    parser.add_argument("--output", type=Path, help="New output directory (default: RUN_DIR/recovered-partial)")
    parser.add_argument("--split", choices=("validation", "test"), help="Required for custom run names")
    parser.add_argument("--inspect", action="store_true", help="Validate and print the recovery audit without writing files")
    args = parser.parse_args()
    try:
        recover(args.run_dir, args.output, args.split, args.inspect)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Recovery refused: {error}\n")


if __name__ == "__main__":
    main()
