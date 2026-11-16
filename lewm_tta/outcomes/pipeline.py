"""Saved-plan physical evaluation, separate from planning and model updates."""

import gzip
import json
from pathlib import Path
import time

import numpy as np

from ..rollout.record_io import read_saved_records
from ..runtime import file_hash, object_hash, read_json, write_json
from .inputs import STAGE, load_sources, raw_plan
from .metrics import metric_contract
from .reporting import format_summary, summarize, write_tables

PROTOCOL = ("Independent fresh P3 queries; original reset plus full recorded past prefix; "
            "execute the saved selected plan for exactly H*frameskip controls toward the supplied future goal. "
            "No replanning, model adaptation, early success termination or query rejection during physical evaluation.")


def compute_record(row):
    result = {name: float(value) for name, value in row.get("counters", {}).items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)}
    for name in ("cem_candidate_evaluations", "cem_transition_predictions", "new_cem_candidate_evaluations"):
        if name in row:
            result[name] = float(row[name])
    adaptation = row.get("adaptation", {})
    for name in ("updates", "actual_updates", "requested_updates", "support_presentations", "support_positions",
                 "support_shortfall_positions", "forward_retrievals", "reverse_retrievals", "reset_count",
                 "sequence_evaluations", "transition_predictions", "reference_transition_predictions",
                 "candidate_trajectories", "candidate_unique_support_sum", "cumulative_unique_support"):
        if name in adaptation and isinstance(adaptation[name], (int, float)):
            result["adaptation_"+name] = float(adaptation[name])
    if any(not np.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Invalid source compute accounting")
    return result


def identity(row):
    return [row["kind"], row["case"]["case_id"], row.get("iterations"), row.get("seed"), row.get("condition")]


def evaluate(paths, output, settings, conditions=None, limit=None):
    from experiments.provenance import backend_identity
    from .replay import FreshReplay
    bundle = load_sources(paths, conditions, limit)
    output = Path(output).resolve()
    # An interrupted physical run stays inspectable. Never append mixed runs.
    output.mkdir(parents=True, exist_ok=False)
    records, started, replay = [], time.perf_counter(), None
    record_path = output/"records.jsonl.gz"
    try:
        replay = FreshReplay(bundle["query"], bundle["config"], settings)
        config = {"settings": settings, "source_fingerprint": bundle["source_fingerprint"],
                  "query_source_fingerprint": bundle["query"]["fingerprint"],
                  "budgets": bundle["budgets"], "conditions": bundle["conditions"],
                  "selected_query_ids": bundle["query_ids"]}
        write_json(output/"manifest.json", {**config, "fingerprint": object_hash(config)})
        with gzip.open(record_path, "xt", encoding="utf-8") as stream:
            def append(row):
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n")
                stream.flush()
            for index, query in enumerate(bundle["query_ids"]):
                selected = [(key, value) for key, value in sorted(bundle["plans"].items()) if key[1] == query]
                case = selected[0][1]["case"]
                control = replay.execute(case)
                append({"kind": "recorded-control", "case": case, **control})
                for (budget, _, seed, label), plan in selected:
                    source = bundle["sources"][plan["source_index"]]
                    row = plan["row"]
                    latency = float(row["primary_seconds_with_checks"])
                    if not np.isfinite(latency) or latency < 0:
                        raise ValueError("Missing/nonfinite source planning latency")
                    actions = raw_plan(row, source["config"], source["normalization"])
                    outcome = replay.execute(plan["case"], actions)
                    if outcome["initial_metrics"] != control["initial_metrics"]:
                        raise ValueError("Paired planned/control executions did not start from the same physical state/goal")
                    append({"kind": "planned", "iterations": budget, "condition": label, "seed": seed,
                        "case": plan["case"], "source_index": plan["source_index"],
                        "source_actions_sha256": object_hash(row["selected_raw_actions"]),
                        "planning_seconds": latency, "source_result_origin": row.get("result_origin", "computed"),
                        "compute": compute_record(row), **outcome})
                write_json(output/"status.json", {"status": "running", "queries_completed": index+1,
                    "queries": len(bundle["query_ids"]), "records": len(records)})
                print(f"Physical P3 outcomes {index+1}/{len(bundle['query_ids'])}: {query}", flush=True)
        report = {"completed": True, "stage": STAGE, "split": bundle["split"], "config": config,
            "code": backend_identity("p3_success"), "protocol": PROTOCOL,
            "metric_contract": metric_contract(replay.backend, settings["pusht_success"], settings["coverage_threshold"]),
            "environment": replay.description, "sources": bundle["sources"],
            "limited_run": bundle["limited_run"], "summary": summarize(records, settings),
            "records_file": str(record_path), "records_sha256": file_hash(record_path),
            "record_count": len(records), "record_identity_sha256": object_hash([identity(row) for row in records]),
            "evaluation_seconds": time.perf_counter()-started,
            "timing_scope": "Source planning latency: independent calls or instrumented single-pass prefixes as pinned in each source. Simulator replay/control verification is separate; loading, measurement overhead and diagnostics follow source receipts."}
        write_json(output/"report.json", report)
        write_tables(output, report)
        write_json(output/"status.json", {"status": "completed", "records": len(records)})
        print(format_summary(report), flush=True)
        print(f"Report: {output/'report.json'}", flush=True)
        return report
    except BaseException as error:
        write_json(output/"status.json", {"status": "failed", "records": len(records), "error": str(error)})
        raise
    finally:
        if replay is not None:
            replay.close()


def verified_report(path):
    report = read_json(path)
    if report.get("stage") != STAGE or not report.get("completed"):
        raise ValueError("Require a completed P3 physical outcome report")
    for source in report["sources"]:
        capture = source.get("budget_capture")
        if capture and file_hash(Path(capture["master_report"])) != capture["master_report_sha256"]:
            raise ValueError("Shared-prefix planning receipt changed")
        if file_hash(Path(source["report"])) != source["report_sha256"]:
            raise ValueError("Saved planning source report changed")
        if file_hash(Path(source["records_file"])) != source["records_sha256"]:
            raise ValueError("Saved planning source records changed")
        if file_hash(Path(source["cohort_manifest"])) != source["cohort_manifest_sha256"]:
            raise ValueError("Saved planning cohort changed")
    records = read_saved_records(report)
    if object_hash([identity(row) for row in records]) != report["record_identity_sha256"]:
        raise ValueError("Physical record pairing changed")
    if summarize(records, report["config"]["settings"]) != report["summary"]:
        raise ValueError("Physical records no longer reproduce the reported summary")
    return report


def analyze(path, output=None):
    report = verified_report(path)
    if output is not None:
        protected = {Path(path).resolve(), Path(report["records_file"]).resolve()}
        for source in report["sources"]:
            protected.update(Path(source[name]).resolve() for name in ("report", "records_file", "cohort_manifest"))
            if source.get("budget_capture"):
                protected.add(Path(source["budget_capture"]["master_report"]).resolve())
        if Path(output).resolve() in protected:
            raise ValueError("Preserve original reports, records and cohort receipts")
        write_json(Path(output), report["summary"])
    print(format_summary(report), flush=True)
    return report
