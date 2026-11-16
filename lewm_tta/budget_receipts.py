"""Read-only provenance for per-budget reports sharing one instrumented search."""

from pathlib import Path

from .runtime import file_hash, object_hash, read_json

STAGE = "P3-interleaved-budget-snapshots"


def verify_budget_capture(report):
    capture = report.get("budget_capture")
    if capture is None:
        return None
    if capture.get("mode") != "single-pass":
        raise ValueError("Unknown budget capture mode")
    path = Path(capture["master_report"])
    if file_hash(path) != capture["master_report_sha256"]:
        raise ValueError("Single-pass search receipt changed")
    master = read_json(path)
    if (not master.get("completed") or master.get("stage") != STAGE or master["code"] != report["code"]
            or master["iterations"] != capture["requested_iterations"]
            or max(master["iterations"]) != capture["max_iterations"]):
        raise ValueError("Single-pass receipt protocol/code mismatch")
    entry = next((entry for entry in master["reports"] if entry["iterations"] == report["config"]["planning"]["iterations"]), None)
    if (entry is None or entry["records_file"] != report["records_file"]
            or entry["records_sha256"] != report["records_sha256"] or entry["record_count"] != report["record_count"]
            or entry["config_fingerprint"] != object_hash(report["config"])
            or entry["manifest_fingerprint"] != report["manifest_fingerprint"]):
        raise ValueError("Budget report is not pinned by the single-pass search receipt")
    return master
