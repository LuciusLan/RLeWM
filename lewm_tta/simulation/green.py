"""Dataset-independent reset cohorts and validation locks for fixed-target MPC.

This module reserves cases without importing/stepping a simulator or a model.
Task geometry/rendering belongs to the environment adapter.
"""

from copy import deepcopy
from itertools import product
from pathlib import Path

from ..runtime import file_hash, object_hash, read_json, write_json
from ..rollout.resources import checked_json, source_metadata


def enabled(config):
    return config.get("simulation", {}).get("task", "future-goal") == "green-target"


def manifest_value(config):
    source, prepared = source_metadata(config)
    settings = config["simulation"]
    history, frameskip = (source["data_config"][key] for key in ("history", "frameskip"))
    splits = {}
    counter = 0
    for split, count in (("validation", settings["episodes"]), ("test", settings["test_episodes"])):
        rows = []
        for index in range(count):
            case_id = f"green-{settings['backend']}:{settings['cohort_seed']}:{split}:{index}"
            reset_seed = int(object_hash([case_id, "reset"])[:8], 16)
            episode_id = 2_000_000_000+counter
            warmup = (history-1)*frameskip
            rows.append({"case_id": case_id, "episode_reset_seed": reset_seed,
                "start_raw_row": 0, "current_raw_row": warmup, "goal_raw_row": None,
                "warmup_raw_actions": [[0., 0.] for _ in range(warmup)],
                "query": {"transition_id": case_id, "source_episode_position": -episode_id-1,
                          "episode_id": episode_id, "start_in_episode": warmup,
                          "raw_observation_rows": [i*frameskip for i in range(history)],
                          "source": f"simulator:green-{settings['backend']}:{settings['cohort_seed']}",
                          "dynamics_regime_id": "green-"+object_hash(settings["shift"])[:16]}})
            counter += 1
        splits[split] = rows
    seeds = [case["episode_reset_seed"] for rows in splits.values() for case in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Green reset-seed collision; choose another cohort_seed")
    value = {"version": 3, "stage": "simulator-evaluation", "task": "green-target",
             "history": history, "frameskip": frameskip, "horizon": config["planning"]["horizon"],
             "splits": splits, "selection": {key: settings[key] for key in ("episodes", "test_episodes", "cohort_seed")},
             "target": {key: settings[key] for key in ("backend", "image_size", "goal_agent_position", "success_coverage", "shift")},
             "p1_manifest_fingerprint": source["fingerprint"], "p1_prepared_fingerprint": prepared["fingerprint"],
             "selection_rule": "all prespecified native resets retained, including initially successful/offscreen cases; no reference-trajectory or evaluated-outcome filtering",
             "warmup": "fixed zero-relative controls to form actual H3 history; excluded from control budget and score",
             "goal_rule": "fixed native green marker; separate RGB goal with block aligned to marker and canonical pusher",
             "memory_policy": "original external memory only; simulator validation/test transitions never enter support"}
    value["fingerprint"] = object_hash(value)
    return value


def split_name(config):
    return config["simulation"].get("split", "validation")


def lock_validation(path, output=None):
    from experiments.provenance import backend_identity
    path = Path(path).resolve()
    report = read_json(path)
    config = report["config"]
    if (not enabled(config) or report.get("stage") != "simulator-evaluation" or report.get("task") != "green-target"
            or report.get("split") != "validation" or report.get("mode") != "mpc"
            or not report.get("completed") or report.get("limited_run") or report.get("budget_exhausted")
            or report.get("code") != backend_identity("sim")):
        raise ValueError("Lock requires complete, unrestricted green validation under its recorded source")
    from .cohorts import load_manifest
    from .reporting import summarize
    from ..rollout.record_io import read_saved_records
    manifest = load_manifest(config)
    expected = manifest["splits"]["validation"]
    if (report["manifest_fingerprint"] != manifest["fingerprint"]
            or report["case_ids"] != [case["case_id"] for case in expected]
            or report["summary"]["cases"] != len(expected)):
        raise ValueError("Validation report does not cover the complete reserved cohort")
    rows = read_saved_records(report)
    wanted = set(product([case["case_id"] for case in expected], config["simulation"]["seeds"],
                         config["adaptation"]["methods"], config["proposals"]["sources"]))
    actual = {(row["case_id"], row["seed"], row["method"], row["proposal_source"]) for row in rows}
    queries = {case["case_id"]: case["query"] for case in expected}
    if (actual != wanted or len(rows) != len(wanted) or report["record_count"] != len(rows)
            or any(row["query"] != queries[row["case_id"]] for row in rows)):
        raise ValueError("Green validation records do not cover the exact reserved six-cell matrix")
    if summarize(rows, config) != report["summary"]:
        raise ValueError("Validation records do not reproduce their summary")
    value = {"stage": "simulator-evaluation", "task": "green-target", "config": config,
             "code": report["code"], "manifest_fingerprint": manifest["fingerprint"],
             "validation_report": str(path), "report_sha256": file_hash(path)}
    value["fingerprint"] = object_hash(value)
    target = Path(output) if output else Path(config["experiment_dir"])/"green-validation-lock.json"
    if target.exists() and checked_json(target) != value:
        raise ValueError("A different validation lock exists; use another output path")
    write_json(target, value)
    print(f"Green validation locked: {target}")
    return value


def locked_test_config(path):
    from experiments.provenance import backend_identity
    lock = checked_json(Path(path))
    report_path = Path(lock["validation_report"])
    report = read_json(report_path)
    if (lock.get("stage") != "simulator-evaluation" or lock.get("task") != "green-target"
            or report.get("task") != "green-target" or not enabled(lock["config"])
            or lock["code"] != backend_identity("sim") or lock["report_sha256"] != file_hash(report_path)
            or lock["config"] != report["config"] or lock["code"] != report["code"]
            or lock["manifest_fingerprint"] != report["manifest_fingerprint"]
            or report.get("mode") != "mpc" or report.get("budget_exhausted")
            or not report.get("completed") or report.get("limited_run") or report.get("split") != "validation"):
        raise ValueError("Green validation lock/report/source changed")
    config = deepcopy(lock["config"])
    config["simulation"]["split"] = "test"
    config["evaluation"]["query_limit"] = None
    return config, {"path": str(Path(path).resolve()), "fingerprint": lock["fingerprint"],
                    "validation_report_sha256": lock["report_sha256"]}
