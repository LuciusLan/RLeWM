"""Pin a paired episode subset of completed offline validation, without inference."""

from pathlib import Path
import numpy as np

from ..runtime import file_hash, object_hash, read_json, write_json
from ..rollout.record_io import open_records
from ..rollout.resources import checked_json, source_metadata


def source_report(config):
    path = Path(config["simulation"]["source_report"])
    report = read_json(path)
    if (not report.get("completed") or report.get("split") != "validation"
            or report.get("stage") != "P3c-offline-factorial" or report.get("limited_run")):
        raise ValueError("The simulator pilot requires a completed full-cohort six-cell validation source")
    source, prepared = source_metadata(config)
    original = Path(source["source"]["path"]).stat()
    if original.st_size != source["source"]["bytes"] or original.st_mtime_ns != source["source"]["mtime_ns"]:
        raise ValueError("Original dataset size/mtime differs from the prepared P1 source")
    if report["p1_prepared_fingerprint"] != prepared["fingerprint"]:
        raise ValueError("Offline source and simulator use different prepared data/model assets")
    records = Path(report["records_file"])
    if file_hash(records) != report["records_sha256"]:
        raise ValueError("Saved offline plans changed")
    return report, source, prepared


def iter_rows(report):
    import json
    count = 0
    with open_records(Path(report["records_file"])) as stream:
        for line in stream:
            if line.strip():
                count += 1
                yield json.loads(line)
    if count != report["record_count"]:
        raise ValueError("Offline record count differs from its completed report")


def manifest_value(config):
    from .green import enabled, manifest_value as green_manifest
    if enabled(config):
        return green_manifest(config)
    if config.get("query_dataset"):
        return fresh_manifest_value(config)
    report, source, prepared = source_report(config)
    cases, episodes = {}, {}
    history, frameskip = (source["data_config"][key] for key in ("history", "frameskip"))
    for row in iter_rows(report):
        query = row["query"]
        case_id = query["transition_id"]
        value = {"case_id": case_id, "query": query, "start_raw_row": query["raw_observation_rows"][0],
                 "current_raw_row": row["current_raw_row"], "goal_raw_row": row["goal_raw_row"]}
        if case_id in cases and cases[case_id] != value:
            raise ValueError("Offline treatments disagree on case context or goal")
        if value["current_raw_row"] != value["start_raw_row"]+(history-1)*frameskip:
            raise ValueError("Source context alignment changed")
        if value["goal_raw_row"]-value["current_raw_row"] != config["planning"]["horizon"]*frameskip:
            raise ValueError("Source horizon differs from simulator horizon")
        episode = query["source_episode_position"]
        if episode not in source["episodes"]["validation"] or episode in source["episodes"]["memory"]:
            raise ValueError("Simulator case is not a memory-disjoint validation episode")
        cases[case_id] = value
        episodes.setdefault(episode, set()).add(case_id)
    count = config["simulation"]["episodes"]
    if len(episodes) < count:
        raise ValueError("Not enough distinct validation source episodes for this pilot")
    rng = np.random.default_rng(config["simulation"]["cohort_seed"])
    chosen = sorted(rng.choice(sorted(episodes), count, replace=False).tolist())
    selected = [cases[str(rng.choice(sorted(episodes[episode])))] for episode in chosen]
    value = {"version": 1, "stage": "simulator-evaluation", "split": "validation", "history": history,
             "frameskip": frameskip, "horizon": config["planning"]["horizon"],
             "selection": {key: config["simulation"][key] for key in ("episodes", "cohort_seed")},
             "source_report": config["simulation"]["source_report"],
             "source_report_sha256": file_hash(Path(config["simulation"]["source_report"])),
             "source_records_sha256": report["records_sha256"],
             "p1_manifest_fingerprint": source["fingerprint"], "p1_prepared_fingerprint": prepared["fingerprint"],
             "cases": selected, "selection_rule": "one saved q4 start per randomly chosen validation episode; outcome-independent",
             "provenance_limit": "previously inspected validation; source checkpoint pretraining overlap unknown"}
    value["fingerprint"] = object_hash(value)
    return value


def fresh_manifest_value(config):
    from ..collection.queries import query_source
    memory, prepared = source_metadata(config)
    source = query_source(config, memory)
    # A validation pilot only. Test cohorts remain reserved for a separately
    # locked evaluation after simulator protocol/compute choices are finalized.
    history, frameskip = (source["data_config"][key] for key in ("history", "frameskip"))
    rng = np.random.default_rng(config["simulation"]["cohort_seed"])
    episodes = source["episodes"]["validation"]
    count = config["simulation"]["episodes"]
    if count > len(episodes):
        raise ValueError("Not enough fresh validation episodes")
    chosen = sorted(rng.choice(episodes, count, replace=False).tolist())
    cases = []
    for episode in chosen:
        original = source["cases"][episode]
        offset = source["episode_metadata"]["offsets"][episode]
        length = source["episode_metadata"]["lengths"][episode]
        candidates = [start for start in source["evaluation_windows"]["validation"]["raw_start_rows"]
                      if offset <= start < offset+length]
        start = int(rng.choice(candidates))
        current = start+(history-1)*frameskip
        goal = current+config["planning"]["horizon"]*frameskip
        if goal >= offset+length:
            raise ValueError("Fresh goal crosses an episode boundary")
        case_id = f"{original['case_id']}:{start-offset}"
        metadata = {"transition_id": case_id, "source_episode_position": -original["episode_id"]-1,
                    "episode_id": original["episode_id"], "start_in_episode": start-offset,
                    "raw_observation_rows": [start+i*frameskip for i in range(history+1)],
                    "source": source["source"]["path"], "dynamics_regime_id": source["regime"]}
        cases.append({"case_id": case_id, "query": metadata, "start_raw_row": start,
                      "current_raw_row": current, "goal_raw_row": goal,
                      "episode_start_raw_row": offset, "episode_reset_seed": original["reset_seed"]})
    value = {"version": 2, "stage": "simulator-evaluation", "split": "validation", "history": history,
             "frameskip": frameskip, "horizon": config["planning"]["horizon"], "cases": cases,
             "selection": {key: config["simulation"][key] for key in ("episodes", "cohort_seed")},
             "p1_manifest_fingerprint": memory["fingerprint"], "p1_prepared_fingerprint": prepared["fingerprint"],
             "query_source_fingerprint": source["fingerprint"], "regime": source["regime"],
             "selection_rule": "one prespecified q4 start per fresh validation episode, independent of evaluated outcomes",
             "provenance_limit": "new simulator executions, exact released-state audit passed; unknown additional pretraining data unaudited"}
    value["fingerprint"] = object_hash(value)
    return value


def prepare(config):
    value = manifest_value(config)
    path = Path(config["experiment_dir"])/"manifest.json"
    if path.exists() and checked_json(path) != value:
        raise ValueError("Simulator cohort/source changed; use a new experiment directory")
    if not path.exists():
        write_json(path, value)
    if value.get("task") == "green-target":
        counts = {split: len(rows) for split, rows in value["splits"].items()}
        print(f"Green simulator reset suites: {counts}.\nManifest: {path}")
    else:
        print(f"Simulator validation: {len(value['cases'])} distinct episodes, one case per episode.\nManifest: {path}")
    return value


def load_manifest(config):
    value = checked_json(Path(config["experiment_dir"])/"manifest.json")
    if value != manifest_value(config):
        raise ValueError("Simulator source/config/cohort changed; prepare a compatible experiment")
    return value


def selected_cases(config, manifest):
    limit = config["evaluation"]["query_limit"]
    from .green import split_name
    cases = manifest["splits"][split_name(config)] if manifest.get("task") == "green-target" else manifest["cases"]
    return cases if limit is None else cases[:limit]


def saved_plans(config, manifest, cases, seeds):
    report, _, prepared = source_report(config)
    wanted = {case["case_id"] for case in cases}
    result = {}
    mean, std = (np.asarray(prepared["normalization"][key]) for key in ("mean", "std"))
    for row in iter_rows(report):
        case_id = row["query"]["transition_id"]
        if (case_id not in wanted or row["seed"] not in seeds
                or row["method"] not in config["adaptation"]["methods"]
                or row["proposal_source"] not in config["proposals"]["sources"]):
            continue
        key = (case_id, row["seed"], row["method"], row["proposal_source"])
        if key in result:
            raise ValueError("Duplicate saved source plan")
        raw = np.asarray(row["selected_raw_actions"], dtype=np.float32)
        normalized = np.asarray(row["selected_actions_normalized"], dtype=np.float32)
        expected = normalized.reshape(manifest["horizon"]*manifest["frameskip"], -1)*std+mean
        if raw.shape != expected.shape or not np.isfinite(raw).all() or not np.allclose(raw, expected, atol=1e-6, rtol=1e-6):
            raise ValueError("Saved raw and normalized action plans disagree")
        result[key] = row
    expected_count = len(cases)*len(seeds)*len(config["adaptation"]["methods"])*len(config["proposals"]["sources"])
    if len(result) != expected_count:
        raise ValueError("Source plans do not cover every selected case/seed/treatment; use saved source seeds")
    return result
