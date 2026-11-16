"""Checksummed saved plans and a common fresh physical problem across budgets."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..collection.manifest import checked
from ..budget_receipts import verify_budget_capture
from ..rollout.record_io import read_saved_records
from ..runtime import file_hash, object_hash, read_json

STAGE = "P3-executed-plan-outcomes"
SOURCE_STAGES = {"P3a-offline-cem": "p3a", "P3b-offline-proposals": "p3b",
                 "P3c-offline-factorial": "p3c", "P3-temporal-cem": "p3_temporal",
                 "P3-interleaved-cem": "p3_interleave"}


def condition(row):
    if "arm" in row:
        return row["arm"]
    if "mode" in row:
        return row["mode"]
    return row["method"]+"/"+row["proposal_source"] if "proposal_source" in row else row["method"]


def raw_plan(row, config, normalization):
    horizon, skip = config["planning"]["horizon"], config["p1"]["data"]["frameskip"]
    width = len(config["planning"]["raw_action_low"])
    normalized = np.asarray(row["selected_actions_normalized"], dtype=np.float64)
    raw = np.asarray(row["selected_raw_actions"], dtype=np.float64)
    if normalized.shape != (horizon, skip*width) or raw.size != horizon*skip*width:
        raise ValueError("Saved plan horizon/action width differs from its source configuration")
    raw = raw.reshape(horizon*skip, width)
    expected = normalized.reshape(horizon*skip, width)*np.asarray(normalization["std"])+np.asarray(normalization["mean"])
    if not np.isfinite(raw).all() or not np.allclose(raw, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("Saved raw/normalized actions disagree")
    low, high = (np.asarray(config["planning"][key]) for key in ("raw_action_low", "raw_action_high"))
    if (raw < low-1e-6).any() or (raw > high+1e-6).any():
        raise ValueError("Saved planned controls violate the declared raw bounds")
    return raw.astype(np.float32)


def family_config(config):
    value = deepcopy(config)
    value.pop("experiment_dir", None)
    value.pop("budget_study", None)
    value["planning"].pop("iterations")
    value.pop("reuse", None)  # Imported cells retain explicitly labeled timings.
    if "interleave" in value:
        value["interleave"].pop("checkpoints")
    return value


def load_sources(paths, conditions=None, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("A physical replay query limit must be a positive integer")
    from experiments.provenance import archive_entry, backend_identity
    from ..collection.queries import query_source
    from ..rollout.resources import source_metadata
    sources, plans, seen_budgets = [], {}, set()
    first_config = first_query = first_code = first_stage = first_model = None
    for filename in paths:
        path = Path(filename).resolve()
        report = read_json(path)
        backend = SOURCE_STAGES.get(report.get("stage"))
        if not report.get("completed") or backend is None or report.get("split") not in ("validation", "test"):
            raise ValueError("Require completed saved-plan P3 validation/test reports")
        if not report.get("code") or (report["code"] != backend_identity(backend)
                and archive_entry(backend, report["code"]) is None):
            raise ValueError("Source report has no current or verified archived implementation")
        config = report["config"]
        verify_budget_capture(report)
        if sources and bool(report.get("budget_capture")) != bool(sources[0].get("budget_capture")):
            raise ValueError("Compare one capture protocol at a time; do not mix independent and shared-prefix timings")
        if sources and report.get("budget_capture") and report["budget_capture"] != sources[0]["budget_capture"]:
            raise ValueError("Single-pass budget curves must come from the same shared search receipt")
        if not config.get("query_dataset"):
            raise ValueError("Physical P3 outcomes require fresh queries with original reset receipts; incomplete source-state resets are not accepted")
        memory, prepared = source_metadata(config)
        query = query_source(config, memory)
        manifest_path = Path(config["experiment_dir"])/"manifest.json"
        cohort = checked(manifest_path)
        if (report["p1_prepared_fingerprint"] != prepared["fingerprint"]
                or report["manifest_fingerprint"] != cohort["fingerprint"]
                or cohort.get("query_source_fingerprint") != query["fingerprint"]
                or report.get("query_source_fingerprint", query["fingerprint"]) != query["fingerprint"]):
            raise ValueError("Prepared source identity differs from saved planning")
        budget = config["planning"]["iterations"]
        if budget in seen_budgets:
            raise ValueError("Use one complete paired source report per CEM iteration budget")
        seen_budgets.add(budget)
        if first_config is None:
            first_config, first_query, first_code, first_stage = config, query, report["code"], report["stage"]
            first_model = report["model"]
            first_split = report["split"]
        elif (family_config(config) != family_config(first_config) or query["fingerprint"] != first_query["fingerprint"]
                or report["code"] != first_code or report["stage"] != first_stage or report["split"] != first_split):
            raise ValueError("Budget reports must share scientific settings, implementation, fresh queries and split; only J and inactive checkpoints may differ")
        if report["model"] != first_model:
            raise ValueError("Budget reports use different checkpoint/native model code or model runtime packages")
        rows = read_saved_records(report)
        if report.get("cohort") != cohort["splits"][report["split"]]:
            raise ValueError("Saved report and prepared cohort disagree")
        allowed_starts = cohort["splits"][report["split"]]["raw_start_rows"]
        query_limit = config["evaluation"]["query_limit"]
        allowed_starts = set(allowed_starts if query_limit is None else allowed_starts[:query_limit])
        selected_starts = set()
        source_index = len(sources)
        for row in rows:
            label = condition(row)
            if conditions and label not in conditions:
                continue
            case_id, seed = row["query"]["transition_id"], row["seed"]
            key = (budget, case_id, seed, label)
            if key in plans or not row.get("checks") or not all(row["checks"].values()):
                raise ValueError("Duplicate or failed saved planning row")
            episode_id = row["query"]["episode_id"]
            positions = query["episode_metadata"]["ids"]
            if episode_id not in positions:
                raise ValueError("Saved query episode is absent from its fresh source")
            position = positions.index(episode_id)
            if position not in query["episodes"][report["split"]]:
                raise ValueError("Saved query belongs to another source split")
            start = query["episode_metadata"]["offsets"][position]+row["query"]["start_in_episode"]
            skip, history = config["p1"]["data"]["frameskip"], config["p1"]["data"]["history"]
            current = start+(history-1)*skip
            goal = current+config["planning"]["horizon"]*skip
            if (row.get("current_raw_row", current) != current or row["goal_raw_row"] != goal
                    or goal >= query["episode_metadata"]["offsets"][position]+query["episode_metadata"]["lengths"][position]
                    or start not in allowed_starts
                    or start not in query["evaluation_windows"][report["split"]]["raw_start_rows"]):
                raise ValueError("Saved query/current/goal does not match the prespecified fresh window")
            selected_starts.add(start)
            raw_plan(row, config, prepared["normalization"])
            plans[key] = {"row": row, "source_index": source_index, "case": {
                "case_id": case_id, "query": row["query"], "episode_position": position,
                "episode_start_raw_row": query["episode_metadata"]["offsets"][position],
                "start_raw_row": start, "current_raw_row": current, "goal_raw_row": goal}}
        if selected_starts != allowed_starts:
            raise ValueError("Source records do not cover the complete selected planning cohort")
        sources.append({"report": str(path), "report_sha256": file_hash(path), "stage": report["stage"],
            "cohort_manifest": str(manifest_path), "cohort_manifest_sha256": file_hash(manifest_path),
            "budget_capture": report.get("budget_capture"),
            "split": report["split"], "iterations": budget, "config": config, "code": report["code"],
            "records_file": report["records_file"], "records_sha256": report["records_sha256"],
            "limited_run": report.get("limited_run", False), "normalization": prepared["normalization"],
            "model": report["model"], "timing_scope": report.get("timing_scope", "source planning runner accounting")})
    if not sources or not plans:
        raise ValueError("No selected saved plans")
    budgets = sorted(seen_budgets)
    reference = {(q, seed, label) for j, q, seed, label in plans if j == budgets[0]}
    if any({(q, seed, label) for j, q, seed, label in plans if j == budget} != reference for budget in budgets):
        raise ValueError("Every iteration budget must contain the identical paired query/seed/condition matrix")
    labels = sorted({label for _, _, label in reference})
    ids = sorted({q for q, _, _ in reference})
    for label in labels:
        keys = {(q, seed) for q, seed, c in reference if c == label}
        seed_sets = [{seed for q, seed in keys if q == case} for case in ids]
        if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets):
            raise ValueError("Each selected condition needs complete seeds at every paired query")
    if conditions and set(conditions) != set(labels):
        raise ValueError("A requested condition is absent from source reports")
    if first_stage == "P3-interleaved-cem":
        union = sorted({j for source in sources for j in source["config"]["interleave"]["checkpoints"]})
        for source in sources:
            if source["config"]["interleave"]["checkpoints"] != [j for j in union if j < source["iterations"]]:
                raise ValueError("Interleave checkpoint schedules differ beyond trimming checkpoints after a stopped search")
    if limit is not None:
        wanted = set(ids[:limit])
        plans = {key: value for key, value in plans.items() if key[1] in wanted}
        ids = ids[:limit]
    return {"sources": sources, "plans": plans, "query": first_query, "config": first_config,
            "budgets": budgets, "conditions": labels, "query_ids": ids, "split": first_split,
            "limited_run": limit is not None or any(source["limited_run"] for source in sources),
            "source_fingerprint": object_hash([{key: value for key, value in source.items() if key != "config"} for source in sources])}
