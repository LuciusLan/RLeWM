"""Schedule comparisons with explicit per-horizon optimization budgets."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from ..reporting import episode_statistics
from .config import effective_p1, schedule_names, scope_names, settings_for_schedule, updates_at, update_budget


def _validate_trace(row: dict, config: dict, learning_rate: float | None = None) -> None:
    schedule = config["schedules"][row["schedule"]]
    if learning_rate is None:
        learning_rate = settings_for_schedule(effective_p1(config)["adaptation"], schedule)["learning_rate"]
    trace, horizon = row["adaptation"], max(config["horizons"])
    if trace["schedule"] != schedule or len(trace["positions"]) != horizon:
        raise ValueError("Saved trajectory schedule differs from its config")
    rounds, seen, presentations, last_ids = trace["rounds"], set(), 0, set()

    def check_rate(entry):
        # Historical traces without an override predate explicit rate metadata.
        if "learning_rate" in schedule or "learning_rate" in entry:
            saved = entry.get("learning_rate")
            if (isinstance(saved, bool) or not isinstance(saved, (int, float))
                    or not np.isclose(saved, learning_rate, rtol=1e-12, atol=0)):
                raise ValueError("Saved learning rate differs from its schedule/config")

    check_rate(trace)
    if "actual_learning_rate_sum" in trace and not np.isclose(
            trace["actual_learning_rate_sum"], len(rounds)*learning_rate, rtol=1e-12, atol=0):
        raise ValueError("Invalid summed learning-rate accounting")
    cursor = optimizer_steps = 0
    reset_count = 1 if schedule["persistence"] == "trajectory" else horizon
    if trace["reset_count"] != reset_count or trace["actual_updates"] != len(rounds):
        raise ValueError("Invalid optimizer reset/update count")
    requested = 0 if row["method"] == "frozen" else update_budget(schedule, horizon)
    if trace["requested_updates"] != requested:
        raise ValueError("Invalid requested trajectory update budget")
    for h, position in enumerate(trace["positions"]):
        check_rate(position)
        count = 0 if row["method"] == "frozen" else updates_at(schedule, h)
        actual = position["actual_updates"]
        if schedule["persistence"] == "position":
            optimizer_steps = 0
        if (position["position"] != h or position["requested_updates"] != count
                or not 0 <= actual <= count or (actual < count and not position["skip_reason"])
                or position["round_start"] != cursor or position["round_end"] != cursor+actual
                or position["optimizer_steps_before"] != optimizer_steps
                or position["optimizer_steps_after"] != optimizer_steps+actual
                or position["reset_before_position"] != (h == 0 or schedule["persistence"] == "position")):
            raise ValueError("Invalid per-position update/optimizer trace")
        if not np.isclose(position["query_mse_after"], row["mse_by_block"][h], rtol=1e-12, atol=1e-15):
            raise ValueError("Committed prediction error differs from the position trace")
        if h and schedule["persistence"] == "trajectory" and position["parameter_displacement_before"] != trace["positions"][h-1]["parameter_displacement_after"]:
            raise ValueError("Parameter anchor displacement changed between carried positions")
        local_support = None
        for local, update in enumerate(rounds[cursor:cursor+actual]):
            check_rate(update)
            ids = {item["transition_id"] for item in update["support"]}
            signature = [(item["transition_id"], item["weight"], item["channels"]) for item in update["support"]]
            if row["method"] != "fr-refresh" and local_support is not None and signature != local_support:
                raise ValueError("Fixed support changed within a position")
            local_support = signature
            if (update["position"] != h or update["update_at_position"] != local+1
                    or update["update"] != cursor+local+1 or update["optimizer_step"] != optimizer_steps+local+1
                    or len(ids) != len(update["support"]) or update["unique_support"] != len(ids)):
                raise ValueError("Invalid support/update trace")
            seen |= ids
            presentations += len(ids)
            if update["presentations_so_far"] != presentations or update["cumulative_unique_support"] != len(seen):
                raise ValueError("Invalid cumulative support accounting")
            expected_turnover = len(ids-last_ids)/len(ids) if last_ids else None
            if update["turnover_fraction"] != expected_turnover:
                raise ValueError("Invalid support turnover")
            last_ids = ids
        cursor += actual
        optimizer_steps += actual
    if (cursor != len(rounds) or trace["support_presentations"] != presentations
            or trace["extra_support_evaluation_presentations"] != presentations
            or trace["cumulative_unique_support"] != len(seen)):
        raise ValueError("Invalid total support/update accounting")


def _details(rows: list[dict], p1: dict, horizons: list[int], learning_rate: float) -> dict:
    episodes = [row["query"]["source_episode_position"] for row in rows]
    positions = {}
    for h in range(max(horizons)):
        entries = [row["adaptation"]["positions"][h] for row in rows]
        updates = [(row, step) for row in rows for step in row["adaptation"]["rounds"] if step["position"] == h]
        values = {"mean_updates": float(np.mean([entry["actual_updates"] for entry in entries])),
                  "learning_rate": learning_rate,
                  "mean_actual_learning_rate_sum": float(np.mean([entry["actual_updates"]*learning_rate for entry in entries])),
                  "mean_displacement_before": float(np.mean([entry["parameter_displacement_before"] for entry in entries])),
                  "mean_displacement_after": float(np.mean([entry["parameter_displacement_after"] for entry in entries])),
                  "query_delta_during_local_updates": episode_statistics(
                      [entry["query_mse_after"]-entry["query_mse_before"] for entry in entries], episodes, p1),
                  "rounds": {}}
        for local in sorted({step["update_at_position"] for _, step in updates}):
            selected = [(row, step) for row, step in updates if step["update_at_position"] == local]
            episode_ids = [row["query"]["source_episode_position"] for row, _ in selected]
            metrics = ("support_loss_before", "support_loss_after", "gradient_norm_before_clipping",
                       "prediction_correction_mse", "parameter_displacement_mse", "support_budget_shortfall")
            values["rounds"][str(local)] = {
                key: episode_statistics([step[key] for _, step in selected], episode_ids, p1) for key in metrics}
            values["rounds"][str(local)]["support_loss_increase_fraction"] = float(np.mean([
                step["support_loss_after"] > step["support_loss_before"] for _, step in selected]))
            for key in ("turnover_fraction", "within_position_turnover"):
                valid = [step[key] for _, step in selected if step[key] is not None]
                values["rounds"][str(local)][key] = float(np.mean(valid)) if valid else None
        positions[str(h+1)] = values
    timing_keys = rows[0]["adaptation"]["timing_seconds"].keys()
    return {"learning_rate": learning_rate,
            "mean_actual_learning_rate_sum": float(np.mean([row["adaptation"]["actual_updates"]*learning_rate for row in rows])),
            "mean_actual_updates": float(np.mean([row["adaptation"]["actual_updates"] for row in rows])),
            "mean_reset_count": float(np.mean([row["adaptation"]["reset_count"] for row in rows])),
            "per_position": positions,
            "mean_retrieval_calls": {key: float(np.mean([row["adaptation"]["retrieval_calls"][key] for row in rows]))
                                     for key in ("forward", "reverse", "state")},
            "mean_timing_seconds": {key: float(np.mean([row["adaptation"]["timing_seconds"][key] for row in rows]))
                                    for key in timing_keys},
            "latency_p95_seconds": float(np.quantile([row["total_seconds"] for row in rows], .95))}


def summarize_schedules(records: list[dict], config: dict, accurate_thresholds=None) -> dict:
    from .reporting import summarize_scopes

    names, scopes, p1 = schedule_names(config), scope_names(config), effective_p1(config)
    rates = {name: settings_for_schedule(p1["adaptation"], config["schedules"][name])["learning_rate"] for name in names}
    grouped, lookups = defaultdict(list), {name: {} for name in names}
    for row in records:
        name = row.get("schedule")
        if name not in names:
            raise ValueError("Unknown or missing schedule in saved records")
        key = (row["scope"], row["query"]["transition_id"], row["seed"], row["method"])
        if key in lookups[name]:
            raise ValueError("Duplicate schedule/scope/query/seed/method record")
        _validate_trace(row, config, rates[name])
        lookups[name][key] = row
        grouped[name].append(row)
    expected = set(lookups[names[0]])
    if not expected or any(set(lookups[name]) != expected for name in names):
        raise ValueError("Schedules must cover the same complete scope/query/seed/method matrix")
    for key in expected:
        first = lookups[names[0]][key]
        for name in names[1:]:
            other = lookups[name][key]
            if first["query"] != other["query"] or first["target_raw_rows"] != other["target_raw_rows"]:
                raise ValueError("Schedules use different query/target alignment")
            if key[-1] == "frozen" and first["mse_by_block"] != other["mse_by_block"]:
                raise ValueError("Frozen predictions differ across schedules")
    threshold, summaries = accurate_thresholds, {}
    for name in names:
        summaries[name] = summarize_scopes(grouped[name], config, threshold)
        if threshold is None:
            threshold = summaries[name]["accurate_thresholds"]
        for scope in scopes:
            for method in p1["adaptation"]["methods"]:
                rows = [row for row in grouped[name] if row["scope"] == scope and row["method"] == method]
                result = summaries[name]["scopes"][scope]["methods"][method]
                result["iteration"] = _details(rows, p1, config["horizons"], rates[name])
                episodes = [row["query"]["source_episode_position"] for row in rows]
                for h in config["horizons"]:
                    harms = [max(0., row["mse_by_block"][h-1]-lookups[name][
                        (scope, row["query"]["transition_id"], row["seed"], "frozen")]["mse_by_block"][h-1]) for row in rows]
                    result["horizons"][str(h)]["mean_harm_magnitude_over_all_queries"] = episode_statistics(harms, episodes, p1)
    comparisons = {}
    for right, left in combinations(names, 2):
        by_scope = {}
        for scope in scopes:
            by_scope[scope] = {}
            for method in p1["adaptation"]["methods"]:
                keys = sorted(key for key in expected if key[0] == scope and key[-1] == method)
                lhs, rhs = [lookups[left][key] for key in keys], [lookups[right][key] for key in keys]
                episodes = [row["query"]["source_episode_position"] for row in lhs]
                by_horizon = {}
                for h in config["horizons"]:
                    counts = [[sum(step["position"] < h for step in row["adaptation"]["rounds"]) for row in rows]
                              for rows in (lhs, rhs)]
                    presentations = [[sum(step["unique_support"] for step in row["adaptation"]["rounds"]
                                          if step["position"] < h) for row in rows] for rows in (lhs, rhs)]
                    lr_sums = [np.asarray(values)*rates[name] for values, name in zip(counts, (left, right))]
                    frozen = np.array([lookups[left][(scope, q, seed, "frozen")]["mse_by_block"][h-1]
                                       for _, q, seed, _ in keys])
                    a, b = (np.array([row["mse_by_block"][h-1] for row in rows]) for rows in (lhs, rhs))
                    by_horizon[str(h)] = {
                        "endpoint_delta": episode_statistics(a-b, episodes, p1),
                        "mean_path_delta": episode_statistics([np.mean(x["mse_by_block"][:h])-np.mean(y["mse_by_block"][:h])
                                                               for x, y in zip(lhs, rhs)], episodes, p1),
                        "harm_fraction_delta": episode_statistics((a-frozen > p1["evaluation"]["harm_tolerance"]).astype(float)
                            -(b-frozen > p1["evaluation"]["harm_tolerance"]).astype(float), episodes, p1),
                        "matched_actual_updates": counts[0] == counts[1],
                        "matched_support_presentations": presentations[0] == presentations[1],
                        "learning_rate_left_right": [rates[left], rates[right]],
                        "matched_per_update_learning_rate": rates[left] == rates[right],
                        "matched_learning_rate_sums": bool(np.allclose(lr_sums[0], lr_sums[1], rtol=1e-12, atol=0)),
                        "mean_learning_rate_sum_left_right": [float(values.mean()) for values in lr_sums],
                        "mean_updates_left_right": [float(np.mean(values)) for values in counts],
                        "mean_presentations_left_right": [float(np.mean(values)) for values in presentations]}
                by_scope[scope][method] = by_horizon
        comparisons[f"{left}-minus-{right}"] = by_scope
    return {"queries": summaries[names[0]]["queries"], "episodes": summaries[names[0]]["episodes"],
            "horizons": config["horizons"], "accurate_thresholds": threshold, "schedules": summaries,
            "cross_schedule": comparisons,
            "schedule_learning_rates": rates,
            "nominal_update_budgets": {name: {str(h): update_budget(config["schedules"][name], h)
                                                for h in config["horizons"]} for name in names},
            "nominal_learning_rate_sums": {name: {str(h): update_budget(config["schedules"][name], h)*rates[name]
                                                  for h in config["horizons"]} for name in names},
            "uncertainty": "paired episode bootstrap; windows and repeated seeds remain within episodes",
            "budget_note": "Counts and summed learning rates are cumulative through each scored horizon; Frozen uses zero updates. Equal counts do not match LR or placement; equal LR sums do not imply equal AdamW displacement."}


def format_schedule_summary(summary: dict, path) -> str:
    h = str(max(summary["horizons"]))
    lines = [f"P2 follow-up | {summary['queries']} queries / {summary['episodes']} episodes",
             f"Scope         Schedule             LR        Method        H={h} MSE     Harm    Updates  Presentations"]
    for name, schedule in summary["schedules"].items():
        for scope, scoped in schedule["scopes"].items():
            for method, row in scoped["methods"].items():
                lines.append(f"{scope:<14}{name:<21}{summary['schedule_learning_rates'][name]:<10.3g}{method:<14}{row['horizons'][h]['endpoint_mse']['mean']:<12.7g}"
                             f"{row['horizons'][h]['harm_fraction']:>7.1%}  {row['iteration']['mean_actual_updates']:>7.2f}"
                             f"  {row['mean_support_presentations']:>13.1f}")
    lines += [f"Report: {path}", "Cross-schedule comparisons mark matched updates, presentations, learning rates and summed LR budgets at each horizon."]
    return "\n".join(lines)+"\n"
