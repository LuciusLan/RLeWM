"""Paired endpoint/path error, horizon-specific harm, and episode uncertainty."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..reporting import episode_statistics
from .config import effective_p1


def summarize_methods(records: list[dict], config: dict, accurate_thresholds: dict | None = None) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty P2 evaluation")
    p1 = effective_p1(config)
    horizons = config["horizons"]
    methods = p1["adaptation"]["methods"]
    grouped, lookups = defaultdict(list), defaultdict(dict)
    for row in records:
        key = (row["query"]["transition_id"], row["seed"])
        if row["method"] not in methods or key in lookups[row["method"]]:
            raise ValueError("Unknown or duplicate P2 method/query/seed record")
        if len(row["mse_by_block"]) != max(horizons):
            raise ValueError(
                f"P2 horizon/record mismatch: expected {max(horizons)} per-block errors "
                f"for horizons={horizons}, got {len(row['mse_by_block'])}; "
                f"method={row['method']}, query={key[0]}, seed={key[1]}"
            )
        grouped[row["method"]].append(row)
        lookups[row["method"]][key] = row
    expected = set(lookups["frozen"])
    if not expected or any(set(lookups[method]) != expected for method in methods):
        raise ValueError("Paired P2 methods do not cover the same queries/seeds")
    query_ids = {query for query, _ in expected}
    seeds = set(p1["evaluation"]["seeds"])
    if expected != {(query, seed) for query in query_ids for seed in seeds}:
        raise ValueError("P2 records are missing query/seed combinations")
    if accurate_thresholds is None:
        unique = {row["query"]["transition_id"]: row["mse_by_block"] for row in grouped["frozen"]}
        accurate_thresholds = {str(h): float(np.quantile([values[h-1] for values in unique.values()],
                                    p1["evaluation"]["accurate_quantile"])) for h in horizons}
    results = {}
    for method in methods:
        rows = grouped[method]
        episodes = [row["query"]["source_episode_position"] for row in rows]
        keys = [(row["query"]["transition_id"], row["seed"]) for row in rows]
        by_horizon = {}
        for h in horizons:
            values = np.array([row["mse_by_block"][h-1] for row in rows])
            frozen = np.array([lookups["frozen"][key]["mse_by_block"][h-1] for key in keys])
            delta = values - frozen
            accurate = frozen <= accurate_thresholds[str(h)]
            result = {"endpoint_mse": episode_statistics(values, episodes, p1),
                      "mean_path_mse": episode_statistics([np.mean(row["mse_by_block"][:h]) for row in rows], episodes, p1),
                      "paired_delta_vs_frozen": episode_statistics(delta, episodes, p1),
                      "improvement_fraction": float(np.mean(delta < -p1["evaluation"]["harm_tolerance"])),
                      "harm_fraction": float(np.mean(delta > p1["evaluation"]["harm_tolerance"])),
                      "initially_accurate_delta": episode_statistics(delta[accurate], np.array(episodes)[accurate], p1),
                      "by_seed": {str(seed): {"mean_mse": float(np.mean([row["mse_by_block"][h-1]
                                  for row in rows if row["seed"] == seed]))} for seed in sorted(seeds)}}
            for control in ("forward", "reverse", "fr-fixed"):
                if control in methods and control != method:
                    difference = values - np.array([lookups[control][key]["mse_by_block"][h-1] for key in keys])
                    result[f"paired_delta_vs_{control}"] = episode_statistics(difference, episodes, p1)
            by_horizon[str(h)] = result
        results[method] = {"horizons": by_horizon,
            "mean_support_presentations": float(np.mean([row["adaptation"]["support_presentations"] for row in rows])),
            "mean_cumulative_unique_support": float(np.mean([row["adaptation"]["cumulative_unique_support"] for row in rows])),
            "latency_with_checks": episode_statistics([row["total_seconds"] for row in rows], episodes, p1)}
    return {"queries": len(query_ids), "episodes": len(set(row["query"]["source_episode_position"] for row in records)),
            "horizons": horizons, "accurate_thresholds": accurate_thresholds, "methods": results,
            "metric": "mean squared projected-latent coordinate error; endpoint and mean path",
            "uncertainty": "paired episode bootstrap; all query windows/seeds resampled within episode; one checkpoint"}


def format_method_summary(summary: dict, report_path) -> str:
    lines = [f"P2 recorded rollouts | {summary['queries']} queries / {summary['episodes']} episodes",
             "Method           " + "  ".join(f"H={h} MSE".ljust(12) for h in summary["horizons"]) + "  Final harm"]
    for method, row in summary["methods"].items():
        cells = "  ".join(f"{row['horizons'][str(h)]['endpoint_mse']['mean']:<12.7g}" for h in summary["horizons"])
        harm = row["horizons"][str(max(summary["horizons"]))]["harm_fraction"]
        lines.append(f"{method:<16} {cells}  {harm:>8.1%}")
    lines.extend([f"Report: {report_path}", "Compact per-query records are persisted; no simulator counterfactual claims."])
    return "\n".join(lines) + "\n"


def summarize_scopes(records: list[dict], config: dict, accurate_thresholds: dict | None = None) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty scope comparison")
    scopes, p1 = config["scopes"], effective_p1(config)
    methods, seeds = p1["adaptation"]["methods"], p1["evaluation"]["seeds"]
    lookups = {scope: {} for scope in scopes}
    grouped = defaultdict(list)
    for row in records:
        scope = row["scope"]
        key = (row["query"]["transition_id"], row["seed"], row["method"])
        if scope not in scopes or row["method"] not in methods or row["seed"] not in seeds:
            raise ValueError("Unknown scope/method/seed in saved records")
        if key in lookups[scope]:
            raise ValueError("Duplicate scope/query/method/seed row")
        lookups[scope][key] = row
        grouped[scope].append(row)
    expected = set(lookups[scopes[0]])
    if not expected or any(set(lookups[scope]) != expected for scope in scopes):
        raise ValueError("Scope comparisons must use exactly the same query/method/seed rows")
    query_ids = {key[0] for key in expected}
    if expected != {(q, seed, method) for q in query_ids for seed in seeds for method in methods}:
        raise ValueError("Incomplete scope/query/method/seed matrix")
    for key in expected:
        first = lookups[scopes[0]][key]
        for scope in scopes[1:]:
            other = lookups[scope][key]
            if first["query"] != other["query"]:
                raise ValueError("Query metadata differs between scopes")
            if "target_raw_rows" in first and first["target_raw_rows"] != other["target_raw_rows"]:
                raise ValueError("Target alignment differs between scopes")
            if key[2] == "frozen" and first["mse_by_block"] != other["mse_by_block"]:
                raise ValueError("Frozen errors differ between scopes")
    summaries, threshold = {}, accurate_thresholds
    for scope in scopes:
        summaries[scope] = summarize_methods(grouped[scope], config, threshold)
        if threshold is None:
            threshold = summaries[scope]["accurate_thresholds"]
    cross = {}
    for left, right in (("predictor", "last-block"), ("predictor-action", "predictor"),
                        ("predictor-action", "last-block")):
        if left not in scopes or right not in scopes:
            continue
        by_method = {}
        for method in methods:
            keys = sorted(key for key in expected if key[2] == method)
            lhs = np.array([lookups[left][key]["mse_by_block"] for key in keys])
            rhs = np.array([lookups[right][key]["mse_by_block"] for key in keys])
            frozen = np.array([lookups[left][(q, seed, "frozen")]["mse_by_block"] for q, seed, _ in keys])
            episodes = [lookups[left][key]["query"]["source_episode_position"] for key in keys]
            by_horizon = {}
            for h in config["horizons"]:
                harm_delta = ((lhs[:, h-1]-frozen[:, h-1] > p1["evaluation"]["harm_tolerance"]).astype(float)
                              -(rhs[:, h-1]-frozen[:, h-1] > p1["evaluation"]["harm_tolerance"]).astype(float))
                by_horizon[str(h)] = {
                    "endpoint_delta": episode_statistics(lhs[:, h-1]-rhs[:, h-1], episodes, p1),
                    "mean_path_delta": episode_statistics((lhs[:, :h]-rhs[:, :h]).mean(1), episodes, p1),
                    "harm_fraction_delta": episode_statistics(harm_delta, episodes, p1)}
            by_method[method] = by_horizon
        cross[f"{left}-minus-{right}"] = by_method
    module_summaries = {}
    for scope in scopes:
        module_summaries[scope] = {}
        for method in methods:
            rows = [row for row in grouped[scope] if row["method"] == method]
            roots = sorted({root for row in rows for root in row["adaptation"]["module_updates"]})
            module_summaries[scope][method] = {root: {
                key: float(np.mean([row["adaptation"]["module_updates"][root][key] for row in rows]))
                for key in ("trainable_parameters", "parameter_tensors_with_gradient",
                            "last_update_gradient_l2_after_clipping", "displacement_from_checkpoint_mse",
                            "displacement_from_checkpoint_l2")} for root in roots}
    return {"queries": len(query_ids), "episodes": summaries[scopes[0]]["episodes"],
            "horizons": config["horizons"], "accurate_thresholds": threshold,
            "scopes": summaries, "cross_scope": cross, "module_updates": module_summaries,
            "uncertainty": "paired episode bootstrap; repeated seeds/windows are retained within episode"}


def format_scope_summary(summary: dict, report_path) -> str:
    lines = [f"P2.5 scopes | {summary['queries']} queries / {summary['episodes']} episodes",
             "Scope             Method       " + "  ".join(f"H={h} MSE".ljust(12) for h in summary["horizons"]) + "  Final harm"]
    for scope, result in summary["scopes"].items():
        for method, row in result["methods"].items():
            cells = "  ".join(f"{row['horizons'][str(h)]['endpoint_mse']['mean']:<12.7g}" for h in summary["horizons"])
            harm = row["horizons"][str(max(summary["horizons"]))]["harm_fraction"]
            lines.append(f"{scope:<18}{method:<13}{cells} {harm:>9.1%}")
    lines.extend([f"Report: {report_path}", "Scope pairs are evaluated on identical windows, seeds and first-update support."])
    return "\n".join(lines)+"\n"


def summarize(records: list[dict], config: dict, accurate_thresholds: dict | None = None) -> dict:
    if "schedules" in config:
        from .schedule_reporting import summarize_schedules
        return summarize_schedules(records, config, accurate_thresholds)
    if "scopes" in config:
        return summarize_scopes(records, config, accurate_thresholds)
    return summarize_methods(records, config, accurate_thresholds)


def format_summary(summary: dict, report_path) -> str:
    if "schedules" in summary:
        from .schedule_reporting import format_schedule_summary
        return format_schedule_summary(summary, report_path)
    if "scopes" in summary:
        return format_scope_summary(summary, report_path)
    return format_method_summary(summary, report_path)
