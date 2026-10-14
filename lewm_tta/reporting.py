"""Paired evaluation and uncertainty clustered by source episode."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .rank_reporting import summarize_topk_records


def episode_statistics(values, episodes, config: dict):
    """Descriptive statistics with the existing paired episode bootstrap."""
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluation metrics")
    unique = np.unique(episodes)
    interval = None
    samples = config["evaluation"]["bootstrap_samples"]
    if len(unique) > 1 and samples:
        totals = np.array([values[episodes == ep].sum() for ep in unique])
        counts = np.array([(episodes == ep).sum() for ep in unique])
        rng = np.random.default_rng(config["data"]["split_seed"])
        means = []
        for _ in range(samples):
            draw = rng.integers(len(unique), size=len(unique))
            means.append(totals[draw].sum() / counts[draw].sum())
        interval = np.quantile(means, [.025, .975]).tolist()
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "p90": float(np.quantile(values, .9)), "mean_ci95": interval,
            "rows": len(values), "episodes": len(unique)}


def score_record(record: dict, target) -> dict:
    target = np.asarray(target, dtype=np.float64)
    def mse(prediction):
        return float(np.mean((np.asarray(prediction, dtype=np.float64) - target) ** 2))
    record["mse_initial"] = mse(record["prediction_initial"])
    record["mse_final"] = mse(record["prediction_final"])
    record["paired_delta"] = record["mse_final"] - record["mse_initial"]
    for row in record["rounds"]:
        row["query_mse_before"] = mse(row["prediction_before"])
        row["query_mse_after"] = mse(row["prediction_after"])
    return record


def summarize(records: list[dict], config: dict, accurate_threshold: float | None = None) -> dict:
    settings = config["evaluation"]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)
    # One initial error per query; seeds and methods do not create new episodes.
    initial = {row["query"]["transition_id"]: row["mse_initial"] for row in records}
    if accurate_threshold is None:
        accurate_threshold = float(np.quantile(list(initial.values()), settings["accurate_quantile"]))

    def statistics(values, episodes):
        return episode_statistics(values, episodes, config)

    results = {}
    lookups = {method: {(row["query"]["transition_id"], row["seed"]): row for row in rows}
               for method, rows in grouped.items()}
    for method, rows in grouped.items():
        episodes = [row["query"]["source_episode_position"] for row in rows]
        accurate = [row for row in rows if row["mse_initial"] <= accurate_threshold]
        result = {"mse": statistics([row["mse_final"] for row in rows], episodes),
                  "paired_delta_vs_frozen": statistics([row["paired_delta"] for row in rows], episodes),
                  "improvement_fraction": float(np.mean([row["paired_delta"] < -settings["harm_tolerance"] for row in rows])),
                  "harm_fraction": float(np.mean([row["paired_delta"] > settings["harm_tolerance"] for row in rows])),
                  "initially_accurate_delta": statistics([row["paired_delta"] for row in accurate],
                      [row["query"]["source_episode_position"] for row in accurate]),
                  "mean_support_presentations": float(np.mean([row["support_presentations"] for row in rows])),
                  "mean_cumulative_unique_support": float(np.mean([row["cumulative_unique_support"] for row in rows])),
                  "latency_with_checks": statistics([row["timing_seconds"]["total_with_checks"] for row in rows], episodes),
                  "mse_by_update": [statistics([row["rounds"][index]["query_mse_after"] for row in rows], episodes)
                       for index in range(len(rows[0]["rounds"]))]}
        result["by_seed"] = {str(seed): {"mean_mse": float(np.mean([row["mse_final"] for row in rows if row["seed"] == seed])),
            "mean_delta": float(np.mean([row["paired_delta"] for row in rows if row["seed"] == seed]))}
            for seed in sorted({row["seed"] for row in rows})}
        for control in ("forward", "fr-fixed"):
            if control in lookups and control != method:
                comparison = [row["mse_final"] - lookups[control][(row["query"]["transition_id"], row["seed"])]["mse_final"]
                              for row in rows]
                result[f"paired_delta_vs_{control}"] = statistics(comparison, episodes)
        ordered = sorted(rows, key=lambda row: row["paired_delta"])
        result["examples"] = {"improved": [row["query"] for row in ordered
            if row["paired_delta"] < -settings["harm_tolerance"]][:3],
            "harmed": [row["query"] for row in reversed(ordered)
                       if row["paired_delta"] > settings["harm_tolerance"]][:3]}
        results[method] = result
    diagnostics = {}
    topk = summarize_topk_records(records, statistics)
    for method, rows in grouped.items():
        valid = [row for row in rows if row["rank_diagnostics_initial"]["memory_size"]]
        if not valid:
            continue
        # Average seeds per query before exploratory correlations.
        by_query = defaultdict(list)
        for row in valid:
            by_query[row["query"]["transition_id"]].append(row)
        gap = [group[0]["rank_diagnostics_initial"]["mean_top1_log2_opposite_rank"] for group in by_query.values()]
        rank_episodes = [group[0]["query"]["source_episode_position"] for group in by_query.values()]
        forward_cross = [group[0]["rank_diagnostics_initial"]["forward_top1_reverse_rank"] for group in by_query.values()]
        reverse_cross = [group[0]["rank_diagnostics_initial"]["reverse_top1_forward_rank"] for group in by_query.values()]
        error = [group[0]["mse_initial"] for group in by_query.values()]
        delta = [float(np.mean([row["paired_delta"] for row in group])) for group in by_query.values()]
        def spearman(a, b):
            def ranks(values):
                values = np.asarray(values)
                order = np.argsort(values, kind="stable")
                result = np.empty(len(values), dtype=float)
                start = 0
                while start < len(values):
                    end = start + 1
                    while end < len(values) and values[order[end]] == values[order[start]]:
                        end += 1
                    result[order[start:end]] = (start + end - 1) / 2
                    start = end
                return result
            if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
                return None
            return float(np.corrcoef(ranks(a), ranks(b))[0, 1])
        diagnostics[method] = {"forward_top1_reverse_rank": statistics(forward_cross, rank_episodes),
            "reverse_top1_forward_rank": statistics(reverse_cross, rank_episodes),
            "top1_log2_opposite_rank": statistics(gap, rank_episodes),
            "supplementary_mean_topk_overlap": float(np.mean([
                row["rank_diagnostics_initial"]["supplementary_topk"]["overlap_fraction"] for row in valid])),
            "top1_cross_rank_spearman_with_frozen_error": spearman(gap, error),
            "top1_cross_rank_spearman_with_adaptation_delta": spearman(gap, delta),
            "interpretation": "exploratory associations; disagreement is not a calibrated error detector"}
        if method in topk:
            diagnostics[method]["topk"] = topk[method]
    return {"queries": len(initial), "episodes": len({row["query"]["source_episode_position"] for row in records}),
            "accurate_threshold": accurate_threshold, "methods": results, "rank_diagnostics": diagnostics,
            "uncertainty": "paired episode bootstrap; all windows/seeds resampled together; conditional on one checkpoint",
            "metric": "mean squared projected-latent coordinate error"}


def format_summary(summary: dict, output) -> str:
    lines = [f"P1 | {summary['queries']} queries / {summary['episodes']} episodes",
             "Method           Mean MSE      Delta vs frozen [95% CI]            Improve   Harm"]
    for method, row in summary["methods"].items():
        delta = row["paired_delta_vs_frozen"]
        interval = delta["mean_ci95"]
        ci = f"[{interval[0]:.4g}, {interval[1]:.4g}]" if interval else "--"
        lines.append(f"{method:<16} {row['mse']['mean']:<13.6g} {delta['mean']:<12.4g} {ci:<22} "
                     f"{row['improvement_fraction']:>7.1%} {row['harm_fraction']:>7.1%}")
    if summary["rank_diagnostics"]:
        ranks = next(iter(summary["rank_diagnostics"].values()))
        if "topk" in ranks:
            topk = ranks["topk"]
            k_label = ",".join(str(value) for value in topk["requested_k_values"])
            lines.append(f"Exact top-{k_label} cross-ranks (pooled candidate medians): "
                f"forward in reverse = {topk['forward_candidates_in_reverse']['median']:g}; "
                f"reverse in forward = {topk['reverse_candidates_in_forward']['median']:g}; "
                f"mean |log2 R/F| = {topk['bidirectional_mean_absolute_log2_rank_ratio']['mean']:.4g}")
            lines.append("Rank diagnostic selection: " + "; ".join(topk["selection_rules"]))
        lines.append("Exact initial cross-ranks (medians): "
            f"forward top-1 in reverse = {ranks['forward_top1_reverse_rank']['median']:g}; "
            f"reverse top-1 in forward = {ranks['reverse_top1_forward_rank']['median']:g}")
    lines.append(f"Report: {output}")
    return "\n".join(lines) + "\n"
