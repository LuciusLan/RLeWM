"""Paired physical performance curves with episode-cluster uncertainty."""

from collections import defaultdict
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


def estimate(values, episodes, samples, seed):
    """Average seeds within query first; resample whole collected episodes."""
    if not values:
        return {"mean": None, "ci95": None, "queries": 0, "episodes": 0}
    clusters = defaultdict(list)
    for query, value in sorted(values.items()):
        clusters[episodes[query]].append(float(value))
    blocks = list(clusters.values())
    point = float(np.mean(list(values.values())))
    interval = None
    if samples and len(blocks) > 1:
        rng = np.random.default_rng(seed)
        sums = np.asarray([sum(block) for block in blocks])
        sizes = np.asarray([len(block) for block in blocks])
        boot = []
        for _ in range(samples):
            selected = rng.integers(len(blocks), size=len(blocks))
            boot.append(float(sums[selected].sum()/sizes[selected].sum()))
        interval = np.quantile(boot, [.025, .975]).tolist()
    return {"mean": point, "ci95": interval, "queries": len(values), "episodes": len(blocks)}


def query_means(rows, value):
    groups = defaultdict(list)
    for row in rows:
        number = value(row)
        if number is not None:
            groups[row["case"]["case_id"]].append(float(number))
    return {query: float(np.mean(numbers)) for query, numbers in groups.items()}


def summarize(records, settings):
    planned = [row for row in records if row["kind"] == "planned"]
    controls = [row for row in records if row["kind"] == "recorded-control"]
    if not planned:
        raise ValueError("No physical planning outcomes")
    episodes = {row["case"]["case_id"]: row["case"]["query"]["episode_id"] for row in records}
    def stats(rows, value):
        return estimate(query_means(rows, value), episodes, settings["bootstrap_samples"], settings["bootstrap_seed"])
    groups = defaultdict(list)
    for row in planned:
        groups[row["iterations"], row["condition"]].append(row)
    curves = []
    for (iterations, label), rows in sorted(groups.items()):
        nontrivial = [row for row in rows if not row["initial_success"]]
        item = {"iterations": iterations, "condition": label, "records": len(rows),
                "seeds": sorted({row["seed"] for row in rows}),
                "final_success_rate_by_seed": {str(seed): float(np.mean([row["final_success"] for row in rows if row["seed"] == seed]))
                                               for seed in sorted({row["seed"] for row in rows})},
                "final_success_rate": stats(rows, lambda row: row["final_success"]),
                "ever_success_rate": stats(rows, lambda row: row["ever_success"]),
                "initial_success_rate": stats(rows, lambda row: row["initial_success"]),
                "nontrivial_final_success_rate": stats(nontrivial, lambda row: row["final_success"]),
                "final_planning_score": stats(rows, lambda row: row["final_metrics"]["planning_score"]),
                "final_goal_distance": stats(rows, lambda row: row["final_metrics"]["goal_distance"]),
                "planning_score_improvement": stats(rows, lambda row: row["planning_score_improvement"]),
                "goal_distance_reduction": stats(rows, lambda row: row["goal_distance_reduction"]),
                "planning_seconds": stats(rows, lambda row: row["planning_seconds"]),
                "planning_seconds_p95": float(np.quantile([row["planning_seconds"] for row in rows], .95)),
                "first_success_raw_step_among_hits": stats(rows, lambda row: row["first_success_raw_step"]),
                "success_by_deadline": [{"seconds": deadline,
                    **stats(rows, lambda row: row["final_success"] and row["planning_seconds"] <= deadline)}
                    for deadline in settings["deadlines"]],
                "compute_means": {name: float(np.mean([row["compute"][name] for row in rows]))
                    for name in sorted(set.intersection(*(set(row["compute"]) for row in rows)))},
                "final_metric_means": {name: float(np.mean([row["final_metrics"][name] for row in rows]))
                    for name in rows[0]["final_metrics"]}}
        curves.append(item)
    targets = []
    for label in sorted({row["condition"] for row in planned}):
        points = [point for point in curves if point["condition"] == label]
        for target in settings["success_targets"]:
            reached = [point for point in points if point["final_success_rate"]["mean"] >= target]
            confident = [point for point in points if point["final_success_rate"]["ci95"] is not None
                         and point["final_success_rate"]["ci95"][0] >= target]
            targets.append({"condition": label, "target_success_rate": target,
                "minimum_observed_iterations": min((p["iterations"] for p in reached), default=None),
                "minimum_observed_mean_planning_seconds": min((p["planning_seconds"]["mean"] for p in reached), default=None),
                "minimum_observed_iterations_lower_ci_reaches_target": min((p["iterations"] for p in confident), default=None)})
    paired = []
    budgets = sorted({j for j, _ in groups})
    labels = sorted({label for _, label in groups})
    for budget in budgets:
        indexed = {label: {(r["case"]["case_id"], r["seed"]): r for r in groups[budget, label]} for label in labels}
        contrasts = [(f"{right} - {left}", {left: -1., right: 1.}) for left, right in combinations(labels, 2)]
        if {"baseline", "rca", "proposals", "combined"}.issubset(labels):
            contrasts.append(("factorial interaction: combined - rca - proposals + baseline",
                              {"combined": 1., "rca": -1., "proposals": -1., "baseline": 1.}))
        for name, coefficients in contrasts:
            common = set.intersection(*(set(indexed[label]) for label in coefficients))
            if not common:
                continue
            keys = sorted(common)
            differences = []
            for query, seed in keys:
                rows = {label: indexed[label][query, seed] for label in coefficients}
                differences.append({"case": rows[next(iter(rows))]["case"],
                    "success_difference": sum(coefficients[label]*row["final_success"] for label, row in rows.items()),
                    "score_difference": sum(coefficients[label]*row["final_metrics"]["planning_score"] for label, row in rows.items())})
            paired.append({"iterations": budget, "contrast": name, "paired_query_seed_rows": len(keys),
                "paired_seeds": sorted({seed for _, seed in keys}),
                "final_success_rate_difference": stats(differences, lambda row: row["success_difference"]),
                "success_positive_difference_rate": stats(differences, lambda row: row["success_difference"] > 0),
                "success_negative_difference_rate": stats(differences, lambda row: row["success_difference"] < 0),
                "final_planning_score_difference": stats(differences, lambda row: row["score_difference"])})
    return {"curves": curves, "targets": targets, "paired_contrasts": paired,
        "recorded_action_control": {"queries": len(controls),
            "final_success_rate": stats(controls, lambda row: row["final_success"]),
            "initial_success_rate": stats(controls, lambda row: row["initial_success"])},
        "uncertainty": "95% percentile bootstrap of collected episodes, averaging planner seeds within query; unavailable with fewer than two episodes or bootstrap_samples=0",
        "targets_scope": "minimum over independently evaluated budgets only; no interpolation, monotonicity assumption or validation-to-test tuning",
        "deadline_scope": "each fixed-J plan must finish planning by the stated deadline and succeed; misses count as failure; no per-query budget selection or online deadline controller"}


def format_summary(report):
    lines = ["# P3 executed-plan success", "", report["protocol"], "",
             f"Physical goal criterion: {report['metric_contract']['success']}.", "",
             "| CEM J | Condition | Final success | 95% CI | Initially successful | Success on initially unsuccessful | Score | Mean / p95 planning s |",
             "|---:|---|---:|---|---:|---:|---:|---:|"]
    for item in report["summary"]["curves"]:
        rate = item["final_success_rate"]
        interval = rate["ci95"]
        ci = "unavailable" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        nontrivial = item["nontrivial_final_success_rate"]["mean"]
        nontrivial_text = "n/a" if nontrivial is None else f"{nontrivial:.3f}"
        lines.append(f"| {item['iterations']} | {item['condition']} | {rate['mean']:.3f} | {ci} | "
                     f"{item['initial_success_rate']['mean']:.3f} | {nontrivial_text} | {item['final_planning_score']['mean']:.4f} | "
                     f"{item['planning_seconds']['mean']:.4f} / {item['planning_seconds_p95']:.4f} |")
    lines += ["", "## Minimum observed CEM iterations to target success", "",
              "| Condition | Target | Minimum observed J | Lower 95% CI reaches target at J |",
              "|---|---:|---:|---:|"]
    for item in report["summary"]["targets"]:
        observed = item["minimum_observed_iterations"]
        confident = item["minimum_observed_iterations_lower_ci_reaches_target"]
        lines.append(f"| {item['condition']} | {item['target_success_rate']:.2f} | {observed if observed is not None else 'not reached'} | {confident if confident is not None else 'not reached'} |")
    lines += ["", "Recorded-action future reconstruction controls must all pass before publishing this report.",
              report["summary"]["uncertainty"], report["summary"]["deadline_scope"],
              "Task-specific scores/distances are not pooled across environments. Model MSE/SSE remain in the original planning reports.",
              "This is fixed-horizon open-loop execution; it does not measure full-episode closed-loop MPC success.",
              "LIMITED RUN: not suitable for validation selection or headline results." if report["limited_run"] else "", ""]
    return "\n".join(lines)


def write_tables(directory, report):
    directory = Path(directory)
    (directory/"summary.md").write_text(format_summary(report))
    columns = ["iterations", "condition", "records", "success_rate", "ci95_low", "ci95_high", "initial_success_rate",
               "nontrivial_success_rate", "planning_score", "goal_distance", "mean_planning_seconds", "p95_planning_seconds"]
    with (directory/"success-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for point in report["summary"]["curves"]:
            low, high = point["final_success_rate"]["ci95"] or (None, None)
            writer.writerow(dict(zip(columns, [point["iterations"], point["condition"], point["records"],
                point["final_success_rate"]["mean"], low, high, point["initial_success_rate"]["mean"],
                point["nontrivial_final_success_rate"]["mean"], point["final_planning_score"]["mean"],
                point["final_goal_distance"]["mean"], point["planning_seconds"]["mean"], point["planning_seconds_p95"]])))
