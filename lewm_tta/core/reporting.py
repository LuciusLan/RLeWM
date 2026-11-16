"""Primary endpoint and paired uncertainty for the configured core matrix."""

from collections import defaultdict
import csv

import numpy as np

from ..outcomes.reporting import estimate, query_means, summarize as physical_summary
from .config import cells, primary_endpoint


def summarize(records, config):
    planned = [r for r in records if r["kind"] == "planned"]
    expected_cells = set(cells(config).values())
    expected = {(j, seed, cell) for j in config["core"]["iterations"]
                for seed in config["p1"]["evaluation"]["seeds"] for cell in expected_cells}
    matrix = defaultdict(list)
    for row in planned:
        matrix[row["case"]["case_id"]].append((row["iterations"], row["seed"], row["condition"]))
    if not matrix or any(len(rows) != len(expected) or set(rows) != expected for rows in matrix.values()):
        raise ValueError("Core results must contain the complete configured cell/budget/seed query matrix")
    control_ids = [r["case"]["case_id"] for r in records if r["kind"] == "recorded-control"]
    if len(set(control_ids)) != len(control_ids) or set(control_ids) != set(matrix):
        raise ValueError("Core reconstruction controls do not cover every completed query exactly once")
    endpoint = primary_endpoint(config)
    projected = [{**row, "final_success": row[endpoint+"_success"]} if row["kind"] == "planned" else row
                 for row in records]
    result = physical_summary(projected, config["core"])
    replacements = {"final_success_rate": "success_rate", "final_success_rate_by_seed": "success_rate_by_seed",
                    "nontrivial_final_success_rate": "nontrivial_success_rate",
                    "final_success_rate_difference": "success_rate_difference"}
    def rename(value):
        if isinstance(value, dict):
            return {replacements.get(k, k): rename(v) for k, v in value.items()}
        return [rename(v) for v in value] if isinstance(value, list) else value
    result = rename(result)
    episodes = {r["case"]["case_id"]: r["case"]["query"]["episode_id"] for r in planned}
    for curve in result["curves"]:
        rows = [r for r in planned if r["iterations"] == curve["iterations"] and r["condition"] == curve["condition"]]
        curve["final_state_success_rate"] = estimate(query_means(rows, lambda r: r["final_success"]), episodes,
            config["core"]["bootstrap_samples"], config["core"]["bootstrap_seed"])
        curve["mean_planning_calls"] = float(np.mean([r["planning_calls"] for r in rows]))
        curve["mean_executed_controls"] = float(np.mean([r["raw_steps"] for r in rows]))
        # Initially successful MPC cases perform zero work; they must contribute
        # zero rather than erase every compute column through key intersection.
        curve["compute_means"] = {name: float(np.mean([r["compute"].get(name, 0.) for r in rows]))
                                  for name in sorted(set().union(*(set(r["compute"]) for r in rows)))}
    result["primary_endpoint"] = endpoint
    result["targets_scope"] = "Minimum observed per-call J over this fixed protocol; first-call snapshots share search, later real replans use independent trajectories. No interpolation or per-query budget selection."
    result["deadline_scope"] = "Deadline applies to total effective planning plus live encoding until this fixed-J episode ends, excluding physics; it is not an online interruption policy. Shared first-call prefixes exclude earlier measurement overhead."
    return result


def format_summary(report):
    from ..interleave.config import adaptation_schedule, checkpoint_updates
    lines = [f"# P3 core: {len(report['cells'])} comparisons", "", report["protocol"], "",
             f"Primary success: {report['summary']['primary_endpoint']}; {report['control_budget']} control budget.",
             f"Completed {report['completed_queries']}/{report['requested_queries']} queries; seeds {report['config']['p1']['evaluation']['seeds']}.", "",
             "| J per call | Cell | Success | 95% CI | Nontrivial success | Score | Planning s mean / p95 | Calls | Controls |",
             "|---:|---|---:|---|---:|---:|---|---:|---:|"]
    if adaptation_schedule(report["config"]) == "sequential-depth":
        lines[2:2] = [f"Adaptation: sequential-depth, {report['config']['adaptation']['updates']} updates per "
                      f"selected depth; {checkpoint_updates(report['config'])} per checkpoint. "
                      "Final weights score every rollout depth.", ""]
    for row in report["summary"]["curves"]:
        interval = row["success_rate"]["ci95"]
        ci = "n/a" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        nontrivial = row["nontrivial_success_rate"]["mean"]
        lines.append(f"| {row['iterations']} | {row['condition']} | {row['success_rate']['mean']:.3f} | {ci} | "
                     f"{nontrivial if nontrivial is not None else 'n/a'} | {row['final_planning_score']['mean']:.4f} | "
                     f"{row['planning_seconds']['mean']:.3f} / {row['planning_seconds_p95']:.3f} | "
                     f"{row['mean_planning_calls']:.2f} | {row['mean_executed_controls']:.1f} |")
    lines += ["", report["summary"]["uncertainty"], report["summary"]["targets_scope"], report["summary"]["deadline_scope"],
              "Actual shared work and elapsed time are charged once in accounting; do not sum prefix counters to estimate actual run cost.",
              "Recorded future controls are reconstruction/diagnostic controls only; every real replan uses actual observations and the unchanged supplied goal."]
    if not report["completed"] or report["limited_run"]:
        lines += ["", "INCOMPLETE OR LIMITED: exploratory only; cannot create a validation lock."]
    return "\n".join(lines)+"\n"


def write_tables(output, report):
    (output/"summary.md").write_text(format_summary(report))
    with (output/"success-curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iterations", "condition", "success_rate", "ci95_low", "ci95_high", "final_state_success_rate",
                         "ever_success_rate", "planning_score", "mean_planning_seconds", "p95_planning_seconds",
                         "mean_planning_calls", "mean_executed_controls"])
        for row in report["summary"]["curves"]:
            writer.writerow([row["iterations"], row["condition"], row["success_rate"]["mean"],
                *(row["success_rate"]["ci95"] or [None, None]), row["final_state_success_rate"]["mean"],
                row["ever_success_rate"]["mean"], row["final_planning_score"]["mean"], row["planning_seconds"]["mean"],
                row["planning_seconds_p95"], row["mean_planning_calls"], row["mean_executed_controls"]])
