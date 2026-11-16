"""Paired observed outcomes; average seeds within case and cluster source episodes."""

from collections import defaultdict
import numpy as np


def statistics(values, episodes, samples, seed):
    groups = defaultdict(list)
    for value, episode in zip(values, episodes):
        groups[episode].append(float(value))
    means = np.asarray([np.mean(group) for _, group in sorted(groups.items())])
    if not len(means) or not np.isfinite(means).all():
        raise ValueError("Empty or nonfinite simulator metrics")
    interval = None
    if len(means) > 1 and samples:
        rng = np.random.default_rng(seed)
        draws = rng.integers(len(means), size=(samples, len(means)))
        interval = np.quantile(means[draws].mean(1), [.025, .975]).tolist()
    return {"mean": float(means.mean()), "ci95": interval, "episodes": len(means), "rows": len(values)}


def summarize(records, config):
    groups, lookup = defaultdict(list), {}
    for row in records:
        cell = (row["method"], row["proposal_source"])
        key = (row["case_id"], row["seed"])
        if (cell, key) in lookup:
            raise ValueError("Duplicate simulator outcome")
        lookup[cell, key] = row
        groups[cell].append(row)
    def fields(row):
        result = {f"final_{key}": float(value) for key, value in row["final_metrics"].items()}
        result.update({f"best_{key}": float(value) for key, value in row["best_metrics"].items()})
        result.update(ever_success=float(row["ever_success"]), raw_steps=row["raw_steps"],
                      initial_success=float(row["initial_success"]), planning_calls=row["planning_calls"],
                      wall_seconds=row["wall_seconds"], planning_seconds=row["planning_seconds"],
                      context_encoding_seconds=row["context_encoding_seconds"])
        if "ever_out_of_frame" in row:
            result["ever_out_of_frame"] = float(row["ever_out_of_frame"])
        if row["planning_calls"]:
            result["seconds_per_planning_call"] = (row["planning_seconds"]+row["context_encoding_seconds"])/row["planning_calls"]
        return result
    samples, seed = config["evaluation"]["bootstrap_samples"], config["simulation"]["cohort_seed"]
    def stats(rows, key):
        return statistics([fields(row)[key] for row in rows], [row["query"]["source_episode_position"] for row in rows], samples, seed)
    cells = {f"{method}/{proposal}": {"rows": len(rows), "metrics": {key: stats(rows, key)
             for key in fields(rows[0]) if all(key in fields(row) for row in rows)}}
             for (method, proposal), rows in groups.items()}
    for (method, proposal), rows in groups.items():
        count = sum(row["planning_calls"] for row in rows)
        cells[f"{method}/{proposal}"]["compute"] = {
            "planning_calls": count,
            "planning_seconds": sum(row["planning_seconds"] for row in rows),
            "encoding_seconds": sum(row["context_encoding_seconds"] for row in rows),
            "planning_plus_encoding_seconds_per_call": (sum(row["planning_seconds"]+row["context_encoding_seconds"] for row in rows)/count if count else None)}
        if config.get("simulation", {}).get("task") == "green-target":
            successes = [row["first_success_raw_step"] for row in rows if row["first_success_raw_step"] is not None]
            cells[f"{method}/{proposal}"]["time_to_success"] = {
                "successful_rows": len(successes), "mean_raw_steps_success_only": float(np.mean(successes)) if successes else None,
                "note": "conditional on success; includes zero-step initially successful cases; failures are not assigned a fictitious success time"}
    # Every completed case/seed must contain all six model/proposal cells.
    expected_cells = [(method, proposal) for method in config["adaptation"]["methods"] for proposal in config["proposals"]["sources"]]
    keys = {key for cell, key in lookup if cell == expected_cells[0]}
    if not keys or any({key for cell, key in lookup if cell == expected} != keys for expected in expected_cells):
        raise ValueError("Incomplete paired six-cell simulator matrix")
    control_cell = ("recorded-control", "recorded-actions")
    if control_cell in groups and {key for cell, key in lookup if cell == control_cell} != keys:
        raise ValueError("Recorded-action controls do not cover every paired case/seed")
    for case_id in {key[0] for key in keys}:
        if set(config["simulation"].get("seeds", [key[1] for key in keys])) != {seed for case, seed in keys if case == case_id}:
            raise ValueError("Simulator cases have different or incomplete seed sets")
    comparisons = {}
    def contrast(name, terms):
        available = sorted(set.intersection(*[{key for cell, key in lookup if cell == condition} for _, condition in terms]))
        metric_names = [key for key in fields(lookup[terms[0][1], available[0]]) if key.startswith(("final_", "best_")) or key in ("ever_success", "ever_out_of_frame")]
        result = {}
        for metric in metric_names:
            values = [sum(weight*fields(lookup[condition, key])[metric] for weight, condition in terms) for key in available]
            episodes = [lookup[terms[0][1], key]["query"]["source_episode_position"] for key in available]
            result[metric] = statistics(values, episodes, samples, seed)
        comparisons[name] = result
    for proposal in config["proposals"]["sources"]:
        for method, control in (("uniform", "frozen"), ("fr-fixed", "frozen"), ("fr-fixed", "uniform")):
            contrast(f"{method}-minus-{control}/{proposal}", [(1, (method, proposal)), (-1, (control, proposal))])
    for method in config["adaptation"]["methods"]:
        contrast(f"guided-minus-gaussian/{method}", [(1, (method, "context-fragments")), (-1, (method, "gaussian"))])
    for control in ("frozen", "uniform"):
        contrast(f"interaction/fr-fixed-minus-{control}", [(1, ("fr-fixed", "context-fragments")), (-1, ("fr-fixed", "gaussian")),
                                                          (-1, (control, "context-fragments")), (1, (control, "gaussian"))])
    if ("recorded-control", "recorded-actions") in groups:
        for condition in expected_cells:
            contrast(f"{condition[0]}/{condition[1]}-minus-recorded-control",
                     [(1, condition), (-1, ("recorded-control", "recorded-actions"))])
    return {"episodes": len({row["query"]["source_episode_position"] for row in records}),
            "seeds": sorted({row["seed"] for row in records}),
            "cases": len({row["case_id"] for row in records}), "rows": len(records), "cells": cells,
            "paired_contrasts": comparisons, "uncertainty": "pointwise unadjusted 95% episode bootstrap; seeds averaged within episode; one case per episode",
            "interpretation": (f"observed fixed-green-target outcomes on prespecified {config['simulation'].get('split', 'validation')} reset cases; canonical-pusher image-goal cost; block-only coverage success"
                               if config.get("simulation", {}).get("task") == "green-target" else
                               "observed outcomes from fresh simulator validation episodes; complete original reset/past-prefix reconstruction; test remains reserved"
                               if config.get("query_dataset") else
                               "observed outcomes in the reconstructed simulator; validation pilot, not an untouched test or exact original-trajectory replay")}


def format_summary(report):
    metrics = [f"final_{name}" for name in report["environment"]["metric_directions"]]
    metrics += ["ever_success", "raw_steps", "wall_seconds"]
    if report.get("task") == "green-target":
        metrics = ["best_coverage", "final_coverage", "ever_success", "initial_success", "ever_out_of_frame", "raw_steps", "wall_seconds"]
    lines = [f"# Simulator {report['mode']} {report.get('split', 'validation')}", "", f"Episodes: {report['summary']['episodes']}; completed: {report['completed']}", "",
             "| Treatment | " + " | ".join(metrics) + " |", "|---|" + "---:|"*len(metrics)]
    for cell, item in report["summary"]["cells"].items():
        values = [f"{item['metrics'][key]['mean']:.5g}" if key in item["metrics"] else "—" for key in metrics]
        lines.append(f"| {cell} | " + " | ".join(values) + " |")
    lines += ["", report["summary"]["interpretation"], "", "Full paired contrasts and episode intervals are in report.json.", ""]
    return "\n".join(lines)
