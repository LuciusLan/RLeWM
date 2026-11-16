"""Episode-paired recorded-action accuracy; clearly labeled search diagnostics."""

from itertools import product
import numpy as np


def interval(values, episodes, draws):
    unique = sorted(set(episodes.tolist()))
    sums = np.asarray([values[episodes == ep].sum() for ep in unique])
    counts = np.asarray([(episodes == ep).sum() for ep in unique])
    if not draws:
        return None
    sampled = np.random.default_rng(20261025).integers(len(unique), size=(draws, len(unique)))
    means = sums[sampled].sum(1)/counts[sampled].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(records, config):
    methods = config["adaptation"]["methods"]
    seeds = config["p1"]["evaluation"]["seeds"]
    horizon = config["planning"]["horizon"]
    lookup = {(row["query"]["transition_id"], row["seed"], row["method"]): row for row in records}
    ids = sorted({row["query"]["transition_id"] for row in records})
    expected = set(product(ids, seeds, methods))
    if not ids or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicated query/seed/method matrix")
    episodes = np.asarray([lookup[q, seeds[0], "frozen"]["query"]["source_episode_position"] for q in ids])
    for query, seed in product(ids, seeds):
        reference = lookup[query, seed, "frozen"]
        for method in methods:
            row = lookup[query, seed, method]
            if row["query"] != reference["query"] or row["goal_raw_row"] != reference["goal_raw_row"]:
                raise ValueError("Paired query/goal mismatch")
            for key in ("provisional_fingerprint", "cem_first_population_fingerprint", "cem_seed"):
                if row[key] != reference[key]:
                    raise ValueError(f"Paired planning randomness differs: {key}")
            if len(row["recorded_action_mse_by_block"]) != horizon:
                raise ValueError("P3 recorded-action horizon/record mismatch")
            expected_budget = config["planning"]["population"]*config["planning"]["iterations"]+1
            if (row["cem_candidate_evaluations"] != expected_budget
                    or row["cem_transition_predictions"] != expected_budget*horizon
                    or len(row["cem_trace"]) != config["planning"]["iterations"]):
                raise ValueError("CEM budget/trace does not match the configuration")
            if (row["diagnostic_sequence_evaluations"] != 4
                    or row["diagnostic_transition_predictions"] != 4*horizon):
                raise ValueError("Offline diagnostic budget mismatch")
            if len(row["selected_actions_normalized"]) != horizon:
                raise ValueError("Returned action plan horizon mismatch")
            adapted = row["adaptation"]
            requested_updates = 0 if method == "frozen" else config["adaptation"]["updates"]
            if (adapted["updates"] != len(adapted["rounds"]) or adapted["updates"] > requested_updates
                    or (adapted["updates"] < requested_updates and not adapted["skip_reason"])
                    or adapted["support_presentations"] != adapted["unique_support"]*adapted["updates"]):
                raise ValueError("Shared adaptation budget mismatch or unexplained skipped updates")
            if adapted["unique_support"] > config["adaptation"]["support_size"]:
                raise ValueError("Shared support exceeds the global unique budget")
            if not all(row["checks"].values()):
                raise ValueError("A per-record acceptance check failed")
    arrays = {method: np.asarray([[lookup[q, seed, method]["recorded_action_mse_by_block"]
                                  for seed in seeds] for q in ids]) for method in methods}
    if any(not np.isfinite(array).all() or (array < 0).any() for array in arrays.values()):
        raise ValueError("Invalid recorded-action errors")
    draws = config["evaluation"]["bootstrap_samples"]
    tolerance = config["p1"]["evaluation"]["harm_tolerance"]
    frozen = arrays["frozen"]
    results = {}
    for method, array in arrays.items():
        rows = [lookup[q, seed, method] for q, seed in product(ids, seeds)]
        by_horizon = {}
        for h in range(horizon):
            delta = (array[:, :, h]-frozen[:, :, h]).mean(1)
            harm = (array[:, :, h] > frozen[:, :, h]+tolerance).mean(1)
            baseline, mean = float(frozen[:, :, h].mean()), float(array[:, :, h].mean())
            by_horizon[f"h{h+1}"] = {
                "mse": mean, "relative_reduction_vs_frozen": 1-mean/baseline if baseline > 0 else None,
                "delta_vs_frozen": float(delta.mean()), "delta_ci95": interval(delta, episodes, draws),
                "harm_fraction": float(harm.mean()), "harm_ci95": interval(harm, episodes, draws),
                "mean_positive_harm": float(np.maximum(array[:, :, h]-frozen[:, :, h], 0).mean())}
        diagnostics = {key: float(np.mean([row[key] for row in rows])) for key in (
            "selected_model_goal_sse", "selected_plan_frozen_model_goal_sse", "frozen_plan_this_model_goal_sse",
            "selected_vs_recorded_raw_action_mse", "first_block_vs_recorded_raw_action_mse",
            "cem_seconds", "primary_seconds_with_checks", "diagnostic_seconds_with_checks",
            "cem_candidate_evaluations", "diagnostic_sequence_evaluations",
            "provisional_generation_seconds", "recorded_raw_action_out_of_bounds_fraction")}
        diagnostics["selected_model_goal_mse"] = diagnostics["selected_model_goal_sse"]/len(rows[0]["selected_predicted_endpoint"])
        diagnostics["cem_clipped_coordinate_fraction"] = float(np.mean([
            step["clipped_coordinate_fraction"] for row in rows for step in row["cem_trace"]]))
        diagnostics["cem_seconds_p95"] = float(np.quantile([row["cem_seconds"] for row in rows], .95))
        diagnostics["primary_seconds_p95"] = float(np.quantile([row["primary_seconds_with_checks"] for row in rows], .95))
        diagnostics["adaptation_seconds"] = float(np.mean([row["adaptation"]["timing_seconds"]["total_with_checks"] for row in rows]))
        for component in ("reset", "selection_and_provisional_prediction", "gradient_updates", "support_loss_evaluation"):
            diagnostics[f"adaptation_{component}_seconds"] = float(np.mean([
                row["adaptation"]["timing_seconds"][component] for row in rows]))
        diagnostics["support_presentations"] = float(np.mean([row["adaptation"]["support_presentations"] for row in rows]))
        diagnostics["unique_support"] = float(np.mean([row["adaptation"]["unique_support"] for row in rows]))
        diagnostics["support_shortfall_fraction"] = float(np.mean([
            row["adaptation"]["unique_support"] < config["adaptation"]["support_size"] for row in rows])) if method != "frozen" and config["adaptation"]["updates"] else 0.0
        results[method] = {"recorded_action_prediction": by_horizon,
                           "recorded_action_path_mse": float(array.mean()),
                           "search_diagnostics_not_physical_performance": diagnostics}
    pairs = {}
    for reference in ("uniform", "forward", "reverse"):
        if "fr-fixed" in arrays and reference in arrays:
            delta = (arrays["fr-fixed"][:, :, -1]-arrays[reference][:, :, -1]).mean(1)
            pairs[f"fr-fixed_minus_{reference}"] = {
                "recorded_endpoint_mse_delta": float(delta.mean()), "ci95": interval(delta, episodes, draws)}
    return {"queries": len(ids), "episodes": len(set(episodes.tolist())), "seeds": seeds,
            "rows": len(records), "methods": results, "selection_comparisons": pairs,
            "uncertainty": "Seed outcomes averaged within query; paired episode bootstrap, 95% percentile intervals; unadjusted exploratory comparisons",
            "physical_success": None, "physical_regret": None,
            "interpretation": "Accuracy refers only to recorded actions. Predicted costs and action agreement do not establish physical planning gains."}


def format_summary(report):
    summary = report["summary"]
    horizon = report["config"]["planning"]["horizon"]
    lines = ["P3-a offline CEM: recorded-action accuracy and model-only search diagnostics", "",
             f"{summary['queries']} queries / {summary['episodes']} episodes; seeds {summary['seeds']}.", "",
             "| Method | Recorded H-end MSE | Harm | Selected predicted SSE | Primary seconds |",
             "|---|---:|---:|---:|---:|"]
    for method, values in summary["methods"].items():
        accuracy = values["recorded_action_prediction"][f"h{horizon}"]
        search = values["search_diagnostics_not_physical_performance"]
        lines.append(f"| {method} | {accuracy['mse']:.7f} | {accuracy['harm_fraction']:.1%} | "
                     f"{search['selected_model_goal_sse']:.5f} | {search['primary_seconds_with_checks']:.3f} |")
    lines += ["", "Selected predicted SSE is each model's own cost; it is not a task score.",
              "Primary time includes shared adaptation, CEM and state checks, excluding offline diagnostics and cache loading.",
              "No selected action is executed; physical success, coverage and regret are unavailable."]
    return "\n".join(lines)+"\n"
