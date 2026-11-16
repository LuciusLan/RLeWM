"""Episode-paired prediction statistics and explicitly labeled CEM diagnostics."""

from itertools import combinations, product

import numpy as np

from ...reporting import episode_statistics
from ...rollout.config import effective_p1
from .config import nominal_budget, seeds_for_mode


def summarize(records, config):
    modes = config["temporal"]["modes"]
    mode_seeds = {mode: seeds_for_mode(config, mode) for mode in modes}
    horizon = config["planning"]["horizon"]
    lookup = {(row["query"]["transition_id"], row["seed"], row["mode"]): row for row in records}
    queries = sorted({key[0] for key in lookup})
    expected = {(query, seed, mode) for mode in modes for query, seed in product(queries, mode_seeds[mode])}
    if not queries or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicate temporal mode/query/seed matrix")
    for mode in modes:
        for query, seed in product(queries, mode_seeds[mode]):
            reference = lookup[query, seed, "frozen"]
            row, budget = lookup[query, seed, mode], nominal_budget(config, mode)
            for key in ("query", "goal_raw_row", "reference_fingerprint", "cem_seed", "cem_first_population_fingerprint"):
                if row[key] != reference[key]:
                    raise ValueError(f"Temporal pairing differs: {key}")
            primary = row["adaptation"]
            if (len(row["recorded_action_mse_by_block"]) != horizon
                    or len(row["selected_actions_normalized"]) != horizon
                    or len(row["cem_trace"]) != config["planning"]["iterations"]
                    or primary["sequence_evaluations"] != budget["cem_sequence_evaluations"]
                    or primary["transition_predictions"] != budget["cem_transition_predictions"]
                    or primary["requested_updates"] != budget["requested_gradient_updates"]
                    or not 0 <= primary["actual_updates"] <= primary["requested_updates"]
                    or primary["support_presentations"] > primary["actual_updates"]*config["adaptation"]["support_size"]):
                raise ValueError("Temporal CEM/adaptation budget or horizon mismatch")
            if primary["actual_updates"] < primary["requested_updates"] and not primary["support_shortfall_positions"]:
                raise ValueError("Missing temporal updates without a support shortfall")
            diagnostic = row["diagnostic_adaptation"]
            diagnostic_updates = 3*horizon*config["adaptation"]["updates"] if mode == "candidate-temporal" else 0
            if (diagnostic["sequence_evaluations"] != 3 or diagnostic["transition_predictions"] != 3*horizon
                    or diagnostic["requested_updates"] != diagnostic_updates
                    or not 0 <= diagnostic["actual_updates"] <= diagnostic_updates
                    or row["diagnostic_common_frozen_sequence_evaluations"] != 1
                    or row["diagnostic_common_frozen_transition_predictions"] != horizon):
                raise ValueError("Temporal diagnostic work is not separately accounted")
            for key in ("selected_model_goal_sse", "selected_plan_frozen_model_goal_sse"):
                if not np.isfinite(row[key]) or row[key] < 0:
                    raise ValueError("Nonfinite or negative temporal planning score")
            if not all(row["checks"].values()):
                raise ValueError("A temporal acceptance check failed")
    p1 = effective_p1(config)
    results = {}
    for mode in modes:
        keys = list(product(queries, mode_seeds[mode]))
        rows = [lookup[q, seed, mode] for q, seed in keys]
        episodes = [row["query"]["source_episode_position"] for row in rows]
        frozen = np.array([lookup[q, seed, "frozen"]["recorded_action_mse_by_block"] for q, seed in keys])
        errors = np.array([row["recorded_action_mse_by_block"] for row in rows])
        if not np.isfinite(errors).all() or (errors < 0).any():
            raise ValueError("Invalid temporal recorded-action errors")
        by_depth = {}
        for h in range(horizon):
            delta = errors[:, h]-frozen[:, h]
            by_depth[str(h+1)] = {
                "endpoint_mse": episode_statistics(errors[:, h], episodes, p1),
                "delta_vs_frozen": episode_statistics(delta, episodes, p1),
                "harm_fraction": episode_statistics(delta > p1["evaluation"]["harm_tolerance"], episodes, p1),
            }
        common_delta = [row["selected_plan_frozen_model_goal_sse"]-
                        lookup[q, seed, "frozen"]["selected_plan_frozen_model_goal_sse"]
                        for row, (q, seed) in zip(rows, keys)]
        results[mode] = {
            "seeds": mode_seeds[mode], "rows": len(rows),
            "recorded_action_prediction": by_depth,
            "path_mse": episode_statistics(errors.mean(axis=1), episodes, p1),
            "common_frozen_plan_score_delta": episode_statistics(common_delta, episodes, p1),
            "mean_selected_model_goal_sse": float(np.mean([row["selected_model_goal_sse"] for row in rows])),
            "mean_selected_plan_frozen_model_goal_sse": float(np.mean([row["selected_plan_frozen_model_goal_sse"] for row in rows])),
            "mean_primary_seconds": float(np.mean([row["primary_seconds_with_checks"] for row in rows])),
            "primary_seconds_p95": float(np.quantile([row["primary_seconds_with_checks"] for row in rows], .95)),
            "mean_diagnostic_seconds": float(np.mean([row["diagnostic_seconds_with_checks"] for row in rows])),
            "mean_primary_accounting": {key: float(np.mean([row["adaptation"][key] for row in rows]))
                for key in rows[0]["adaptation"] if key not in ("rounds", "audit_sha256")},
            "mean_diagnostic_accounting": {key: float(np.mean([row["diagnostic_adaptation"][key] for row in rows]))
                for key in rows[0]["diagnostic_adaptation"]},
            "by_seed": {str(seed): {
                "queries": len(queries),
                "final_position_mse": float(np.mean([row["recorded_action_mse_by_block"][-1]
                                                    for row in rows if row["seed"] == seed])),
                "mean_selected_plan_frozen_model_goal_sse": float(np.mean([
                    row["selected_plan_frozen_model_goal_sse"] for row in rows if row["seed"] == seed])),
            } for seed in mode_seeds[mode]},
        }
    paired = {}
    for right, left in combinations(modes, 2):
        shared = [seed for seed in config["p1"]["evaluation"]["seeds"]
                  if seed in mode_seeds[left] and seed in mode_seeds[right]]
        pair = {"left": left, "right": right, "seeds": shared,
                "rows": len(queries)*len(shared), "available": bool(shared)}
        paired[f"{left}-minus-{right}"] = pair
        if not shared:
            continue
        keys = list(product(queries, shared))
        lhs, rhs = ([lookup[q, seed, mode] for q, seed in keys] for mode in (left, right))
        episodes = [row["query"]["source_episode_position"] for row in lhs]
        a, b = (np.array([row["recorded_action_mse_by_block"] for row in rows]) for rows in (lhs, rhs))
        frozen = np.array([lookup[q, seed, "frozen"]["recorded_action_mse_by_block"] for q, seed in keys])
        tolerance = p1["evaluation"]["harm_tolerance"]
        pair["recorded_action_prediction"] = {str(h+1): {
            "endpoint_delta": episode_statistics(a[:, h]-b[:, h], episodes, p1),
            "harm_fraction_delta": episode_statistics(
                (a[:, h]-frozen[:, h] > tolerance).astype(float)
                -(b[:, h]-frozen[:, h] > tolerance).astype(float), episodes, p1),
        } for h in range(horizon)}
        pair["path_mse_delta"] = episode_statistics((a-b).mean(axis=1), episodes, p1)
        pair["common_frozen_plan_score_delta"] = episode_statistics([
            x["selected_plan_frozen_model_goal_sse"]-y["selected_plan_frozen_model_goal_sse"]
            for x, y in zip(lhs, rhs)], episodes, p1)
    all_episodes = {row["query"]["source_episode_position"] for row in records}
    return {"queries": len(queries), "episodes": len(all_episodes), "modes": results,
            "seeds_by_mode": mode_seeds, "paired_modes": paired,
            "nominal_primary_budgets": {mode: nominal_budget(config, mode) for mode in modes},
            "seed_pairing": "Mode summaries use their own seed sets; paired_modes use only common query/seed pairs, with no imputation or repetition of missing seeds.",
            "uncertainty": "paired episode bootstrap; sampled contexts and repeated seeds stay within episodes",
            "interpretation": "Recorded-action errors have observed targets. CEM/common-Frozen costs are model diagnostics, not physical success or a guaranteed upper bound."}


def format_summary(report):
    summary = report["summary"]
    horizon = str(report["config"]["planning"]["horizon"])
    lines = [f"P3 temporal | {summary['queries']} contexts / {summary['episodes']} episodes",
             "Mode                  Seeds        Rows  Recorded H-final MSE   Harm     Primary updates   Seconds"]
    for mode, values in summary["modes"].items():
        entry = values["recorded_action_prediction"][horizon]
        seeds = ",".join(map(str, values["seeds"]))
        lines.append(f"{mode:<22}{seeds:<13}{values['rows']:<6}{entry['endpoint_mse']['mean']:<23.7g}{entry['harm_fraction']['mean']:>7.1%}"
                     f"  {values['mean_primary_accounting']['actual_updates']:>15.1f}  {values['mean_primary_seconds']:>8.2f}")
    lines.append("Paired mode comparisons (common seeds only; final-position MSE delta, left minus right):")
    for name, pair in summary["paired_modes"].items():
        if pair["available"]:
            lines.append(f"  {name}: seeds={pair['seeds']}, rows={pair['rows']}, "
                         f"delta={pair['recorded_action_prediction'][horizon]['endpoint_delta']['mean']:.7g}")
        else:
            lines.append(f"  {name}: unavailable (no common seeds)")
    lines += ["Headline mode means may use different seed sets; use paired comparisons for direct differences.",
              "Primary adaptation budgets include setup and all CEM candidates, including the returned mean.",
              "Recorded-action/selected-plan diagnostic adaptations are charged separately.",
              "Search scores and Frozen rescoring are model proxies; no simulator execution or upper-bound guarantee."]
    return "\n".join(lines)+"\n"
