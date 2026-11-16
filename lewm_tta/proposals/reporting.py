"""Paired proposal comparisons under each fixed model; no physical-score claims."""

from itertools import product
import numpy as np

from ..planning.reporting import summarize as summarize_planning, interval
from .effects import factorial_effects, proposal_pairs
from .config import goal_settings


def paired_records(records, config):
    """Audit the complete factorial matrix before estimating any effects."""
    methods = config["adaptation"]["methods"]
    sources = config["proposals"]["sources"]
    seeds = config["p1"]["evaluation"]["seeds"]
    ids = sorted({row["query"]["transition_id"] for row in records})
    lookup = {(row["query"]["transition_id"], row["seed"], row["method"], row["proposal_source"]): row for row in records}
    if not ids or len(lookup) != len(records) or set(lookup) != set(product(ids, seeds, methods, sources)):
        raise ValueError("Incomplete/duplicate query x seed x TTT x proposal matrix")
    for query, seed in product(ids, seeds):
        baseline = lookup[query, seed, "frozen", "gaussian"]
        for method, source in product(methods, sources):
            row = lookup[query, seed, method, source]
            if "new_cem_candidate_evaluations" in row:
                imported = row.get("result_origin") == "reused"
                expected_new = 0 if imported else config["planning"]["population"]*config["planning"]["iterations"]+1
                if row["new_cem_candidate_evaluations"] != expected_new or (imported and not row.get("reuse")):
                    raise ValueError("New/reused CEM work or origin is inconsistent")
            for key in ("query", "current_raw_row", "goal_raw_row", "target_raw_rows", "scope",
                        "provisional_seed", "provisional_fingerprint", "cem_seed"):
                if row[key] != baseline[key]:
                    raise ValueError(f"Paired TTT/proposal inputs differ: {key}")
            if row["base_first_population_fingerprint"] != baseline["base_first_population_fingerprint"]:
                raise ValueError("Base Gaussian population differs between proposal/TTT arms")
            reference = lookup[query, seed, method, "gaussian"]
            if row["adaptation"]["support_fingerprint"] != reference["adaptation"]["support_fingerprint"]:
                raise ValueError("Proposal source changed TTT support")
            for key in ("reset_seed", "updates", "unique_support", "support_presentations",
                        "parameter_displacement_mse", "skip_reason"):
                if row["adaptation"][key] != reference["adaptation"][key]:
                    raise ValueError(f"Proposal source changed shared adaptation: {key}")
            if row["recorded_action_mse_by_block"] != reference["recorded_action_mse_by_block"]:
                raise ValueError("Proposal search changed the shared model's recorded-action predictions")
            same_proposal = lookup[query, seed, "frozen", source]
            for key in ("fragment_fingerprint", "injection_fingerprint"):
                if row["proposal"][key] != same_proposal["proposal"][key]:
                    raise ValueError("TTT changed action-proposal selection/noise")
            if source == "goal-context-fragments" and config["proposals"]["injection_slots"]:
                signature = row["proposal"].get("goal_query_fingerprint")
                if not signature or signature != same_proposal["proposal"].get("goal_query_fingerprint"):
                    raise ValueError("Missing or mismatched supplied retrieval goal across models")
            if row["proposal"]["source"] != source:
                raise ValueError("Proposal metadata differs from the comparison cell")
            slots = row["proposal"]["injected_slots"]
            if slots != len(row["proposal"]["slot_provenance"]) or slots >= config["planning"]["population"]:
                raise ValueError("Invalid injection population accounting")
            if not 0 <= slots <= config["proposals"]["injection_slots"]:
                raise ValueError("Injected slots exceed the configured budget")
            if any(step.get("directly_injected_candidates", 0) for step in row["cem_trace"][1:]):
                raise ValueError("Injection occurred after the first CEM iteration")
            # These diagnostic controls must collapse the appropriate factor.
            parity = []
            if config["proposals"]["injection_slots"] == 0:
                parity.append(reference)
            if config["adaptation"]["updates"] == 0:
                parity.append(same_proposal)
            if (source == "goal-context-fragments" and "context-fragments" in sources
                    and goal_settings(config)["goal_weight"] == 0):
                parity.append(lookup[query, seed, method, "context-fragments"])
            for control in parity:
                for key in ("cem_first_population_fingerprint", "selected_actions_normalized",
                            "selected_model_goal_sse", "selected_plan_frozen_model_goal_sse",
                            "recorded_action_mse_by_block"):
                    if row[key] != control[key]:
                        raise ValueError(f"Zero injection/update/goal-weight control did not reproduce its reference: {key}")
    return lookup, ids


def summarize(records, config):
    methods, sources = config["adaptation"]["methods"], config["proposals"]["sources"]
    seeds = config["p1"]["evaluation"]["seeds"]
    lookup, ids = paired_records(records, config)
    groups = {source: summarize_planning([row for row in records if row["proposal_source"] == source], config)
              for source in sources}
    episodes = np.asarray([lookup[query, seeds[0], "frozen", "gaussian"]["query"]["source_episode_position"] for query in ids])
    draws = config["evaluation"]["bootstrap_samples"]

    def values(method, source, metric):
        return np.asarray([[lookup[query, seed, method, source][metric] for seed in seeds] for query in ids]).mean(1)

    comparisons = {}
    pairs = proposal_pairs(sources)
    for method in methods:
        comparisons[method] = {}
        for left, right in pairs:
            metrics = {}
            for metric in ("selected_model_goal_sse", "selected_plan_frozen_model_goal_sse",
                           "selected_vs_recorded_raw_action_mse", "primary_seconds_with_checks"):
                delta = values(method, left, metric)-values(method, right, metric)
                metrics[metric] = {"mean_difference": float(delta.mean()), "ci95": interval(delta, episodes, draws)}
            comparisons[method][f"{left}_minus_{right}"] = metrics
    injection = {}
    for source in sources:
        rows = [lookup[q, seed, "frozen", source] for q, seed in product(ids, seeds)]
        injection[source] = {
            "mean_selected_fragments": float(np.mean([len(r["proposal"]["fragments"]) for r in rows])),
            "mean_injected_candidates": float(np.mean([r["proposal"]["injected_slots"] for r in rows])),
            "fragment_shortfall_fraction": float(np.mean([r["proposal"]["diagnostics"]["fragment_shortfall"] > 0 for r in rows])),
            "mean_direct_first_iteration_elite_fraction": float(np.mean([r["direct_elite_source_fraction"] for r in rows])),
            "direct_first_iteration_elite_fraction_by_model": {
                method: float(np.mean([lookup[q, seed, method, source]["direct_elite_source_fraction"]
                                       for q, seed in product(ids, seeds)])) for method in methods},
            "diagnostic_scope": "Fragment counts, selection time and distances are shared across models; legacy mean elite fraction is Frozen only.",
            "mean_proposal_seconds": float(np.mean([r["proposal"]["total_seconds"] for r in rows])),
            "mean_context_distance": (float(np.mean([item["context_distance"] for r in rows for item in r["proposal"]["fragments"]]))
                                      if any(r["proposal"]["fragments"] for r in rows) else None),
            "mean_goal_endpoint_distance": (float(np.mean([item["goal_endpoint_distance"] for r in rows
                for item in r["proposal"]["fragments"] if "goal_endpoint_distance" in item]))
                if any("goal_endpoint_distance" in item for r in rows for item in r["proposal"]["fragments"]) else None),
            "mean_combined_retrieval_distance": (float(np.mean([item["retrieval_distance"] for r in rows
                for item in r["proposal"]["fragments"] if "retrieval_distance" in item]))
                if any("retrieval_distance" in item for r in rows for item in r["proposal"]["fragments"]) else None)}
    effects = factorial_effects(lookup, ids, config)
    interaction = {}
    if "context-fragments" in sources:
        for method in methods:
            if method == "frozen":
                continue
            estimate = effects["difference_in_differences"][method]["context-fragments_minus_gaussian"]
            interaction[method] = {"common_frozen_model_sse_difference_in_differences": estimate["mean_difference"],
                                   "ci95": estimate["ci95"], "physical_interaction": None}
    return {"queries": len(ids), "episodes": len(set(episodes.tolist())), "rows": len(records),
            "seeds": seeds, "by_proposal": groups, "paired_proposal_effects": comparisons,
            "proposal_diagnostics": injection, "factorial_proxy_interaction": interaction,
            "factorial_effects": effects,
            "pairing_checks": {"complete_matrix": True, "same_query_goal_provisional_actions_and_cem_seed": True,
                               "same_ttt_support_and_update_across_proposals": True,
                               "same_fragments_and_noise_across_models": True,
                               "recorded_predictions_identical_across_proposals": True},
            "physical_success": None, "physical_regret": None,
            "interpretation": "Within-model predicted costs measure search, not physical performance. Frozen cross-scoring is also a model proxy.",
            "uncertainty": "Paired episode bootstrap after seed averaging per query; unadjusted exploratory 95% percentile intervals"}


def format_summary(report):
    summary = report["summary"]
    lines = [f"{report['stage']}: fixed-budget memory-action proposals", "",
             f"{summary['queries']} queries / {summary['episodes']} episodes; seeds {summary['seeds']}.", "",
             "| TTT | Proposal | Own-model SSE | Common Frozen SSE | Primary seconds | Injected slots |",
             "|---|---|---:|---:|---:|---:|"]
    for source, group in summary["by_proposal"].items():
        for method, row in group["methods"].items():
            search = row["search_diagnostics_not_physical_performance"]
            injected = summary["proposal_diagnostics"][source]["mean_injected_candidates"]
            lines.append(f"| {method} | {source} | {search['selected_model_goal_sse']:.5f} | "
                         f"{search['selected_plan_frozen_model_goal_sse']:.5f} | "
                         f"{search['primary_seconds_with_checks']:.3f} | {injected:.1f} |")
    horizon = report["config"]["planning"]["horizon"]
    lines += ["", "Recorded-action accuracy (one estimate per model; proposal copies are not independent observations):", "",
              "| TTT | Recorded endpoint MSE | Delta vs Frozen [95% CI] | Harm |",
              "|---|---:|---|---:|"]

    def ci_text(ci):
        return "disabled" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"

    for method, row in summary["by_proposal"]["gaussian"]["methods"].items():
        accuracy = row["recorded_action_prediction"][f"h{horizon}"]
        lines.append(f"| {method} | {accuracy['mse']:.7f} | {accuracy['delta_vs_frozen']:+.7f} "
                     f"{ci_text(accuracy['delta_ci95'])} | {accuracy['harm_fraction']:.1%} |")
    if summary["factorial_effects"]:
        lines += ["", "Factorial contrasts of selected plans scored by the common Frozen reference:", "",
                  "| Effect | TTT | Proposal / contrast | SSE difference [95% CI] |",
                  "|---|---|---|---|"]
        effects = summary["factorial_effects"]
        for field, label in (("ttt_minus_frozen_by_proposal", "TTT minus Frozen"),
                             ("combined_minus_frozen_gaussian", "Combined minus Frozen/Gaussian"),
                             ("difference_in_differences", "TTT x proposal interaction vs Frozen"),
                             ("support_selection_minus_uniform_by_proposal", "Retrieved minus random TTT"),
                             ("support_selection_difference_in_differences", "Retrieval x proposal interaction vs random TTT")):
            for method, comparisons in effects.get(field, {}).items():
                for name, estimate in comparisons.items():
                    lines.append(f"| {label} | {method} | {name} | {estimate['mean_difference']:+.6f} "
                                 f"{ci_text(estimate['ci95'])} |")
        lines += ["", effects["sign"], effects["interpretation"]]
    lines += ["", "Proposal effects within each model (own-model search score):", "",
              "| TTT | Proposal contrast | Own-model SSE difference [95% CI] |",
              "|---|---|---|"]
    for method, comparisons in summary["paired_proposal_effects"].items():
        for name, metrics in comparisons.items():
            estimate = metrics["selected_model_goal_sse"]
            lines.append(f"| {method} | {name} | {estimate['mean_difference']:+.6f} {ci_text(estimate['ci95'])} |")
    lines += ["", "Lower predicted SSE is a model-search diagnostic; no action is executed.",
              "Recorded-action prediction errors must be identical across proposal arms within a TTT method.",
              "Primary time charges proposal generation and shared TTT to each standalone call, plus CEM/state checks."]
    reuse = report.get("reuse", {})
    if reuse.get("reused_cem_calls"):
        lines += [f"Reused CEM calls: {reuse['reused_cem_calls']}; newly computed calls: {reuse['new_cem_calls']}.",
                  reuse["verification"], reuse["timing"]]
    goal = report.get("goal_retrieval", {})
    if goal.get("enabled"):
        lines += [f"Goal retrieval: weight {goal['goal_weight']:g}, achieved endpoint {goal['horizon']} blocks ahead, "
                  f"memory-only endpoint scale {goal['endpoint_scale']:.7g}; no goal enters TTT."]
    return "\n".join(lines)+"\n"
