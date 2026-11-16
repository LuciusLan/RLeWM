"""Paired factorial contrasts evaluated on one common, explicitly model-only scale."""

import numpy as np

from ..planning.reporting import interval


COMMON_SCORE = "selected_plan_frozen_model_goal_sse"


def proposal_pairs(sources):
    pairs = [(source, "gaussian") for source in sources if source != "gaussian"]
    if "context-fragments" in sources and "uniform-fragments" in sources:
        pairs.append(("context-fragments", "uniform-fragments"))
    if "goal-context-fragments" in sources:
        pairs += [("goal-context-fragments", reference) for reference in ("context-fragments", "uniform-fragments")
                  if reference in sources]
    return pairs


def factorial_effects(lookup, ids, config):
    """Average seeds first, construct paired contrasts, then bootstrap episodes.

    Each cell's own-model cost uses a different scoring function when TTT is
    changed. Only the shared Frozen reference is used for cross-model effects.
    This reference is a diagnostic, not an observed counterfactual outcome.
    """
    seeds = config["p1"]["evaluation"]["seeds"]
    methods = [method for method in config["adaptation"]["methods"] if method != "frozen"]
    sources = config["proposals"]["sources"]
    if not methods:
        return {}
    episodes = np.asarray([lookup[q, seeds[0], "frozen", "gaussian"]["query"]["source_episode_position"] for q in ids])
    draws = config["evaluation"]["bootstrap_samples"]
    values = {(method, source): np.asarray([
        [lookup[q, seed, method, source][COMMON_SCORE] for seed in seeds] for q in ids]).mean(1)
        for method in ["frozen"]+methods for source in sources}
    if any(not np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite common-reference planning scores")

    def estimate(delta):
        return {"mean_difference": float(delta.mean()), "ci95": interval(delta, episodes, draws)}

    ttt, combined, interactions = {}, {}, {}
    for method in methods:
        ttt[method] = {source: estimate(values[method, source]-values["frozen", source]) for source in sources}
        combined[method] = {source: estimate(values[method, source]-values["frozen", "gaussian"])
                            for source in sources if source != "gaussian"}
        interactions[method] = {
            f"{left}_minus_{right}": estimate(
                values[method, left]-values[method, right]
                -values["frozen", left]+values["frozen", right])
            for left, right in proposal_pairs(sources)}
    selection, selection_interactions = {}, {}
    if "uniform" in methods:
        for method in methods:
            if method == "uniform":
                continue
            selection[method] = {
                source: estimate(values[method, source]-values["uniform", source]) for source in sources}
            selection_interactions[method] = {
                f"{left}_minus_{right}": estimate(
                    values[method, left]-values[method, right]
                    -values["uniform", left]+values["uniform", right])
                for left, right in proposal_pairs(sources)}
    return {"metric": COMMON_SCORE, "ttt_minus_frozen_by_proposal": ttt,
            "combined_minus_frozen_gaussian": combined, "difference_in_differences": interactions,
            "support_selection_minus_uniform_by_proposal": selection,
            "support_selection_difference_in_differences": selection_interactions,
            "sign": "Negative differences lower the common Frozen score; a negative interaction exceeds additive score reductions.",
            "interpretation": "The Frozen reference can favor its own plans; these contrasts do not identify physical benefit or synergy.",
            "physical_interaction": None}
