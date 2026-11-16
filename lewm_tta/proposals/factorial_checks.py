"""User-run checks for independent factors, paired estimates and reset isolation."""

from copy import deepcopy
from itertools import product

import torch

from ..planning.actions import provisional_actions, stream_seed
from ..planning.cem import score_sequences
from ..planning.shared_ttt import adapt_shared
from ..rollout.inference import fixed_predictor
from ..rollout.record_io import check_record_roundtrip
from .cem import solve_guided_cem
from .effects import COMMON_SCORE, factorial_effects
from .providers import proposal_bundle
from .reporting import paired_records


def statistical_checks(config):
    """Known effects and corrupted pairing must be distinguished before a run."""
    fixture = deepcopy(config)
    fixture["adaptation"].update(methods=["frozen", "fr-fixed"], updates=1)
    fixture["p1"]["evaluation"]["seeds"] = [11, 12]
    fixture["proposals"].update(sources=["gaussian", "uniform-fragments", "context-fragments", "goal-context-fragments"],
                                injection_slots=3, goal_weight=1.)
    fixture["evaluation"]["bootstrap_samples"] = 32
    records = []
    for query, seed, method, source in product(range(2), [11, 12], fixture["adaptation"]["methods"], fixture["proposals"]["sources"]):
        # Frozen: 10/9/8; adapted: 7/5/3. Expected TTT: -3/-4/-5;
        # combined: -5/-7; interactions: uniform -1, context -2, context-uniform -1.
        index = fixture["proposals"]["sources"].index(source)
        adapted = method != "frozen"
        score = 10-index-(3+index if adapted else 0)+query*2+seed*.1
        if adapted:
            score += (-.5 if seed == 11 else .5)*(query+1)
        records.append({"query": {"transition_id": str(query), "source_episode_position": query},
            "seed": seed, "method": method, "proposal_source": source, "scope": "last-block",
            "current_raw_row": 10, "goal_raw_row": 35, "target_raw_rows": [15, 20, 25, 30, 35],
            "provisional_seed": seed, "provisional_fingerprint": "shared-actions", "cem_seed": seed+1,
            "base_first_population_fingerprint": "shared-base",
            "cem_first_population_fingerprint": source, "cem_trace": [{}, {}],
            "selected_actions_normalized": [[score]], "selected_model_goal_sse": score,
            COMMON_SCORE: score, "recorded_action_mse_by_block": [.2 if adapted else .3],
            "adaptation": {"support_fingerprint": method, "reset_seed": seed,
                           "updates": int(adapted), "unique_support": int(adapted),
                           "support_presentations": int(adapted), "parameter_displacement_mse": .1 if adapted else 0.,
                           "skip_reason": None, "rounds": [], "support": []},
            "proposal": {"source": source, "fragment_fingerprint": source, "injection_fingerprint": source,
                         "goal_query_fingerprint": "shared-goal" if source == "goal-context-fragments" else None,
                         "injected_slots": int(index > 0), "slot_provenance": [{}] if index else []}})

    def audit(rows, selected):
        lookup, ids = paired_records(rows, selected)
        return factorial_effects(lookup, ids, selected)

    effects = audit(records, fixture)
    expected = {"ttt_minus_frozen_by_proposal": {"gaussian": -3, "uniform-fragments": -4, "context-fragments": -5,
                                               "goal-context-fragments": -6},
                "combined_minus_frozen_gaussian": {"uniform-fragments": -5, "context-fragments": -7,
                                                   "goal-context-fragments": -9},
                "difference_in_differences": {"uniform-fragments_minus_gaussian": -1,
                                             "context-fragments_minus_gaussian": -2,
                                             "context-fragments_minus_uniform-fragments": -1,
                                             "goal-context-fragments_minus_gaussian": -3,
                                             "goal-context-fragments_minus_context-fragments": -1,
                                             "goal-context-fragments_minus_uniform-fragments": -2}}
    for field, comparisons in expected.items():
        for name, value in comparisons.items():
            result = effects[field]["fr-fixed"][name]
            assert abs(result["mean_difference"]-value) < 1e-10
            assert all(abs(bound-value) < 1e-10 for bound in result["ci95"])

    def reject(rows):
        try:
            audit(rows, fixture)
        except ValueError:
            return
        raise AssertionError("A corrupted or incomplete factorial pairing was accepted")

    reject(records[:-1])
    reject(records+[records[0]])
    for field, value in (("provisional_fingerprint", "changed"), ("goal_raw_row", 36),
                         ("recorded_action_mse_by_block", [.99])):
        changed = deepcopy(records)
        changed[-1][field] = value
        reject(changed)
    for section, field, value in (("adaptation", "support_fingerprint", "changed"),
                                  ("adaptation", "updates", 2),
                                  ("proposal", "fragment_fingerprint", "changed"),
                                  ("proposal", "injection_fingerprint", "changed")):
        changed = deepcopy(records)
        changed[-1][section][field] = value
        reject(changed)
    changed = deepcopy(records)
    changed[-1]["proposal"]["goal_query_fingerprint"] = "different-goal"
    reject(changed)
    changed = deepcopy(records)
    changed[-1]["cem_trace"][1]["directly_injected_candidates"] = 1
    reject(changed)
    parity_keys = ("cem_first_population_fingerprint", "selected_actions_normalized",
                   "selected_model_goal_sse", COMMON_SCORE, "recorded_action_mse_by_block")
    for factor in ("injection", "updates"):
        disabled, collapsed = deepcopy(fixture), deepcopy(records)
        if factor == "injection":
            disabled["proposals"]["injection_slots"] = 0
        else:
            disabled["adaptation"]["updates"] = 0
        original = {(r["query"]["transition_id"], r["seed"], r["method"], r["proposal_source"]): r for r in records}
        for row in collapsed:
            method = row["method"] if factor == "injection" else "frozen"
            source = "gaussian" if factor == "injection" else row["proposal_source"]
            reference = original[row["query"]["transition_id"], row["seed"], method, source]
            for key in parity_keys:
                row[key] = deepcopy(reference[key])
            if factor == "injection":
                row["proposal"].update(injected_slots=0, slot_provenance=[])
            else:
                row["adaptation"] = deepcopy(reference["adaptation"])
        audit(collapsed, disabled)
        collapsed[-1]["selected_model_goal_sse"] += .1
        try:
            audit(collapsed, disabled)
        except ValueError:
            pass
        else:
            raise AssertionError("Disabled factor failed to enforce reference plan/score parity")
    check_record_roundtrip(records, fixture, audit)
    return {"known_simple_combined_and_interaction_effects": True,
            "seed_averaging_and_paired_episode_intervals": True,
            "incomplete_duplicate_and_confounding_records_rejected": True,
            "zero_injection_and_update_factor_collapse_enforced": True,
            "compact_full_and_no_record_effect_parity": True}


def real_factorial_check(resources, task, goal, config, coordinates, provider, source, injection, proposal):
    """Search order and model adaptation cannot change the other factor.

    Uses the first validation context, a small search budget and one configured
    TTT treatment. The full evaluation additionally audits every saved cell.
    """
    if config["comparison"] != "factorial":
        return {"applicable": False}
    adapter, settings = resources.adapter, config["planning"]
    method = next(m for m in config["adaptation"]["methods"] if m != "frozen")
    seed = resources.config["evaluation"]["seeds"][0]
    provisional = provisional_actions(coordinates, settings["provisional_actions"], settings["initial_std"],
                                     stream_seed(seed, task.metadata["transition_id"], "provisional"))
    cem_seed = stream_seed(seed, task.metadata["transition_id"], "factorial-check-cem")
    small = {**settings, "population": max(8, config["proposals"]["injection_slots"]+1),
             "iterations": 2, "elites": 4}

    def solve(selected):
        with fixed_predictor(adapter):
            return solve_guided_cem(lambda actions: score_sequences(
                adapter, task.context, goal, actions, settings["scoring_batch_size"]),
                small, coordinates, cem_seed, selected)

    try:
        adapt_shared(resources.engine, task.context, task.metadata, provisional, "frozen", seed, settings["support_scan_limit"])
        frozen = solve(injection)
        adapted = adapt_shared(resources.engine, task.context, task.metadata, provisional, method, seed, settings["support_scan_limit"])
        # A deliberate post-TTT re-selection is only a check, never the experimental policy.
        _, repeated = proposal_bundle(provider, adapter, task.context, task.metadata, source, seed, goal=goal)
        for field in ("fragment_fingerprint", "injection_fingerprint", "goal_query_fingerprint"):
            assert proposal.get(field) == repeated.get(field)
        forward = {label: solve(value) for label, value in (("gaussian", None), (source, injection))}
        backward = {label: solve(value) for label, value in ((source, injection), ("gaussian", None))}
        for label in forward:
            left, right = forward[label], backward[label]
            torch.testing.assert_close(left.plan.actions, right.plan.actions, rtol=0, atol=0)
            assert left.plan.cost == right.plan.cost
            assert left.plan.first_population_fingerprint == right.plan.first_population_fingerprint
            assert left.base_first_population_fingerprint == frozen.base_first_population_fingerprint
        with fixed_predictor(adapter):
            prediction = adapter.predict(task.context, provisional[:1])
        again = adapt_shared(resources.engine, task.context, task.metadata, provisional, method, seed, settings["support_scan_limit"])
        assert adapted["support_fingerprint"] == again["support_fingerprint"]
        with fixed_predictor(adapter):
            torch.testing.assert_close(prediction, adapter.predict(task.context, provisional[:1]), rtol=0, atol=0)
        adapt_shared(resources.engine, task.context, task.metadata, provisional, "frozen", seed, settings["support_scan_limit"])
        reset = solve(injection)
        torch.testing.assert_close(frozen.plan.actions, reset.plan.actions, rtol=0, atol=0)
        assert frozen.plan.cost == reset.plan.cost
        return {"applicable": True, "method": method, "proposal_source": source,
                "post_ttt_fragment_and_noise_identity": True, "proposal_search_order_independence": True,
                "shared_adaptation_repeatability_after_search": True, "guided_plan_reset_parity": True,
                "population": small["population"], "iterations": small["iterations"], "planning_calls": 6}
    finally:
        adapter.reset(stream_seed(seed, task.metadata["transition_id"], "model-reset"))
