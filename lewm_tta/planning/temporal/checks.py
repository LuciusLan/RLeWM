"""User-run numerical acceptance checks; never executed by the coding agent."""

from copy import deepcopy
from types import SimpleNamespace

import torch

from ...rollout.inference import fixed_predictor
from ...rollout.temporal import run_scheduled_rollout
from ..cem import solve_cem
from ..cohorts import planning_task, goal_latent
from .config import nominal_budget
from .rollouts import reference_actions
from .scoring import TemporalScorer


def run_checks(resources, bank, config, coordinates):
    fixture = deepcopy(config)
    fixture["planning"].update(horizon=min(2, config["planning"]["horizon"]), population=4, iterations=1, elites=2)
    fixture["adaptation"]["updates"] = 1
    # Share only immutable cache/retriever and the explicitly reset adapter.
    engine = SimpleNamespace(adapter=resources.adapter, memory=resources.memory, retriever=resources.engine.retriever,
                             settings=resources.engine.settings | {"updates": 1})
    adapter = engine.adapter
    task = planning_task(bank, 0, fixture["planning"]["horizon"])
    goal = goal_latent(bank, task)
    seed = resources.config["evaluation"]["seeds"][0]
    references = reference_actions(coordinates, fixture, task.metadata, seed)
    actions = references[:2]

    def build(mode, chosen=fixture, chosen_engine=engine):
        return TemporalScorer(chosen_engine, task.context, task.metadata, references, mode, seed, chosen)

    try:
        frozen = build("frozen")
        expected = frozen.rollout(actions)
        with fixed_predictor(adapter):
            native = adapter.rollout(SimpleNamespace(
                latents=task.context.latents.expand(len(actions), -1, -1),
                past_actions=task.context.past_actions.expand(len(actions), -1, -1)), actions)
        torch.testing.assert_close(expected, native, rtol=1e-5, atol=1e-7)
        initial, shared = build("shared-initial"), build("shared-depth")
        assert initial.snapshots[0].fingerprint == shared.snapshots[0].fingerprint
        assert initial.setup["support_fingerprint"] == shared.setup["positions"][0]["support_fingerprint"]
        shared_values = shared.rollout(actions)
        candidate = build("candidate-temporal")
        # The real CEM callback is under no_grad. Loss construction/backward
        # must still work, and a candidate cannot inherit another's state.
        with torch.no_grad():
            candidate_values = candidate.rollout(actions)
            reversed_values = candidate.rollout(actions.flip(0)).flip(0)
            separate = torch.cat([candidate.rollout(sequence) for sequence in actions.split(1)])
        torch.testing.assert_close(candidate_values, reversed_values, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(candidate_values, separate, rtol=1e-6, atol=1e-7)
        # Intervening candidate updates must not change a fixed depth scorer.
        torch.testing.assert_close(shared.rollout(actions), shared_values, rtol=0, atol=0)
        torch.testing.assert_close(shared.rollout(actions.flip(0)).flip(0), shared_values, rtol=1e-5, atol=1e-7)
        query = SimpleNamespace(context=task.context, metadata=task.metadata, action=actions[:1, 0])
        schedule = {"placement": "each-step", "updates": 1, "persistence": fixture["temporal"]["persistence"]}
        p2, _ = run_scheduled_rollout(engine, query, actions[:1], fixture["temporal"]["selector"], seed, schedule)
        torch.testing.assert_close(p2, candidate_values[:1], rtol=1e-6, atol=1e-7)
        # Every mode becomes exactly the same mathematical Frozen rollout at U0.
        zero = deepcopy(fixture)
        zero["adaptation"]["updates"] = 0
        zero_engine = SimpleNamespace(**{**vars(engine), "settings": engine.settings | {"updates": 0}})
        for mode in ("shared-initial", "shared-depth", "candidate-temporal"):
            torch.testing.assert_close(build(mode, zero, zero_engine).rollout(actions), expected, rtol=1e-5, atol=1e-7)
        scorer = build("candidate-temporal")
        plan = solve_cem(lambda blocks: scorer.costs(blocks, goal), fixture["planning"], coordinates, seed)
        primary = scorer.accounting()
        assert primary["requested_updates"] == nominal_budget(fixture, "candidate-temporal")["requested_gradient_updates"]
        assert primary["sequence_evaluations"] == plan.candidate_evaluations
        torch.testing.assert_close(scorer.costs(plan.actions[None], goal), torch.tensor([plan.cost], device=goal.device), rtol=1e-5, atol=1e-6)
        assert engine.settings == resources.engine.settings | {"updates": 1}
        return {"frozen_native_rollout_parity": True, "first_depth_matches_existing_shared_ttt": True,
                "candidate_temporal_matches_p2": True, "candidate_order_and_batch_independence": True,
                "candidate_backward_works_inside_cem_no_grad": True, "shared_depth_snapshots_restore_after_other_treatments": True,
                "zero_updates_equal_frozen_all_modes": True, "final_mean_reproducibility": True,
                "primary_counts_include_returned_mean": True, "engine_settings_preserved": True}
    finally:
        adapter.reset(seed)
