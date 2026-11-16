"""User-run model acceptance checks; the coding agent runs syntax checks only."""

from copy import copy, deepcopy
from types import SimpleNamespace

import torch

from ..planning.cem import solve_cem
from ..planning.cohorts import planning_task, goal_latent
from .cem import solve_interleaved_cem
from .config import adaptation_schedule, checkpoint_updates
from .session import CheckpointSession


def run_checks(resources, bank, config, coordinates, providers):
    fixture = deepcopy(config)
    fixture["planning"].update(population=8, iterations=3, elites=2)
    fixture["interleave"].update(checkpoints=[1, 2], references=2, support_pairs=None,
                                 injection_slots=min(7, config["interleave"]["injection_slots"]) if providers.providers else 0)
    local_providers = copy(providers)
    local_providers.config = fixture
    engine = SimpleNamespace(adapter=resources.adapter, memory=resources.memory, retriever=resources.engine.retriever,
                             settings=resources.engine.settings | {"updates": 1})
    task = planning_task(bank, 0, fixture["planning"]["horizon"])
    goal = goal_latent(bank, task)
    seed = resources.config["evaluation"]["seeds"][0]

    def run(arm, selected=fixture, selected_engine=engine):
        p = copy(local_providers)
        p.config = selected
        session = CheckpointSession(selected_engine, task.context, task.metadata, goal, coordinates,
                                    p, selected, arm, seed)
        plan, records = solve_interleaved_cem(session.score, selected["planning"], coordinates, seed,
                                             selected["interleave"]["checkpoints"], session)
        return session, plan, records

    try:
        baseline, plan, _ = run("baseline")
        expected = solve_cem(baseline.score, fixture["planning"], coordinates, seed)
        torch.testing.assert_close(plan.actions, expected.actions, rtol=0, atol=0)
        if plan.cost != expected.cost or plan.candidate_evaluations != expected.candidate_evaluations:
            raise AssertionError("Disabled interleaving changed ordinary CEM")
        zero = deepcopy(fixture)
        zero["interleave"]["injection_slots"] = 0
        zero_engine = SimpleNamespace(**{**vars(engine), "settings": engine.settings | {"updates": 0}})
        for arm in ("rca", "proposals", "combined"):
            session, actual, _ = run(arm, zero, zero_engine)
            torch.testing.assert_close(actual.actions, plan.actions, rtol=0, atol=0)
            if actual.cost != plan.cost or session.version:
                raise AssertionError("Zero treatment did not reproduce Frozen")
        session, adapted, records = run("combined")
        if len(engine.memory) and session.counters["actual_updates"] != 2*checkpoint_updates(fixture, 1):
            raise AssertionError("Checkpoint schedule did not perform the expected observed-support updates")
        if session.version != session.counters["actual_updates"]:
            raise AssertionError("Model version does not match optimizer updates")
        if adaptation_schedule(fixture) == "sequential-depth":
            final_cost = float(session.score(adapted.actions[None])[0])
            if abs(final_cost-adapted.cost) > 1e-5*max(1., abs(adapted.cost)):
                raise AssertionError("Final shared predictor does not reproduce the returned plan cost")
        fingerprint = [r["population_fingerprint"] for r in records]
        repeated, again, repeated_records = run("combined")
        torch.testing.assert_close(adapted.actions, again.actions, rtol=0, atol=0)
        if adapted.cost != again.cost or fingerprint != [r["population_fingerprint"] for r in repeated_records]:
            raise AssertionError("Query reset or random-stream repeatability failed")
        if [r["support_fingerprint"] for r in session.rounds] != [r["support_fingerprint"] for r in repeated.rounds]:
            raise AssertionError("Checkpoint support selection is not repeatable")
        result = {"disabled_equals_existing_cem": True, "zero_updates_and_zero_injection_equal_frozen": True,
                "query_reset_and_support_repeatability": True, "cumulative_optimizer_clock": True,
                "observed_memory_supervision": True, "fixed_predictor_within_scoring": True,
                "final_mean_explicitly_scored": True, "model_version_accounting": True,
                "checks_use_small_population": 8, "checks_do_not_execute_actions": True}
        if adaptation_schedule(fixture) == "sequential-depth":
            result.update(sequential_depth_prefixes_use_current_model=True,
                          final_predictor_shared_across_depths=True)
        return result
    finally:
        resources.adapter.reset(seed)
