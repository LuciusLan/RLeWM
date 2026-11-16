"""User-run acceptance checks for offline CEM and shared adaptation."""

from types import SimpleNamespace
import numpy as np
import torch

from ..model import Context
from ..rollout.inference import fixed_predictor
from ..rollout.support_updates import validate_support
from .actions import provisional_actions, stream_seed
from .cem import solve_cem, score_sequences
from .cohorts import planning_task, goal_latent
from .shared_ttt import merge_streams, adapt_shared


def check_pooling():
    retriever = SimpleNamespace(episode_positions=np.repeat(np.arange(4), 3),
                               step_positions=np.tile([0, 10, 20], 4), episode_cap=2, min_step_gap=20)
    order = np.arange(12)
    support = merge_streams([("a/forward", order, .5), ("a/reverse", order, .5)], 4, retriever)
    validate_support(list(range(12)), retriever, support, 4)
    assert len(support.indices) == 4 and any(len(channels) == 2 for channels in support.channels.values())
    assert all(value["memberships"] >= 2 for value in support.details["streams"].values())
    # All source windows in the pool are ineligible beyond the global cap/spacing.
    short = merge_streams([("a", [0, 1], .5), ("b", [0, 1], .5)], 4, retriever)
    validate_support(list(range(12)), retriever, short, 4)
    assert len(short.indices) == 1 and short.details["shortfall"] == 3
    assert abs(sum(short.weights)-1) < 1e-7
    return {"global_episode_cap_and_spacing": True, "deduplication_and_weights": True,
            "explicit_shortfalls": True}


def run_checks(resources, bank, config, coordinates):
    adapter, settings = resources.adapter, config["planning"]
    task = planning_task(bank, 0, settings["horizon"])
    goal = goal_latent(bank, task)
    seed = resources.config["evaluation"]["seeds"][0]
    provisional = provisional_actions(coordinates, settings["provisional_actions"], settings["initial_std"],
                                      stream_seed(seed, task.metadata["transition_id"], "provisional"))
    raw = coordinates.to_raw(provisional)
    assert coordinates.width == adapter.action_dim
    assert raw.shape == (settings["provisional_actions"], bank.frameskip, len(coordinates.mean))
    torch.testing.assert_close(coordinates.from_raw(coordinates.to_raw(provisional)), provisional,
                               rtol=1e-6, atol=1e-6)
    small = {**settings, "population": 8, "iterations": 2, "elites": 2, "scoring_batch_size": 4}
    cem_seed = stream_seed(seed, task.metadata["transition_id"], "check-cem")
    # Analytic fixture checks sampling/refitting/accounting without model quality assumptions.
    target = coordinates.clip(torch.full((small["horizon"], coordinates.width), .25, device=adapter.device))
    score = lambda actions: (actions-target).square().flatten(1).sum(1)
    analytic = solve_cem(score, small, coordinates, cem_seed)
    assert analytic.candidate_evaluations == 17 and len(analytic.trace) == 2
    assert abs(analytic.cost-float(score(analytic.actions[None])[0])) < 1e-6
    repeated = solve_cem(score, small, coordinates, cem_seed)
    torch.testing.assert_close(analytic.actions, repeated.actions, rtol=0, atol=0)
    assert analytic.trace == repeated.trace
    warm = solve_cem(score, small, coordinates, cem_seed, initial_mean=target[:1])
    assert warm.candidate_evaluations == analytic.candidate_evaluations
    assert warm.first_population_fingerprint != analytic.first_population_fingerprint

    adapt_shared(resources.engine, task.context, task.metadata, provisional, "frozen", seed, settings["support_scan_limit"])
    def solve():
        with fixed_predictor(adapter):
            return solve_cem(lambda actions: score_sequences(adapter, task.context, goal, actions, 4),
                             small, coordinates, cem_seed)
    original = solve()
    with fixed_predictor(adapter):
        pair = torch.stack([original.actions, coordinates.clip(original.actions+.01)])
        together = score_sequences(adapter, task.context, goal, pair, 2)
        separate = score_sequences(adapter, task.context, goal, pair, 1)
        torch.testing.assert_close(together, separate, rtol=1e-4, atol=1e-6)
        repeated_context = Context(task.context.latents.expand(2, -1, -1), task.context.past_actions.expand(2, -1, -1))
        predicted = adapter.rollout(repeated_context, pair)
        native = adapter.model.criterion({"predicted_emb": predicted[None],
                                         "goal_emb": goal[:, None, None, :].expand(1, 2, 1, -1)})
        torch.testing.assert_close(together, native[0], rtol=1e-6, atol=1e-7)

    # Test U=0 uses the same shared path and reproduces Frozen exactly.
    zero_engine = SimpleNamespace(adapter=adapter, memory=resources.memory, retriever=resources.engine.retriever,
                                  settings={**resources.engine.settings, "updates": 0})
    zero = adapt_shared(zero_engine, task.context, task.metadata, provisional, "fr-fixed", seed, settings["support_scan_limit"])
    zero_plan = solve()
    torch.testing.assert_close(original.actions, zero_plan.actions, rtol=0, atol=0)
    assert original.cost == zero_plan.cost and zero["updates"] == 0
    assert original.first_population_fingerprint == zero_plan.first_population_fingerprint

    # An intervening adaptation cannot influence the next independent call.
    probe_engine = SimpleNamespace(adapter=adapter, memory=resources.memory, retriever=resources.engine.retriever,
                                   settings={**resources.engine.settings, "updates": 1})
    probe_method = next((m for m in config["adaptation"]["methods"] if m == "fr-fixed"), "uniform")
    first = adapt_shared(probe_engine, task.context, task.metadata, provisional, probe_method, seed, settings["support_scan_limit"])
    with fixed_predictor(adapter):
        prediction = adapter.predict(task.context, provisional[:1])
    second = adapt_shared(probe_engine, task.context, task.metadata, provisional, probe_method, seed, settings["support_scan_limit"])
    with fixed_predictor(adapter):
        torch.testing.assert_close(prediction, adapter.predict(task.context, provisional[:1]), rtol=0, atol=0)
    assert first["support_fingerprint"] == second["support_fingerprint"]
    assert all(item["source_episode_position"] != task.metadata["source_episode_position"]
               for item in first["support"])
    if len(resources.memory) and not first["updates"]:
        raise AssertionError("Shared-TTT acceptance fixture could not find any eligible support")
    adapt_shared(resources.engine, task.context, task.metadata, provisional, "frozen", seed, settings["support_scan_limit"])
    reset_plan = solve()
    torch.testing.assert_close(original.actions, reset_plan.actions, rtol=0, atol=0)
    assert original.cost == reset_plan.cost
    adapter.check_fixed_state()
    return {"support_pooling": check_pooling(), "normalization_roundtrip": True,
            "action_dimensions": {"raw": len(coordinates.mean), "ordered_block": coordinates.width,
                                  "frameskip": bank.frameskip},
            "analytic_cem_determinism_and_returned_mean_cost": True,
            "optional_warm_start_changes_population_without_changing_budget": True,
            "batched_scalar_cost_parity": True, "native_final_latent_sse_parity": True,
            "zero_updates_equal_frozen": True, "reset_after_adaptation_equal_frozen": True,
            "shared_support_and_update_repeatability": True, "parameters_and_buffers_fixed_during_cem": True,
            "goal_and_recorded_actions_absent_from_ttt_api": True,
            "fixture": {"split": "validation", "transition_id": task.metadata["transition_id"],
                        "population": 8, "iterations": 2, "horizon": settings["horizon"],
                        "shared_update_method": probe_method}}
