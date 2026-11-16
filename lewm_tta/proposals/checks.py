"""User-run acceptance checks for proposal alignment, exact search and CEM parity."""

from types import SimpleNamespace
import numpy as np
import torch

from ..frame_bank import FrameLatentBank
from ..indexed_keys import KeyLayout
from ..model import Context
from ..planning.actions import ActionCoordinates, stream_seed
from ..planning.cem import solve_cem, score_sequences
from ..planning.checks import run_checks as planning_checks
from ..planning.cohorts import planning_task, goal_latent
from ..rollout.inference import fixed_predictor
from .cem import solve_guided_cem
from .context_search import ContextSearch, exact_prefix
from .fragments import FragmentStore
from .injection import Injection, build_injection
from .providers import proposal_bundle
from .factorial_checks import statistical_checks, real_factorial_check
from .goal_checks import run_goal_checks


def synthetic_checks(resources):
    device = resources.adapter.device
    manifest = {"source": {"path": "synthetic"}, "data_config": {"history": 2, "frameskip": 2},
                "episode_metadata": {"offsets": [0, 20, 40], "lengths": [20]*3, "ids": [100, 101, 102]},
                "episodes": {"memory": [0, 1], "validation": [2], "test": []}}
    store = SimpleNamespace(latents=torch.arange(180, dtype=torch.float32, device=device).reshape(60, 3)/100,
                            actions=torch.arange(120, dtype=torch.float32, device=device).reshape(60, 2)/200)
    starts = np.asarray([0, 1, 11, 12, 20, 21, 31, 32], dtype=np.int64)
    memory = FrameLatentBank(store, manifest, "memory", starts, {})
    coordinates = ActionCoordinates({"mean": [0, 0], "std": [1, 1]},
                                     {"raw_action_low": [-1, -1], "raw_action_high": [1, 1]}, 2, device)
    fragments = FragmentStore(memory, 3, coordinates)
    assert fragments.eligible.tolist() == [True, True, True, False, True, True, True, False]
    torch.testing.assert_close(fragments.actions_at(0), store.actions[2:8].reshape(3, 4), rtol=0, atol=0)
    selected = fragments.select(range(8), 4, 0, 2, 6)
    assert [row["memory_index"] for row in selected.sources] == [4, 6]
    assert selected.diagnostics["fragment_shortfall"] == 2
    assert all(row["source_episode_position"] == 1 for row in selected.sources)
    assert fragments.select(range(8), 4, 0, 1, 6).diagnostics["selected_fragments"] == 1
    try:
        fragments.actions_at(3)
    except ValueError:
        pass
    else:
        raise AssertionError("Fragment accepted an unavailable final observation")
    store.actions[22] = 5
    rejected = fragments.select([4], 1, 0, 2, 6)
    assert rejected.diagnostics["rejections"]["invalid_or_out_of_bounds_actions"] == 1
    store.actions[22] = torch.tensor([44, 45], device=device)/200

    scales = {"context": 2., "past_actions": 3., "action": 7., "successor": 4.}
    calibration = {"scales": scales}
    key_settings = {"beta_action": 1., "beta_successor": 1.}
    layout = KeyLayout.from_memory(memory, calibration, key_settings)
    context = Context(memory.contexts[:1], memory.past_actions[:1])
    torch_search = ContextSearch(SimpleNamespace(memory=memory, scales=scales), chunk_rows=3)
    direct = torch_search.distances(context)
    base = resources.engine.retriever
    backend = "torch"
    if hasattr(base, "runtime"):
        keys = layout.transform("forward", {"context": memory.contexts[:], "past_actions": memory.past_actions[:],
                                             "action": memory.actions[:]})
        local = SimpleNamespace(memory=memory, layout=layout, scales=scales, keys={"forward": keys}, runtime=base.runtime)
        search = ContextSearch(local, chunk_rows=3)
        found = search.distances(context)
        torch.testing.assert_close(found, direct, rtol=1e-4, atol=1e-6)
        prefix_width = layout.dimensions["context"]+layout.dimensions["past_actions"]
        local.keys["forward"][:, prefix_width:] += 1000
        torch.testing.assert_close(search.distances(context), found, rtol=0, atol=0)
        backend = "cuvs"
    tied = torch.tensor([1., 0., 0., 2., 0.], device=device)
    assert exact_prefix(tied, np.asarray([0, 1, 2, 3, 4]), 2) == [1, 2]
    # Fewer sources use their nominal quotas, leaving other slots Gaussian.
    proposal_settings = {"fragments": 4, "injection_slots": 8, "noise_std": .1}
    injection = build_injection(selected, proposal_settings, 7)
    assert injection.slots == [1, 2, 3, 4] and injection.requested_slots == 8
    torch.testing.assert_close(injection.unbounded_actions[0], selected.actions[0], rtol=0, atol=0)
    torch.testing.assert_close(injection.unbounded_actions[2], selected.actions[1], rtol=0, atol=0)

    settings = {"horizon": 3, "population": 12, "iterations": 3, "elites": 3,
                "initial_std": 1., "minimum_std": 0.}
    target = torch.full((3, 4), .1, device=device)
    score = lambda actions: (actions-target).square().flatten(1).sum(1)
    warm = target[:1]
    plain = solve_cem(score, settings, coordinates, 19, initial_mean=warm)
    disabled = solve_guided_cem(score, settings, coordinates, 19, initial_mean=warm)
    torch.testing.assert_close(plain.actions, disabled.plan.actions, rtol=0, atol=0)
    assert plain.trace == disabled.plan.trace and plain.cost == disabled.plan.cost
    # Exercise the full new loop with a nonempty injection equal to the ordinary
    # samples. Every later elite update must reproduce the original solver.
    mean = torch.zeros(3, 4, device=device)
    mean[:1] = warm
    normal = torch.randn(12, 3, 4, generator=torch.Generator().manual_seed(19)).to(device)+mean
    identity = Injection([1, 2], normal[1:3], [], 2)
    same = solve_guided_cem(score, settings, coordinates, 19, identity, initial_mean=warm)
    torch.testing.assert_close(plain.actions, same.plan.actions, rtol=0, atol=0)
    assert plain.cost == same.plan.cost and same.plan.candidate_evaluations == 37
    for left, right in zip(plain.trace, same.plan.trace):
        assert all(right[key] == value for key, value in left.items())
    sampled = []
    def recording_score(actions):
        sampled.append(actions.clone())
        return score(actions)
    guided = solve_guided_cem(recording_score, settings, coordinates, 19, injection, initial_mean=warm)
    normal[0] = mean
    keep = [0]+list(range(5, 12))
    torch.testing.assert_close(sampled[0][keep], coordinates.clip(normal)[keep], rtol=0, atol=0)
    torch.testing.assert_close(sampled[0][injection.slots], coordinates.clip(injection.unbounded_actions), rtol=0, atol=0)
    assert guided.plan.candidate_evaluations == plain.candidate_evaluations
    assert all(step["directly_injected_candidates"] == 0 for step in guided.plan.trace[1:])
    assert len(sampled) == settings["iterations"]+1
    assert abs(guided.plan.cost-float(score(guided.plan.actions[None])[0])) < 1e-6
    return {"episode_boundaries_and_action_alignment": True, "query_episode_exclusion": True,
            "source_cap_and_spacing": True, "invalid_source_actions_rejected": True,
            "context_metric_parity_backend": backend, "stable_ties": True,
            "future_action_columns_excluded_from_cuvs_context_metric": hasattr(base, "runtime"),
            "shortage_preserves_gaussian_slots": True, "exact_seed_preserved": True,
            "no_injection_and_identity_injection_equal_p3a": True,
            "population_and_warm_start_preserved": True, "first_iteration_only": True,
            "returned_mean_scored": True}


def run_checks(resources, bank, config, coordinates, provider):
    statistics = statistical_checks(config)
    base_checks = planning_checks(resources, bank, config, coordinates)
    fixtures = synthetic_checks(resources)
    goal_checks = run_goal_checks(resources)
    task = planning_task(bank, 0, config["planning"]["horizon"])
    goal = goal_latent(bank, task)
    seed = resources.config["evaluation"]["seeds"][0]
    source = ("goal-context-fragments" if "goal-context-fragments" in config["proposals"]["sources"] else
              "context-fragments" if "context-fragments" in config["proposals"]["sources"] else
              next((s for s in config["proposals"]["sources"] if s != "gaussian"), "gaussian"))
    injection, record = proposal_bundle(provider, resources.adapter, task.context, task.metadata, source, seed, goal=goal)
    again, repeated = proposal_bundle(provider, resources.adapter, task.context, task.metadata, source, seed, goal=goal)
    assert record["fragment_fingerprint"] == repeated["fragment_fingerprint"]
    assert record["injection_fingerprint"] == repeated["injection_fingerprint"]
    if provider.goal_search and provider.goal_search.weight == 0:
        _, control = proposal_bundle(provider, resources.adapter, task.context, task.metadata, "context-fragments", seed)
        assert record["injection_fingerprint"] == control["injection_fingerprint"]
    resources.adapter.reset(stream_seed(seed, task.metadata["transition_id"], "model-reset"))
    with fixed_predictor(resources.adapter):
        horizon = config["planning"]["horizon"]
        small = {**config["planning"], "population": max(8, config["proposals"]["injection_slots"]+1),
                 "iterations": 2, "elites": 4}
        guided = solve_guided_cem(lambda actions: score_sequences(resources.adapter, task.context, goal, actions,
            config["planning"]["scoring_batch_size"]), small, coordinates, seed, injection)
        assert guided.plan.actions.shape == (horizon, coordinates.width)
    return {"shared_planning": base_checks, "proposal_fixtures": fixtures,
            "factorial_statistics": statistics,
            "goal_retrieval": goal_checks,
            "factorial_model_pairing": real_factorial_check(
                resources, task, goal, config, coordinates, provider, source, injection, record),
            "real_memory_fragment_repeatability": True, "real_memory_guided_cem_fixed_state": True,
            "real_fixture_source": source, "real_fixture_injected_slots": record["injected_slots"]}
