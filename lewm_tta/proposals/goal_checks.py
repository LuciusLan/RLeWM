"""User-run alignment, goal sensitivity, calibration and exact-metric fixtures."""

from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from ..frame_bank import FrameLatentBank
from ..indexed_keys import KeyLayout
from ..model import Context
from ..planning.actions import ActionCoordinates
from ..runtime import read_json, write_json
from .context_search import ContextSearch
from .fragments import FragmentStore, FragmentProvider
from .goal_search import GoalContextSearch
from .injection import build_injection


def run_goal_checks(resources):
    device = resources.adapter.device
    manifest = {"source": {"path": "synthetic"}, "data_config": {"history": 2, "frameskip": 2},
                "episode_metadata": {"offsets": [0, 20, 40], "lengths": [20]*3, "ids": [100, 101, 102]},
                "episodes": {"memory": [0, 1], "validation": [2], "test": []}}
    latents = torch.zeros(60, 3, device=device)
    latents[8:12] = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]], device=device)
    latents[28:32] = latents[8:12]
    actions = torch.zeros(60, 2, device=device)
    actions[8] = .2  # Distinguishable fragments, identical past actions for anchors 0..3.
    store = SimpleNamespace(latents=latents, actions=actions)
    starts = np.asarray([0, 1, 2, 3, 12, 20, 21, 22, 23, 32], dtype=np.int64)
    memory = FrameLatentBank(store, manifest, "memory", starts, {})
    coordinates = ActionCoordinates({"mean": [0, 0], "std": [1, 1]},
        {"raw_action_low": [-1, -1], "raw_action_high": [1, 1]}, 2, device)
    fragments = FragmentStore(memory, 3, coordinates)
    settings = {"goal_weight": 1., "goal_calibration_pairs": 64, "goal_calibration_seed": 7,
                "goal_distance_floor": 1e-8, "context_chunk_rows": 3, "max_scan_candidates": 10,
                "fragments": 2, "max_fragments_per_episode": 2, "min_start_step_gap": 0,
                "injection_slots": 4, "noise_std": .1}
    scales = {"context": 2., "past_actions": 3., "action": 4., "successor": 5.}
    base = SimpleNamespace(memory=memory, scales=scales)
    context_search = ContextSearch(base, settings["context_chunk_rows"])
    context = Context(memory.contexts[:1], memory.past_actions[:1])
    goal = latents[8:9].clone()
    eligible = np.flatnonzero(fragments.eligible)
    assert fragments.final_rows[:5].tolist() == [8, 9, 10, 11, 20]
    assert not fragments.eligible[4] and not fragments.eligible[-1]

    with TemporaryDirectory(prefix="lewm-goal-check-") as directory:
        search = GoalContextSearch(context_search, fragments, settings, directory)
        repeated = GoalContextSearch(context_search, fragments, settings, directory)
        assert search.calibration == repeated.calibration
        combined = search.distances(context, goal)
        indices = torch.as_tensor(eligible, device=device)
        expected = context_search.distances(context)[indices]+search.endpoint_distances(goal, eligible)
        torch.testing.assert_close(combined[indices], expected, rtol=1e-6, atol=1e-7)
        assert torch.isinf(combined[4]) and torch.isinf(combined[-1])
        # Explicit scaled concatenation must implement the same metric.
        layout = KeyLayout.from_memory(memory, {"scales": scales}, {"beta_action": 1., "beta_successor": 1.})
        prefix_width = layout.dimensions["context"]+layout.dimensions["past_actions"]
        keys = layout.transform("forward", {"context": memory.contexts[:],
            "past_actions": memory.past_actions[:], "action": memory.actions[:]})
        endpoint_rows = torch.as_tensor(fragments.final_rows[eligible], device=device)
        factor = (settings["goal_weight"]/(3*search.scale))**.5
        joint_keys = torch.cat([keys[indices, :prefix_width], latents[endpoint_rows]*factor], 1)
        query_prefix = torch.cat([context.latents.reshape(1, -1)*layout.factor("context"),
                                  context.past_actions.reshape(1, -1)*layout.factor("past_actions")], 1)
        joint_query = torch.cat([query_prefix, goal*factor], 1)
        torch.testing.assert_close(combined[indices], (joint_keys-joint_query).square().sum(1), rtol=1e-5, atol=1e-6)

        backend = "torch"
        if hasattr(resources.engine.retriever, "runtime"):
            native = SimpleNamespace(memory=memory, scales=scales, layout=layout,
                                     keys={"forward": keys}, runtime=resources.engine.retriever.runtime)
            native_search = GoalContextSearch(ContextSearch(native, 3), fragments, settings)
            torch.testing.assert_close(native_search.distances(context, goal)[indices], combined[indices], rtol=1e-4, atol=1e-6)
            assert native_search.calibration == search.calibration
            backend = "cuvs"

        provider = FragmentProvider(fragments, context_search, settings, search)
        metadata = {"source_episode_position": 1}
        positive = provider.retrieve(context, metadata, "goal-context-fragments", 19, goal)
        negative = provider.retrieve(context, metadata, "goal-context-fragments", 19, -goal)
        assert positive.sources[0]["memory_index"] == 0
        assert negative.sources[0]["memory_index"] == 1
        assert all(row["source_episode_position"] == 0 for row in positive.sources+negative.sources)
        torch.testing.assert_close(positive.actions[0], actions[2:8].reshape(3, 4), rtol=0, atol=0)
        assert positive.sources[0]["fragment_final_observation_row"] == 8
        plain = provider.retrieve(context, metadata, "context-fragments", 19)
        irrelevant_goal = provider.retrieve(context, metadata, "context-fragments", 19, -goal)
        assert plain.fingerprint == irrelevant_goal.fingerprint
        zero = GoalContextSearch(context_search, fragments, {**settings, "goal_weight": 0.}, directory)
        assert zero.calibration == search.calibration  # Weight changes do not recalibrate endpoints.
        torch.testing.assert_close(zero.distances(context, goal), context_search.distances(context), rtol=0, atol=0)
        zero_provider = FragmentProvider(fragments, context_search, settings, zero)
        zero_fragments = zero_provider.retrieve(context, metadata, "goal-context-fragments", 19, goal)
        assert [r["memory_index"] for r in zero_fragments.sources] == [r["memory_index"] for r in plain.sources]
        assert build_injection(zero_fragments, settings, 23).fingerprint == build_injection(plain, settings, 23).fingerprint

        short_store = FragmentStore(memory, 1, coordinates)
        short = GoalContextSearch(context_search, short_store, settings, directory)
        assert short.calibration_path != search.calibration_path
        assert short.store.final_rows[0] == 4
        assert float(short.endpoint_distances(goal, [0])[0]) > 0
        assert float(search.endpoint_distances(goal, [0])[0]) == 0
        # Changing held-out latents cannot change memory-only endpoint calibration.
        latents[40:] = 99
        assert GoalContextSearch(context_search, fragments, settings).calibration == search.calibration
        for invalid_goal in (None, torch.zeros(2, 3, device=device), torch.full((1, 3), float("nan"), device=device)):
            try:
                search.distances(context, invalid_goal)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing, malformed or nonfinite retrieval goal accepted")
        try:
            search.endpoint_distances(goal, [4])
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-episode memory endpoint accepted")
        damaged = read_json(search.calibration_path)
        damaged["scale"] += 1
        write_json(search.calibration_path, damaged)
        try:
            GoalContextSearch(context_search, fragments, settings, directory)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed goal calibration checksum accepted")

    return {"backend": backend, "weighted_concatenation_parity": True, "exact_full_corpus_metric": True,
            "goal_changes_fragment_selection": True, "horizon_specific_endpoints_and_calibration": True,
            "query_episode_exclusion_and_fragment_alignment": True, "boundary_endpoints_never_gathered": True,
            "zero_weight_context_and_injection_parity": True, "context_source_ignores_supplied_goal": True,
            "calibration_is_memory_only_repeatable_and_checksummed": True, "invalid_goals_rejected": True}
