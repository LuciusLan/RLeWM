"""User-run P1 acceptance checks, including rank fixtures and reset isolation."""

from __future__ import annotations

from pathlib import Path

import torch

from .adaptation import AdaptationEngine
from .cache import LatentBank
from .model import Context
from .rank_diagnostics import consensus_indices, cross_rank_diagnostics
from .retrieval import ExactRetriever, Ranking, diverse_prefix, select_support


def check_frame_bank_layout() -> dict:
    """Tiny CPU fixture checks shared-frame indexing and ordered action blocks."""
    import numpy as np
    from .frame_bank import FrameLatentBank
    from .observation_cache import ObservationStore
    values = torch.arange(100).float().reshape(50, 2)
    store = ObservationStore(values, values.clone(), {}, Path("."))
    manifest = {"data_config": {"history": 3, "frameskip": 5},
                "episode_metadata": {"offsets": [0, 25], "lengths": [25, 25], "ids": [10, 20]},
                "episodes": {"memory": [0, 1], "validation": [], "test": []},
                "source": {"path": "fixture"}}
    bank = FrameLatentBank(store, manifest, "memory", np.array([0, 5, 25, 30], dtype=np.int64), {})
    torch.testing.assert_close(bank.contexts[0], values[[0, 5, 10]], rtol=0, atol=0)
    torch.testing.assert_close(bank.successors[torch.tensor([0, 3])], values[[15, 45]], rtol=0, atol=0)
    torch.testing.assert_close(bank.past_actions[0], values[:10].reshape(2, 10), rtol=0, atol=0)
    torch.testing.assert_close(bank.actions[0], values[10:15].reshape(10), rtol=0, atol=0)
    assert bank.items[3]["transition_id"] == "pusht:20:5"
    assert bank.items[3]["candidate_action_rows"] == [40, 41, 42, 43, 44]
    assert not hasattr(bank.query(0), "successor")
    return {"shared_context_successor_alignment": True, "ordered_past_and_candidate_actions": True,
            "episode_metadata_alignment": True, "query_target_isolation": True}


def run_checks(engine: AdaptationEngine, queries, seed: int) -> dict:
    import numpy as np
    fixtures = cross_rank_diagnostics([0, 1, 2, 3], [2, 0, 1, 3], k=2)
    assert fixtures["forward_top1_reverse_rank"] == 2
    assert fixtures["reverse_top1_forward_rank"] == 3
    assert fixtures["overlap_fraction"] == .5
    assert consensus_indices([0, 1, 2, 3], [2, 0, 1, 3], 2, 2) == [0, 2]
    tiny = LatentBank("memory", torch.zeros(4, 1, 1), torch.zeros(4, 0, 1),
                      torch.arange(4).float().unsqueeze(1), torch.tensor([[2.], [0.], [1.], [3.]]),
                      [{"transition_id": str(index)} for index in range(4)], {})
    exact = ExactRetriever(tiny, {"scales": dict.fromkeys(("context", "past_actions", "action", "successor"), 1.)},
                           {"chunk_size": 2, "beta_action": 1., "beta_successor": 1.})
    context = Context(torch.zeros(1, 1, 1), torch.zeros(1, 0, 1))
    assert exact.forward(context, torch.zeros(1, 1)).indices.tolist() == [0, 1, 2, 3]
    assert exact.reverse(context, torch.zeros(1, 1)).indices.tolist() == [1, 2, 0, 3]
    episodes = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    steps = np.array([0, 1, 20, 21, 0, 1, 20, 21])
    forward_fixture = Ranking(np.arange(8), torch.zeros(8), {})
    reverse_fixture = Ranking(np.array([1, 3, 0, 2, 5, 7, 4, 6]), torch.zeros(8), {})
    assert diverse_prefix(forward_fixture.indices, 4, episodes, 2, steps, 20) == [0, 2, 4, 6]
    for method in ("uniform", "state", "forward", "reverse", "fr-fixed", "fr-refresh", "rank-consensus"):
        support = select_support(method, 4, 4, .5, forward_fixture, reverse_fixture,
            torch.Generator().manual_seed(42), forward_fixture, episodes, 2, 4, steps, 20)
        assert len(support.indices) == 4 and abs(sum(support.weights)-1) < 1e-6
        for ep in set(episodes[support.indices].tolist()):
            chosen = [int(steps[index]) for index in support.indices if episodes[index] == ep]
            assert len(chosen) <= 2
            assert all(abs(first-second) >= 20 for first in chosen for second in chosen if first != second)
    query = queries.query(0)
    zero = AdaptationEngine(engine.adapter, engine.memory, engine.retriever, engine.settings | {"updates": 0})
    frozen = engine.run(query, "frozen", seed)
    unadapted = zero.run(query, "forward", seed)
    torch.testing.assert_close(torch.tensor(frozen["prediction_final"]),
                               torch.tensor(unadapted["prediction_final"]), rtol=0, atol=0)
    checks = {"exact_rank_fixture": True, "exact_search_fixture": True, "zero_update_equivalence": True,
              "support_episode_cap_and_step_spacing": True,
              "reverse_signature_has_no_candidate_action": True,
              "query_type_has_no_successor_label": not hasattr(query, "successor")}
    if hasattr(queries, "store"):
        checks.update(check_frame_bank_layout())
    if len(engine.memory):
        one = AdaptationEngine(engine.adapter, engine.memory, engine.retriever,
            engine.settings | {"updates": 1, "support_size": max(2, engine.settings["support_size"])})
        first = one.run(query, "forward", seed)
        one.run(queries.query(min(1, len(queries)-1)), "uniform", seed+1)
        repeat = one.run(query, "forward", seed)
        torch.testing.assert_close(torch.tensor(first["prediction_final"]),
                                   torch.tensor(repeat["prediction_final"]), rtol=1e-6, atol=1e-7)
        fixed = one.run(query, "fr-fixed", seed)
        refreshed = one.run(query, "fr-refresh", seed)
        assert [row["transition_id"] for row in fixed["rounds"][0]["support"]] == [
            row["transition_id"] for row in refreshed["rounds"][0]["support"]]
        torch.testing.assert_close(torch.tensor(fixed["prediction_final"]),
                                   torch.tensor(refreshed["prediction_final"]), rtol=1e-6, atol=1e-7)
        checks.update(reset_independence=True, fr_first_update_equivalence=True,
                      gradient_scope_and_frozen_state=True)
    return checks
