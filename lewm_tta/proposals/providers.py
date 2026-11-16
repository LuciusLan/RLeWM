"""Compose action selection and noise; supplied goals enter only the explicit goal source."""

import time
from pathlib import Path

from ..planning.actions import stream_seed
from ..runtime import object_hash
from ..rollout.support_updates import synchronize
from .context_search import ContextSearch
from .fragments import FragmentStore, FragmentProvider
from .injection import build_injection
from .config import goal_settings
from .goal_search import GoalContextSearch


def make_provider(resources, config, coordinates):
    search = ContextSearch(resources.engine.retriever, config["proposals"]["context_chunk_rows"])
    store = FragmentStore(resources.memory, config["planning"]["horizon"], coordinates)
    goal = (GoalContextSearch(search, store, goal_settings(config), Path(config["experiment_dir"])/"retrieval")
            if "goal-context-fragments" in config["proposals"]["sources"] and config["proposals"]["injection_slots"] else None)
    return FragmentProvider(store, search, config["proposals"], goal)


def proposal_bundle(provider, adapter, context, metadata, source, seed, goal=None):
    settings = provider.settings
    if source == "gaussian" or settings["injection_slots"] == 0:
        return None, {"source": source, "enabled": False, "selection_seconds": 0.0, "generation_seconds": 0.0,
                      "total_seconds": 0.0, "fragments": [], "fragment_fingerprint": None,
                      "injection_fingerprint": None, "injected_slots": 0, "requested_slots": 0,
                      "slot_provenance": [], "diagnostics": {"fragment_shortfall": 0}}
    selection_seed = stream_seed(seed, metadata["transition_id"], "action-fragment-selection")
    noise_seed = stream_seed(seed, metadata["transition_id"], "action-fragment-noise")
    synchronize(adapter)
    started = time.perf_counter()
    fragments = provider.retrieve(context, metadata, source, selection_seed,
                                  goal=goal if source == "goal-context-fragments" else None)
    synchronize(adapter)
    selected = time.perf_counter()
    injection = build_injection(fragments, settings, noise_seed)
    synchronize(adapter)
    finished = time.perf_counter()
    return injection, {"source": source, "enabled": True,
        "goal_query_fingerprint": (object_hash(provider.goal_search.checked_goal(goal).cpu().tolist())
                                   if source == "goal-context-fragments" else None),
        "selection_seed": selection_seed, "noise_seed": noise_seed,
        "selection_seconds": selected-started, "generation_seconds": finished-selected,
        "total_seconds": finished-started, "fragments": fragments.sources,
        "fragment_actions_normalized": fragments.actions.cpu().tolist(),
        "fragment_fingerprint": fragments.fingerprint, "injection_fingerprint": injection.fingerprint,
        "injected_slots": len(injection.slots), "requested_slots": injection.requested_slots,
        "slot_provenance": injection.provenance, "diagnostics": fragments.diagnostics}
