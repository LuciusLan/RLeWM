"""One shared observed-memory adaptation phase before a complete CEM call.

Goals, recorded future actions, and CEM candidates are absent from this API.
Supports are balanced over independent provisional actions, then fixed for U
updates. The global unique budget and episode limits apply after merging.
"""

import time
from itertools import islice
import torch

from ..model import Context
from ..retrieval import Support, diverse_prefix
from ..runtime import object_hash
from ..rollout.support_updates import observed_support_step, synchronize, validate_support
from .actions import stream_seed


def merge_streams(streams, budget, retriever):
    """Round-robin quotas, deduplicated union, then fill vacant unique slots.

Each stream has (label, order, mass). Shared IDs retain contributions from
each stream. Quotas balance action/channel memberships; overlap does not buy
additional training presentations. Empty streams have their mass redistributed
proportionally over available streams and are explicitly reported.
"""
    selected, counts, steps = [], {}, {}
    memberships = {name: [] for name, _, _ in streams}
    iterators = {name: iter(order) for name, order, _ in streams}
    exhausted = set()

    def take(name, unique=False):
        if name in exhausted:
            return False
        for value in iterators[name]:
            index = int(value)
            if index in memberships[name] or (unique and index in selected):
                continue
            if index not in selected:
                episode = int(retriever.episode_positions[index])
                step = int(retriever.step_positions[index])
                if retriever.episode_cap and counts.get(episode, 0) >= retriever.episode_cap:
                    continue
                if any(abs(step-previous) < retriever.min_step_gap for previous in steps.get(episode, [])):
                    continue
                selected.append(index)
                counts[episode] = counts.get(episode, 0)+1
                steps.setdefault(episode, []).append(step)
            memberships[name].append(index)
            return True
        exhausted.add(name)
        return False

    quotas = {name: budget//len(streams)+(position < budget % len(streams))
              for position, (name, _, _) in enumerate(streams)}
    for round_index in range(max(quotas.values(), default=0)):
        for name, _, _ in streams:
            if round_index < quotas[name]:
                take(name)
    while len(selected) < budget and len(exhausted) < len(streams):
        for name, _, _ in streams:
            if len(selected) == budget:
                break
            take(name, unique=True)
    mass = sum(weight for name, _, weight in streams if memberships[name])
    weights = [sum(weight/len(memberships[name]) for name, _, weight in streams
                   if index in memberships[name])/mass for index in selected] if mass else []
    channels = {index: [name for name, _, _ in streams if index in memberships[name]] for index in selected}
    return Support(selected, weights, channels, {
        "rule": "balanced stream quotas, deduplicated union, round-robin fill; normalized per-stream loss",
        "requested_unique_support": budget, "shortfall": budget-len(selected),
        "streams": {name: {"quota": quotas[name], "memberships": len(memberships[name]),
                            "requested_loss_mass": weight,
                            "effective_loss_mass": weight/mass if memberships[name] else 0.0}
                    for name, _, weight in streams},
        "empty_streams": [name for name in memberships if not memberships[name]]})


def select_shared_support(engine, context, actions, method, seed, scan_limit):
    retriever, budget = engine.retriever, engine.settings["support_size"]
    if method == "uniform":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(engine.memory), generator=generator).numpy()
        selected = diverse_prefix(order, budget, retriever.episode_positions, retriever.episode_cap,
                                  retriever.step_positions, retriever.min_step_gap)
        return (Support(selected, [1/len(selected)]*len(selected) if selected else [],
                        {index: ["uniform"] for index in selected},
                        {"rule": "uniform permutation of memory, then global diversity constraints",
                         "shortfall": budget-len(selected)}),
                {"forward": 0, "reverse": 0}, 0)
    alpha = engine.settings["alpha"]
    directions = ({"forward": 1.0} if method == "forward" else {"reverse": 1.0}
                  if method == "reverse" else {"forward": 1-alpha, "reverse": alpha})
    directions = {name: mass for name, mass in directions.items() if mass > 0}
    predictions = None
    if "reverse" in directions:
        repeated = Context(context.latents.expand(len(actions), -1, -1),
                           context.past_actions.expand(len(actions), -1, -1))
        predictions = engine.adapter.predict(repeated, actions).detach()
    streams, rankings, calls = [], {}, {"forward": 0, "reverse": 0}
    for position in range(len(actions)):
        for direction, mass in directions.items():
            ranking = (retriever.forward(context, actions[position:position+1]) if direction == "forward"
                       else retriever.reverse(context, predictions[position:position+1]))
            name = f"action-{position}/{direction}"
            # Lazy cuVS prefixes remain exact; bound total eligibility work explicitly.
            order = islice(ranking.order, min(scan_limit, len(ranking)))
            streams.append((name, order, mass/len(actions)))
            rankings[name] = ranking
            calls[direction] += 1
    support = merge_streams(streams, budget, retriever)
    support.details["scan_limit_per_stream"] = scan_limit
    support.details["selected_distances"] = {
        str(index): {name: float(rankings[name].distances[index]) for name in support.channels[index]}
        for index in support.indices}
    return support, calls, len(actions) if predictions is not None else 0


def adapt_shared(engine, context, metadata, actions, method, seed, scan_limit):
    adapter, memory, retriever = engine.adapter, engine.memory, engine.retriever
    if method not in ("frozen", "uniform", "forward", "reverse", "fr-fixed"):
        raise ValueError("Unknown shared-TTT selection method")
    if memory.split != "memory" or retriever.memory is not memory:
        raise ValueError("Only the shared observed-memory bank may train the model")
    if metadata["source_episode_position"] in retriever.episode_set:
        raise ValueError("Query episode appears in TTT memory")
    synchronize(adapter)
    started = time.perf_counter()
    reset_seed = stream_seed(seed, metadata["transition_id"], "model-reset")
    adapter.reset(reset_seed)
    synchronize(adapter)
    reset_seconds = time.perf_counter()-started
    support = None
    rounds, calls, predictions = [], {"forward": 0, "reverse": 0}, 0
    selection_seconds = gradient_seconds = support_evaluation_seconds = 0.0
    updates = 0 if method == "frozen" else engine.settings["updates"]
    if updates and len(memory):
        selection_started = time.perf_counter()
        support, calls, predictions = select_shared_support(
            engine, context, actions, method, stream_seed(seed, metadata["transition_id"], "support"), scan_limit)
        synchronize(adapter)
        selection_seconds = time.perf_counter()-selection_started
        if support.indices:
            validate_support(memory, retriever, support, engine.settings["support_size"])
            optimizer = torch.optim.AdamW(adapter.parameters, lr=engine.settings["learning_rate"], weight_decay=0)
            for update in range(updates):
                values = observed_support_step(adapter, memory, support, optimizer, engine.settings)
                gradient_seconds += values.pop("gradient_seconds")
                support_evaluation_seconds += values.pop("support_evaluation_seconds")
                steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
                if not steps or any(step != update+1 for step in steps):
                    raise AssertionError("Shared TTT optimizer clock mismatch")
                rounds.append({"update": update+1, **values})
    adapter.check_fixed_state()
    synchronize(adapter)
    items = [] if support is None else [
        {**memory.items[index], "weight": support.weights[position], "channels": support.channels[index]}
        for position, index in enumerate(support.indices)]
    return {"reset_seed": reset_seed, "updates": len(rounds), "rounds": rounds, "support": items,
            "selection": support.details if support else {}, "search_calls": calls,
            "provisional_successor_predictions": predictions,
            "support_fingerprint": object_hash(items), "unique_support": len(items),
            "support_presentations": len(items)*len(rounds),
            "extra_support_evaluation_presentations": len(items)*len(rounds),
            "parameter_displacement_mse": float(adapter.anchor_loss().detach()),
            "skip_reason": ("empty_memory" if updates and not len(memory) else
                            "no_eligible_support" if updates and not rounds else None),
            "timing_seconds": {"reset": reset_seconds, "selection_and_provisional_prediction": selection_seconds,
                               "gradient_updates": gradient_seconds, "support_loss_evaluation": support_evaluation_seconds,
                               "total_with_checks": time.perf_counter()-started}}
