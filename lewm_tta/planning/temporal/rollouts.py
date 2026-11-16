"""Build reference-conditioned depth models and run isolated P2-style candidates.

No goal, recorded future action, or evaluator target is accepted by this API.
All gradient targets come from the existing observed-memory support operation.
"""

from itertools import islice
from types import SimpleNamespace

import torch

from ...model import Context
from ...runtime import object_hash
from ...rollout.support_updates import observed_support_step, validate_support
from ...rollout.temporal import AdaptationSession
from ..actions import provisional_actions, stream_seed
from ..shared_ttt import merge_streams, select_shared_support
from .state import ParameterSnapshot, advance


def reference_actions(coordinates, config, metadata, seed):
    settings = config["planning"]
    # Depth zero exactly matches the existing P3 provisional action stream.
    blocks = [provisional_actions(coordinates, settings["provisional_actions"], settings["initial_std"],
              stream_seed(seed, metadata["transition_id"], "provisional" if h == 0 else f"depth-reference-{h}"))
              for h in range(settings["horizon"])]
    return torch.stack(blocks, dim=1)


def select_depth_support(engine, contexts, actions, method, seed, scan_limit, depth):
    if depth == 0 or method == "uniform":
        # Existing merger and streams guarantee first-depth/initial-only parity.
        initial = Context(contexts.latents[:1], contexts.past_actions[:1])
        return select_shared_support(engine, initial, actions, method, seed, scan_limit)
    alpha, count = engine.settings["alpha"], len(actions)
    directions = ({"forward": 1.} if method == "forward" else {"reverse": 1.}
                  if method == "reverse" else {"forward": 1-alpha, "reverse": alpha})
    directions = {name: mass for name, mass in directions.items() if mass > 0}
    predicted = engine.adapter.predict(contexts, actions).detach() if "reverse" in directions else None
    streams, calls = [], {"forward": 0, "reverse": 0}
    for index in range(count):
        context = Context(contexts.latents[index:index+1], contexts.past_actions[index:index+1])
        for direction, mass in directions.items():
            ranking = (engine.retriever.forward(context, actions[index:index+1]) if direction == "forward"
                       else engine.retriever.reverse(context, predicted[index:index+1]))
            streams.append((f"action-{index}/{direction}", islice(ranking.order, min(scan_limit, len(ranking))), mass/count))
            calls[direction] += 1
    support = merge_streams(streams, engine.settings["support_size"], engine.retriever)
    support.details["scan_limit_per_stream"] = scan_limit
    return support, calls, count if predicted is not None else 0


def build_depth_models(engine, context, metadata, references, method, seed, config, counters):
    adapter, settings = engine.adapter, engine.settings
    reset_seed = stream_seed(seed, metadata["transition_id"], "model-reset")
    adapter.reset(reset_seed)
    counters["reset_count"] += 1
    optimizer = torch.optim.AdamW(adapter.parameters, lr=settings["learning_rate"], weight_decay=0)
    contexts = Context(context.latents.expand(len(references), -1, -1).detach(),
                       context.past_actions.expand(len(references), -1, -1).detach())
    snapshots, positions = [], []
    for depth, actions in enumerate(references.unbind(1)):
        if depth and config["temporal"]["persistence"] == "position":
            adapter.reset(reset_seed)
            optimizer = torch.optim.AdamW(adapter.parameters, lr=settings["learning_rate"], weight_decay=0)
            counters["reset_count"] += 1
        support, calls, rounds = None, {"forward": 0, "reverse": 0}, []
        requested = settings["updates"]
        counters["requested_updates"] += requested
        if requested:
            counters["support_positions"] += 1
            if len(engine.memory):
                support, calls, _ = select_depth_support(engine, contexts, actions, method,
                    stream_seed(seed, metadata["transition_id"], "support" if depth == 0 else f"depth-support-{depth}"),
                    config["planning"]["support_scan_limit"], depth)
        size = len(support.indices) if support is not None else 0
        if requested and size < settings["support_size"]:
            counters["support_shortfall_positions"] += 1
        counters["forward_retrievals"] += calls["forward"]
        counters["reverse_retrievals"] += calls["reverse"]
        if size:
            validate_support(engine.memory, engine.retriever, support, settings["support_size"])
            for update in range(requested):
                # CEM callers can be under no_grad; the entire loss needs a graph.
                with torch.enable_grad():
                    values = observed_support_step(adapter, engine.memory, support, optimizer, settings)
                counters["actual_updates"] += 1
                counters["support_presentations"] += size
                rounds.append({key: value for key, value in values.items()
                               if key not in ("losses_before", "losses_after", "module_updates")})
        items = [] if not size else [{**engine.memory.items[index], "weight": support.weights[j],
                                      "channels": support.channels[index]}
                                     for j, index in enumerate(support.indices)]
        snapshot = ParameterSnapshot(adapter)
        snapshots.append(snapshot)
        positions.append({"depth": depth, "requested_updates": requested, "actual_updates": len(rounds),
                          "support": items, "support_fingerprint": object_hash(items), "rounds": rounds,
                          "model_fingerprint": snapshot.fingerprint,
                          "reset_before_position": depth == 0 or config["temporal"]["persistence"] == "position",
                          "skip_reason": "no_eligible_support" if requested and not size else None})
        with torch.no_grad():
            prediction = adapter.predict(contexts, actions).detach()
        contexts = advance(contexts, actions, prediction)
        counters["reference_transition_predictions"] += len(references)
    adapter.check_fixed_state()
    return snapshots, {"rounds": [], "positions": positions,
                       "reference_fingerprint": object_hash(references.cpu().tolist()),
                       "model_fingerprints": [snapshot.fingerprint for snapshot in snapshots]}


def candidate_rollout(engine, context, metadata, actions, method, seed, config, counters):
    # Same seed for the same query at every invocation: independent of candidate
    # ordering, CEM iteration, batch size, and previous candidates' model states.
    query = SimpleNamespace(metadata=metadata)
    schedule = {"placement": "each-step", "updates": engine.settings["updates"],
                "persistence": config["temporal"]["persistence"]}
    session = AdaptationSession(engine, query, seed, schedule)
    current = Context(context.latents.detach(), context.past_actions.detach())
    predictions, audit = [], []
    for depth, action in enumerate(actions.unbind(1)):
        if depth and schedule["persistence"] == "position":
            session.reset()
        with torch.enable_grad():
            position = session.adapt_position(current, action, method, schedule["updates"], depth)
        prediction = position.pop("prediction_tensor").detach()
        local = session.rounds[position["round_start"]:position["round_end"]]
        audit.append({"depth": depth, "requested_updates": position["requested_updates"],
                      "actual_updates": position["actual_updates"],
                      "optimizer_steps_before": position["optimizer_steps_before"],
                      "optimizer_steps_after": position["optimizer_steps_after"],
                      "support": [row["support"] for row in local],
                      "skip_reason": position["skip_reason"]})
        predictions.append(prediction)
        current = advance(current, action, prediction)
    counters["requested_updates"] += actions.shape[1]*schedule["updates"]
    counters["actual_updates"] += len(session.rounds)
    counters["support_presentations"] += session.presentations
    counters["support_positions"] += actions.shape[1] if schedule["updates"] else 0
    counters["support_shortfall_positions"] += sum(
        position["requested_updates"] and (not position["support"] or
        any(len(items) < engine.settings["support_size"] for items in position["support"])) for position in audit)
    counters["forward_retrievals"] += session.search_calls["forward"]
    counters["reverse_retrievals"] += session.search_calls["reverse"]
    counters["reset_count"] += session.resets
    counters["candidate_trajectories"] += 1
    counters["candidate_unique_support_sum"] += len(session.cumulative)
    engine.adapter.check_fixed_state()
    return torch.stack(predictions, 1), audit
